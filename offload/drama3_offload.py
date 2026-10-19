"""MoE expert offload for Drama 3 on GPUs with less than ~45 GB of VRAM.

The Space's engine moves the whole checkpoint to one GPU: ~6 GiB of dense weights,
1.7 GiB of codec and 30 GiB of block-FP8 routed experts (40 layers x 256 experts x
3 MiB). This module keeps the experts of some layers in pinned host RAM instead.

How the host-resident experts are reached, without changing the model's numerics:

- Each offloaded weight is a CUDA tensor that *aliases* pinned host memory (UVA,
  zero-copy). torch.compile, the Triton FP8 kernels and CUDA-graph capture all see an
  ordinary CUDA tensor; reads travel over PCIe.
- Decode routes one token to 8 experts. Read in place, the FP8 GEMM keeps too few
  requests in flight and gets ~6 GiB/s, so a decode step first gathers the routed
  experts into a small device buffer with a wide copy kernel (~50 GiB/s on PCIe 5
  x16), then runs the unchanged GEMM on that buffer. The routing never leaves the GPU,
  so the step stays capturable as a CUDA graph.
- Prefill (hundreds of tokens, every expert hit) reads the aliased weights in place,
  which already runs at ~40 GiB/s.

The same kernels run on the same bytes, so results are identical to the
all-on-GPU path. Install with `install()` before the engine loads its weights.

Environment:
  DRAMA3_OFFLOAD_LAYERS   "auto" (default) or how many layers keep their experts in RAM
                          (0 = upstream behaviour, 40 = all of them)
  DRAMA3_VRAM_RESERVE_GB  VRAM left free for the KV cache, CUDA graphs, activations
                          and the desktop when sizing "auto" (default 12: ~8.7 GiB measured peak with a
                          60 s reference, a 2-take directing history and the 4096-row decode graph
                          compiled, plus ~3 GiB for the desktop)
"""

from __future__ import annotations

import json
import os
import time

import torch
import triton
import triton.language as tl

# (layers whose experts are in host RAM, expert layers), for the UI badge
PLACEMENT: tuple[int, int] = (0, 0)

GATHER_MAX_PAIRS = 64  # (token, expert) pairs up to which a step gathers its experts
GATHER_BLOCK = 1024  # int64 words per program of the gather kernel


def log(message: str) -> None:
    print(f"[offload] {message}", flush=True)


# ─── Host-aliased tensors ─────────────────────────────────────────────────────


class _CudaArrayInterface:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.__cuda_array_interface__ = {
            "shape": (tensor.numel(),),
            "typestr": "|u1",
            "data": (tensor.data_ptr(), False),
            "version": 3,
            "strides": None,
        }


def host_alias(cpu_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A CUDA tensor over a pinned host copy of cpu_tensor (no VRAM is used)."""
    pinned = cpu_tensor.contiguous().pin_memory()
    with torch.cuda.device(device):
        flat = torch.as_tensor(_CudaArrayInterface(pinned.view(-1).view(torch.uint8)), device=device)
    alias = flat.view(cpu_tensor.dtype).view(cpu_tensor.shape)
    if alias.data_ptr() != pinned.data_ptr():
        raise RuntimeError("pinned memory is not mapped at the same address on the GPU (no UVA?)")
    alias._drama3_host_storage = pinned  # keeps the pinned allocation alive
    return alias


# ─── Gather kernel ────────────────────────────────────────────────────────────


@triton.jit
def gather_experts_kernel(host_ptr, cache_ptr, ids_ptr, slots_ptr, dst_ptr, row_words, BLOCK: tl.constexpr):
    """dst[pair] = expert ids[pair]'s row: from the VRAM cache if it holds that expert
    (slots[expert] >= 0), else from host RAM. One row of int64 words per pair."""
    pair = tl.program_id(0)
    chunk = tl.program_id(1)
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    slot = tl.load(slots_ptr + expert).to(tl.int64)
    offs = chunk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_words
    if slot >= 0:
        words = tl.load(cache_ptr + slot * row_words + offs, mask=mask)
    else:
        words = tl.load(host_ptr + expert * row_words + offs, mask=mask)
    tl.store(dst_ptr + pair.to(tl.int64) * row_words + offs, words, mask=mask)


def _words(weight: torch.Tensor) -> torch.Tensor:
    return weight.view(torch.uint8).view(weight.shape[0], -1).view(torch.int64)


def gather_experts(weight: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """weight[ids] for a host-aliased [E, rows, cols] FP8 tensor, into device memory,
    reading the experts the cache holds from VRAM instead."""
    src, cached = _words(weight), _words(cache)
    row_words = src.shape[1]
    dst = torch.empty((ids.numel(), row_words), dtype=torch.int64, device=ids.device)
    with torch.cuda.device(ids.device):
        gather_experts_kernel[(ids.numel(), triton.cdiv(row_words, GATHER_BLOCK))](
            src, cached, ids, slots, dst, row_words, BLOCK=GATHER_BLOCK
        )
    return dst.view(torch.uint8).view(weight.dtype).view(ids.numel(), *weight.shape[1:])


# ─── Patched expert forward ───────────────────────────────────────────────────


# Kept out of torch.compile: it is Triton kernels already, and each layer's cache has its own
# size, which would recompile every layer for every decode graph (past the recompile limit)
@torch.compiler.disable
def offloaded_experts_forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward

    pairs = topk_ids.numel()
    if pairs > GATHER_MAX_PAIRS:
        # Prefill: nearly every expert is hit; reading in place is already bandwidth-bound
        # (running the cached experts from VRAM instead measured no faster: 19 ms either way).
        return fp8_moe_forward(
            x, topk_ids, topk_weights,
            self.w13_weight, self.w13_scale_inv, self.w2_weight, self.w2_scale_inv,
        )
    flat = topk_ids.reshape(-1).contiguous()
    # Pair p uses staged expert p; the GEMM, its scales and routing weights are unchanged.
    local_ids = torch.arange(pairs, device=flat.device).view_as(topk_ids)
    return fp8_moe_forward(
        x, local_ids, topk_weights,
        gather_experts(self.w13_weight, self._drama3_cache["w13_weight"], self._drama3_slots, flat),
        self.w13_scale_inv[flat],
        gather_experts(self.w2_weight, self._drama3_cache["w2_weight"], self._drama3_slots, flat),
        self.w2_scale_inv[flat],
    )


# ─── Placement ────────────────────────────────────────────────────────────────
#
# Decode routes each token to 8 of a layer's 256 experts, and not evenly: measured over
# 8 takes (clone, Surprise, whispers, anger), a layer's 64 most-used experts took 66% of
# its routings. So VRAM holds the most-used experts of every layer (a cache, ranked by
# offload/expert_profile.json) instead of all the experts of some layers: with the same
# 5 GiB, 44% of the expert reads of other takes came from VRAM instead of 17%. Every
# layer keeps all its experts in pinned RAM too; a layer whose experts all fit is
# simply a GPU layer. Without the profile, whole layers are placed as before.

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_profile.json")
NAMES = ("w13_weight", "w2_weight")


def tensor_bytes(module: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))


def offloadable_bytes(experts: torch.nn.Module) -> int:
    """The FP8 weights that move to host RAM; their small fp32 scales stay on the GPU."""
    return sum(experts._buffers[name].numel() * experts._buffers[name].element_size() for name in NAMES)


# Startup facts the placement is recomputed from when the reference limit changes
STATE: dict = {"experts": [], "device": None, "budget": 0, "per_layer": 0, "fixed": False, "ranking": []}


def ref_unlocked_at_start() -> bool:
    return os.environ.get("DRAMA3_REF_UNLOCKED", "0").strip().lower() in {"1", "true", "yes", "on"}


def reserve_bytes(unlocked: bool) -> float:
    """VRAM auto leaves free: the engine's measured working memory + ~2.5-3 GiB for the desktop.

    Measured worst cases (2-take directing histories of long takes, long scripts, Surprise
    retakes): ~6.5 GiB above the weights with 20 s references, ~8.7 GiB with 60 s ones."""
    name, default = ("DRAMA3_VRAM_RESERVE_60S_GB", 12) if unlocked else ("DRAMA3_VRAM_RESERVE_GB", 9)
    return float(os.environ.get(name, default)) * 2**30


def ranking(count: int, experts: int, use_profile: bool) -> list[tuple[int, int]]:
    """(layer, expert) in the order they get VRAM: most-used first (by the profile),
    unused ones layer by layer; or, without a profile, whole layers from the first."""
    by_layer = [(layer, expert) for layer in range(count) for expert in range(experts)]
    if not use_profile:
        return by_layer
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            counts = json.load(f)["counts"]
        if len(counts) != count or any(len(row) != experts for row in counts):
            raise ValueError(f"shape {len(counts)} x {len(counts[0]) if counts else 0}")
    except (OSError, ValueError, KeyError) as error:
        log(f"no usable expert profile ({error}); placing whole layers")
        return by_layer
    return sorted(by_layer, key=lambda cell: -counts[cell[0]][cell[1]])  # stable: ties stay in layer order


def plan(unlocked: bool) -> list[list[int]]:
    """The experts each layer keeps in VRAM, most-used first, for a reference limit (20 s or 60 s)."""
    count, experts = len(STATE["experts"]), STATE["experts"][0].w13_scale_inv.shape[0]
    if STATE["fixed"] is not False:
        cells = (count - STATE["fixed"]) * experts
    else:
        per_expert = STATE["per_layer"] / experts
        cells = int(max(0, STATE["budget"] - reserve_bytes(unlocked)) // per_expert)
    hot: list[list[int]] = [[] for _ in range(count)]
    for layer, expert in STATE["ranking"][:cells]:
        hot[layer].append(expert)
    return hot


def describe(hot: list[list[int]]) -> tuple[int, int, int]:
    """(layers with every expert in VRAM, layers with some in RAM, experts cached in those)."""
    experts = STATE["experts"][0].w13_scale_inv.shape[0]
    full = sum(len(h) == experts for h in hot)
    return full, len(hot) - full, sum(len(h) for h in hot if len(h) < experts)


def size_budget(expert_modules: list, resident_bytes: int, device: torch.device) -> None:
    setting = os.environ.get("DRAMA3_OFFLOAD_LAYERS", "auto").strip().lower() or "auto"
    count = len(expert_modules)
    experts = expert_modules[0].w13_scale_inv.shape[0]
    STATE.update(experts=expert_modules, device=device, per_layer=offloadable_bytes(expert_modules[0]))
    if setting != "auto":
        try:
            STATE["fixed"] = max(0, min(count, int(setting)))
        except ValueError:
            raise ValueError(f"DRAMA3_OFFLOAD_LAYERS must be 'auto' or a number of layers (0-{count}); got {setting!r}") from None
    STATE["ranking"] = ranking(count, experts, STATE["fixed"] is False)
    free, total = torch.cuda.mem_get_info(device)
    STATE["budget"] = free - resident_bytes
    if STATE["fixed"] is not False:
        return
    for mode in (False, True):
        short = reserve_bytes(mode) - STATE["budget"]
        if short > 0:
            log(f"warning: even with every expert in RAM, free VRAM is {short / 2**30:.1f} GiB short of the "
                f"{'60' if mode else '20'} s reserve (working memory + ~2.5 GiB of desktop headroom): the longest "
                "prompts may run out of memory. Close other GPU apps or run without a desktop; lowering "
                "the reserve trades away that headroom.")
    unlocked = ref_unlocked_at_start()
    full, partial, cached = describe(plan(unlocked))
    other = describe(plan(not unlocked))
    log(f"{torch.cuda.get_device_name(device)}: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free; "
        f"dense weights + codec {resident_bytes / 2**30:.1f} GiB, reserve {reserve_bytes(unlocked) / 2**30:.1f} GiB "
        f"({'60' if unlocked else '20'} s references), experts {STATE['per_layer'] / 2**20:.0f} MiB per layer -> "
        f"{full} of {count} layers fit, plus the {cached} most-used experts of the other {partial} "
        f"(with {'20' if unlocked else '60'} s references: {other[0]} layers + {other[2]} experts)")


def pin(experts: torch.nn.Module, device: torch.device) -> None:
    """Keep one layer's FP8 experts in pinned RAM (aliased for the GPU)."""
    host = {}
    for name in NAMES:
        weight = experts._buffers[name]
        try:
            host[name] = host_alias(weight if weight.device.type == "cpu" else weight.cpu(), device)
        except RuntimeError as error:
            raise RuntimeError(
                f"could not pin RAM for the experts: {error}. They need 30 GiB of free RAM; close other apps."
            ) from error
        experts._buffers[name] = host[name]
    experts._drama3_host = host


def place(experts: torch.nn.Module, hot: list[int], device: torch.device) -> bool:
    """Give one layer its VRAM share: all its experts (an ordinary GPU layer) or a cache of
    the hot ones in front of its pinned RAM. False when it already has that share."""
    if getattr(experts, "_drama3_hot", None) == hot:
        return False
    host, count = experts._drama3_host, experts.w13_scale_inv.shape[0]
    for attribute in ("_drama3_cache", "_drama3_slots"):
        experts.__dict__.pop(attribute, None)
    if len(hot) == count:
        for name in NAMES:
            experts._buffers[name] = host[name].clone()  # allocates in VRAM, reads RAM over PCIe
        experts.__dict__.pop("forward", None)  # back to MoEExperts.forward (the FP8 kernel on VRAM)
    else:
        ids = torch.tensor(hot or [0], dtype=torch.long, device=device)  # a 1-row placeholder when none fit
        slots = torch.full((count,), -1, dtype=torch.int32, device=device)
        slots[ids[:len(hot)]] = torch.arange(len(hot), dtype=torch.int32, device=device)
        for name in NAMES:
            experts._buffers[name] = host[name]
        experts._drama3_cache = {name: host[name][ids] for name in NAMES}  # VRAM copies, read over PCIe
        experts._drama3_slots = slots
        experts.forward = offloaded_experts_forward.__get__(experts)
    experts._drama3_hot = hot
    return True


def apply(hot: list[list[int]]) -> int:
    """Place every layer; how many changed."""
    global PLACEMENT, PLACEMENT_LABEL
    device = STATE["device"]
    changed = sum(place(experts, layer_hot, device) for experts, layer_hot in zip(STATE["experts"], hot))
    full, partial, cached = describe(hot)
    PLACEMENT = (partial, len(hot))
    experts = STATE["experts"][0].w13_scale_inv.shape[0]
    PLACEMENT_LABEL = (f"{partial}/{len(hot)} expert layers in RAM" + (f", {cached / experts / max(1, partial):.0%} of "
                       "their experts cached in VRAM" if cached else "")) if partial else ""
    return changed


PLACEMENT_LABEL = ""


def offload(model: torch.nn.Module, codec: torch.nn.Module, device: torch.device) -> None:
    """Pin the experts in RAM and give the GPU its share (on CPU, before the model moves)."""
    expert_modules = [layer.mlp.experts for layer in model.backbone.model.layers
                      if getattr(layer.mlp, "experts", None) is not None and layer.mlp.experts.fp8]
    if not expert_modules:
        log("no FP8 expert layers found; nothing to offload")
        return
    expert_bytes = sum(offloadable_bytes(m) for m in expert_modules)
    # Everything that goes to the GPU whatever the plan: dense weights, scales, codec
    resident = tensor_bytes(model) + tensor_bytes(codec) - expert_bytes
    size_budget(expert_modules, resident, device)
    experts = expert_modules[0].w13_scale_inv.shape[0]
    hot = plan(ref_unlocked_at_start())
    if all(len(h) == experts for layout in (hot, plan(not ref_unlocked_at_start())) for h in layout):
        log("all experts fit on the GPU; offload off")
        STATE["experts"] = []  # nothing to rebalance
        return
    started = time.perf_counter()
    for module in expert_modules:
        pin(module, device)
    apply(hot)
    full, partial, cached = describe(hot)
    log(f"all {len(expert_modules)} layers keep their experts in pinned RAM ({expert_bytes / 2**30:.1f} GiB); "
        f"VRAM holds {full} whole layers and {cached} experts of the other {partial} "
        f"({(full * experts + cached) * STATE['per_layer'] / experts / 2**30:.1f} GiB, "
        f"placed in {time.perf_counter() - started:.0f}s)")


def rebalance(engine, unlocked: bool) -> str:
    """Re-place the experts for a new reference limit, between takes (caller holds the
    take lock). The decode CUDA graphs captured the experts' addresses, so they are
    dropped first (which also frees their memory before anything moves into VRAM)
    and captured again afterwards."""
    experts, device = STATE["experts"], STATE["device"]
    if not experts:
        return "no offload in use"
    hot = plan(unlocked)
    if all(getattr(e, "_drama3_hot", None) == h for e, h in zip(experts, hot)):
        return f"{PLACEMENT_LABEL or 'every expert on the GPU'} (unchanged)"
    started = time.perf_counter()
    with torch.cuda.device(device):
        if engine is not None:
            engine.graphs = None
            engine.model.decode_graphs = None
        torch.cuda.synchronize(device)
        # Shrink first, then grow: a layer losing VRAM frees it before another takes it
        order = sorted(range(len(experts)), key=lambda i: len(hot[i]) - len(getattr(experts[i], "_drama3_hot", [])))
        changed = 0
        for index in order:
            changed += place(experts[index], hot[index], device)
            torch.cuda.empty_cache()
        apply(hot)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        if engine is not None:
            import drama3_engine

            for bucket in drama3_engine.WARM_UP_BUCKETS:
                engine.graph_for(bucket)
    message = (f"{PLACEMENT_LABEL or 'every expert on the GPU'} ({changed} layers re-placed in "
               f"{time.perf_counter() - started:.0f}s)")
    log(message)
    return message


# ─── Installation ─────────────────────────────────────────────────────────────


def check_system(device: torch.device) -> None:
    """Fail early, with a readable message, where the Space's kernels cannot run."""
    if not torch.cuda.is_available():
        raise RuntimeError("Drama 3 needs an NVIDIA GPU with CUDA; none is visible to PyTorch")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"{torch.cuda.get_device_name(device)} (compute capability {major}.{minor}) has no FP8 tensor cores; "
            "Drama 3's FP8 experts need an RTX 40-series / Ada, Hopper or Blackwell GPU (8.9+)")
    try:  # The checkpoint is loaded into RAM before anything moves to the GPU
        with open("/proc/meminfo", encoding="ascii") as f:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
        usable = info["MemAvailable"] + info.get("SwapFree", 0)
        if usable < 40 * 2**30:
            log(f"warning: {usable / 2**30:.0f} GiB of RAM + swap available; loading Drama 3 needs about 40 GiB "
                "and may be killed by the OOM killer")
    except (OSError, KeyError, ValueError):
        pass



def install() -> None:
    """Wrap drama3_engine.load_weights: load on CPU, offload, then move to the GPU."""
    import drama3_engine

    original = drama3_engine.load_weights
    if getattr(original, "_drama3_offload", False):
        return

    def load_weights(checkpoint_dir=None, device: str = "cuda"):
        if device == "cpu":
            return original(checkpoint_dir, device="cpu")
        target = torch.device(device)
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        check_system(target)
        weights = original(checkpoint_dir, device="cpu")
        offload(weights.model, weights.codec, target)
        weights.model.to(target)
        weights.codec.to(target)
        log(f"weights on {target}: {torch.cuda.memory_allocated(target) / 2**30:.1f} GiB of VRAM allocated")
        return weights

    load_weights._drama3_offload = True
    drama3_engine.load_weights = load_weights
