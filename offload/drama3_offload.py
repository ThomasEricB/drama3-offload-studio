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
def gather_experts_kernel(src_ptr, ids_ptr, dst_ptr, row_words, BLOCK: tl.constexpr):
    """dst[pair] = src[ids[pair]], one expert row of int64 words per pair."""
    pair = tl.program_id(0)
    chunk = tl.program_id(1)
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    offs = chunk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_words
    words = tl.load(src_ptr + expert * row_words + offs, mask=mask)
    tl.store(dst_ptr + pair.to(tl.int64) * row_words + offs, words, mask=mask)


def gather_experts(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """weight[ids] for a (host-aliased) [E, rows, cols] FP8 tensor, into device memory."""
    experts = weight.shape[0]
    src = weight.view(torch.uint8).view(experts, -1).view(torch.int64)
    row_words = src.shape[1]
    dst = torch.empty((ids.numel(), row_words), dtype=torch.int64, device=ids.device)
    with torch.cuda.device(ids.device):
        gather_experts_kernel[(ids.numel(), triton.cdiv(row_words, GATHER_BLOCK))](
            src, ids, dst, row_words, BLOCK=GATHER_BLOCK
        )
    return dst.view(torch.uint8).view(weight.dtype).view(ids.numel(), *weight.shape[1:])


# ─── Patched expert forward ───────────────────────────────────────────────────


def offloaded_experts_forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward

    pairs = topk_ids.numel()
    if pairs > GATHER_MAX_PAIRS:
        # Prefill: nearly every expert is hit; reading in place is already bandwidth-bound.
        return fp8_moe_forward(
            x, topk_ids, topk_weights,
            self.w13_weight, self.w13_scale_inv, self.w2_weight, self.w2_scale_inv,
        )
    flat = topk_ids.reshape(-1).contiguous()
    # Pair p uses staged expert p; the GEMM, its scales and routing weights are unchanged.
    local_ids = torch.arange(pairs, device=flat.device).view_as(topk_ids)
    return fp8_moe_forward(
        x, local_ids, topk_weights,
        gather_experts(self.w13_weight, flat), self.w13_scale_inv[flat],
        gather_experts(self.w2_weight, flat), self.w2_scale_inv[flat],
    )


# ─── Placement ────────────────────────────────────────────────────────────────


def tensor_bytes(module: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))


def offloadable_bytes(experts: torch.nn.Module) -> int:
    """The FP8 weights that move to host RAM; their small fp32 scales stay on the GPU."""
    return sum(experts._buffers[name].numel() * experts._buffers[name].element_size()
               for name in ("w13_weight", "w2_weight"))


# Startup facts the placement is recomputed from when the reference limit changes
STATE: dict = {"experts": [], "device": None, "budget": 0, "per_layer": 0, "fixed": False}


def ref_unlocked_at_start() -> bool:
    return os.environ.get("DRAMA3_REF_UNLOCKED", "0").strip().lower() in {"1", "true", "yes", "on"}


def reserve_bytes(unlocked: bool) -> float:
    """VRAM auto leaves free: the engine's measured working memory + ~2.5-3 GiB for the desktop.

    Measured worst cases (2-take directing histories of long takes, long scripts, Surprise
    retakes): ~6.5 GiB above the weights with 20 s references, ~8.7 GiB with 60 s ones."""
    name, default = ("DRAMA3_VRAM_RESERVE_60S_GB", 12) if unlocked else ("DRAMA3_VRAM_RESERVE_GB", 9)
    return float(os.environ.get(name, default)) * 2**30


def layers_in_ram(unlocked: bool) -> int:
    """Expert layers kept in host RAM for a reference limit (20 s or 60 s)."""
    count = len(STATE["experts"])
    if STATE["fixed"] is not False:
        return STATE["fixed"]
    on_gpu = int((STATE["budget"] - reserve_bytes(unlocked)) // STATE["per_layer"])
    return count - max(0, min(count, on_gpu))


def plan_layers(expert_modules: list, resident_bytes: int, device: torch.device) -> int:
    """How many layers keep their experts in host RAM."""
    setting = os.environ.get("DRAMA3_OFFLOAD_LAYERS", "auto").strip().lower() or "auto"
    count = len(expert_modules)
    STATE.update(experts=expert_modules, device=device, per_layer=offloadable_bytes(expert_modules[0]))
    if setting != "auto":
        try:
            STATE["fixed"] = max(0, min(count, int(setting)))
        except ValueError:
            raise ValueError(f"DRAMA3_OFFLOAD_LAYERS must be 'auto' or a number of layers (0-{count}); got {setting!r}") from None
        return STATE["fixed"]
    free, total = torch.cuda.mem_get_info(device)
    STATE["budget"] = free - resident_bytes
    unlocked = ref_unlocked_at_start()
    in_ram = layers_in_ram(unlocked)
    for mode in (False, True):
        short = reserve_bytes(mode) - STATE["budget"]
        if short > 0:
            log(f"warning: even with every expert in RAM, free VRAM is {short / 2**30:.1f} GiB short of the "
                f"{'60' if mode else '20'} s reserve (working memory + ~2.5 GiB of desktop headroom): the longest "
                "prompts may run out of memory. Close other GPU apps or run without a desktop; lowering "
                "the reserve trades away that headroom.")
    log(f"{torch.cuda.get_device_name(device)}: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free; "
        f"dense weights + codec {resident_bytes / 2**30:.1f} GiB, reserve {reserve_bytes(unlocked) / 2**30:.1f} GiB "
        f"({'60' if unlocked else '20'} s references), experts {STATE['per_layer'] / 2**20:.0f} MiB per layer -> "
        f"{count - in_ram} of {count} layers fit (with {'20' if unlocked else '60'} s references: "
        f"{count - layers_in_ram(not unlocked)})")
    return in_ram


def to_ram(experts: torch.nn.Module, device: torch.device) -> int:
    """Move one layer's FP8 experts into pinned RAM (aliased for the GPU). Bytes moved."""
    moved = 0
    for name in ("w13_weight", "w2_weight"):
        weight = experts._buffers[name]
        source = weight if weight.device.type == "cpu" else weight.cpu()
        try:
            alias = host_alias(source, device)
        except RuntimeError as error:
            raise RuntimeError(
                f"could not pin RAM for the offloaded experts: {error}. Free some RAM, or keep more layers on the "
                "GPU with a lower DRAMA3_VRAM_RESERVE_GB / DRAMA3_OFFLOAD_LAYERS in start.js."
            ) from error
        experts._buffers[name] = alias
        moved += alias.numel() * alias.element_size()
    experts.forward = offloaded_experts_forward.__get__(experts)
    return moved


def to_gpu(experts: torch.nn.Module) -> int:
    """Copy one layer's experts from pinned RAM back into VRAM. Bytes moved."""
    moved = 0
    for name in ("w13_weight", "w2_weight"):
        experts._buffers[name] = experts._buffers[name].clone()  # allocates in VRAM, reads RAM over PCIe
        moved += experts._buffers[name].numel() * experts._buffers[name].element_size()
    experts.__dict__.pop("forward", None)  # back to MoEExperts.forward (the FP8 kernel on VRAM)
    return moved


def offload(model: torch.nn.Module, codec: torch.nn.Module, device: torch.device) -> None:
    """Replace the routed-expert weights of the planned layers by host aliases (on CPU, before the move)."""
    expert_modules = [layer.mlp.experts for layer in model.backbone.model.layers
                      if getattr(layer.mlp, "experts", None) is not None and layer.mlp.experts.fp8]
    if not expert_modules:
        log("no FP8 expert layers found; nothing to offload")
        return
    expert_bytes = sum(offloadable_bytes(m) for m in expert_modules)
    # Everything that goes to the GPU whatever the plan: dense weights, scales, codec
    resident = tensor_bytes(model) + tensor_bytes(codec) - expert_bytes
    count = plan_layers(expert_modules, resident, device)
    global PLACEMENT
    PLACEMENT = (count, len(expert_modules))
    if count == 0:
        log("all experts fit on the GPU; offload off")
        return
    started = time.perf_counter()
    # The last layers go to host RAM; which ones does not change the PCIe traffic.
    host = sum(to_ram(experts, device) for experts in expert_modules[len(expert_modules) - count:])
    log(f"{count} of {len(expert_modules)} layers keep their experts in pinned RAM "
        f"({host / 2**30:.1f} GiB, pinned in {time.perf_counter() - started:.0f}s); "
        f"{(expert_bytes - host) / 2**30:.1f} GiB of experts on the GPU")


def rebalance(engine, unlocked: bool) -> str:
    """Re-place the experts for a new reference limit, between takes (caller holds the
    take lock). The decode CUDA graphs captured the experts' addresses, so they are
    dropped first (which also frees their memory before anything moves into VRAM)
    and captured again afterwards."""
    global PLACEMENT
    experts, device = STATE["experts"], STATE["device"]
    if not experts:
        return "no offload in use"
    target, current = layers_in_ram(unlocked), PLACEMENT[0]
    count = len(experts)
    if target == current:
        return f"{count - current}/{count} expert layers on the GPU (unchanged)"
    started = time.perf_counter()
    with torch.cuda.device(device):
        if engine is not None:
            engine.graphs = None
            engine.model.decode_graphs = None
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        if target > current:  # more in RAM (60 s): layers count-target .. count-current move out
            moved = sum(to_ram(e, device) for e in experts[count - target:count - current])
        else:  # fewer in RAM (20 s): layers count-current .. count-target come back
            moved = sum(to_gpu(e) for e in experts[count - current:count - target])
        PLACEMENT = (target, count)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()  # give freed pinned RAM back to the system
        if engine is not None:
            import drama3_engine

            for bucket in drama3_engine.WARM_UP_BUCKETS:
                engine.graph_for(bucket)
    message = (f"{count - target}/{count} expert layers on the GPU, {target} in RAM "
               f"({moved / 2**30:.1f} GiB moved in {time.perf_counter() - started:.0f}s)")
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
