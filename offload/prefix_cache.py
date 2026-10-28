"""A voice reference's prefill, made once and reused by the takes that follow.

Every take reads its whole prompt again, and for a cloned voice most of it is the
reference clip: a 20 s clip is about 630 of a short line's 670 rows, 1.5 s of prefill and
of the time to first audio on an RTX 5090 (2.2 s for a 39 s clip, against 0.33 s for a
random voice's 37 rows). Adapted from drama-tensorfold's engine patch 0012
(plotarmordev/drama-tensorfold #17), with its corrections (BobClawblaw/drama-tensorfold
#1), TensorFold's CUDA version (plotarmordev/TensorFold #4) and SGLang Omni's contract
for the same cache (sgl-project/sglang-omni #3235):

- What is kept: the model's state after the prompt's first `cut` rows, every tensor a
  prefill writes: the full-attention layers' keys and values, the linear-attention
  (gated delta) layers' conv window and fp32 recurrent state, and the audio decoder's
  keys and values (row 0, the one a take without guidance uses), all at the same row.
- Where it is cut (prefix_cut): at the end of the system turn, which holds the
  reference ([Reference 1] Text / Speech), the row before the first user turn.
- The same cuts with and without the cache (TensorFold's "preserve the uncached chunk
  cuts"): a prompt with a reference is always read in two parts, cut there, also with
  the cache off, so a take never depends on whether its prefix came from the cache or on
  DRAMA3_PREFIX_CACHE: seeded takes are byte-identical built, reused and with it off. A
  cut where the Space's own prefill does not break (every PREFILL_CHUNK_ROWS rows from
  row 0) changes a rounding: measured, a cut at a multiple of 64 (the gated delta rule's
  block) changed the audio too (cuBLAS picks its kernels by row count), so only
  TensorFold's rule, whole 512-row chunks (DRAMA3_PREFIX_CACHE_ALIGN=512), keeps the
  bytes of the engine before this cache, and it keeps nothing for a system turn under
  512 rows (a clip under about 22 s: most of them, at the Studio's 20 s limit).
- Identity: the key is the prompt's rows up to and including row `cut`, every column
  (codes included): the audio decoder's state at row cut-1 is labelled with frame cut
  (SGLang Omni's "complete frame identity"). They are compared exactly; the hash only
  finds the entry.
- Safety: the kept states are copies, so another take overwriting the model's caches,
  or a prefill failing half way, cannot make them stale; a prefill that fails before
  the cut keeps nothing. An entry is used only while the caches it came from are the
  ones in use (a new memory layout or a 3-row guided cache drops it).
- Memory: pinned RAM, about 25 MB per 100 rows, for the last ENTRIES voices; an evicted
  voice's buffers are reused for the next one instead of pinning new memory each switch.

Only the CUDA engines (full model, Small GGUF), and only takes without guidance: tag and
voice guidance read 2-3 rows, one of them without the reference. DRAMA3_PREFIX_CACHE=0
turns the keeping and reusing off; the audio is the same either way.

Measured on an RTX 5090, a 20 s clip (533 rows): reading the whole prompt took 1.34 s,
94% of it the MoE expert kernel reading the experts from RAM; the 26 rows after a reused
prefix take 0.27-0.32 s. The gated delta rule, attention and the rest are under 5%, so
capturing the delta rule's prefill in CUDA graphs (sglang #46619) would gain little here.
"""

from __future__ import annotations

import collections
import hashlib
import os
import time

import torch

ENABLED = os.environ.get("DRAMA3_PREFIX_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
ENTRIES = int(os.environ.get("DRAMA3_PREFIX_CACHE_ENTRIES", "3"))
ALIGN = int(os.environ.get("DRAMA3_PREFIX_CACHE_ALIGN", "1"))  # 512: whole prefill chunks only
MIN_AUDIO_ROWS = 32  # a prefix with less reference audio is not worth keeping
CAPACITY_STEP = 256  # rows: buffers are sized up to this, so a voice switch can reuse them


def log(message: str) -> None:
    print(f"[prefix] {message}", flush=True)


def prefix_part(kind: str, tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Row 0's share of a cache: positions 0..rows-1 of keys / values, or a whole state."""
    return tensor[:1, :, :rows] if kind == "kv" else tensor[:1]


def model_caches(model) -> list[tuple[str, torch.Tensor]]:
    """Every tensor a prefill writes: the backbone's keys / values and recurrent states,
    and the audio decoder's keys / values."""
    out = []
    for layer in model.backbone.model.layers:
        if layer.layer_type == "full_attention":
            out += [("kv", layer.self_attn.k_cache), ("kv", layer.self_attn.v_cache)]
        else:
            out += [("state", layer.linear_attn.conv_state), ("state", layer.linear_attn.recurrent_state)]
    tower = model.audio_decoder
    for layer in (*tower.shared_layers, *tower.residual_layers):
        attention = layer.transformer_layer.attention
        out += [("kv", attention.k_cache), ("kv", attention.v_cache)]
    if any(tensor is None for _, tensor in out):
        return []
    return out


class Saved:
    """The model's state after a prompt prefix, in pinned RAM, and the rows it holds."""

    def __init__(self, caches: list[tuple[str, torch.Tensor]], rows: int, key_rows: torch.Tensor,
                 spare: "Saved | None") -> None:
        self.sources = [(kind, tensor, tensor.data_ptr(), tuple(tensor.shape)) for kind, tensor in caches]
        self.rows, self.key_rows = rows, key_rows.clone()
        capacity = -(-rows // CAPACITY_STEP) * CAPACITY_STEP
        buffers = spare.buffers if spare is not None and spare.reusable(caches, rows) else None
        self.buffers, self.capacity = buffers or [], spare.capacity if buffers else capacity
        for index, (kind, tensor) in enumerate(caches):
            part = prefix_part(kind, tensor, rows)
            if not buffers:
                shape = prefix_part(kind, tensor, self.capacity).shape if kind == "kv" else part.shape
                self.buffers.append(torch.empty(shape, dtype=part.dtype, pin_memory=True))
            self.view(index, kind).copy_(part, non_blocking=True)
        torch.cuda.synchronize()
        self.bytes = sum(buffer.numel() * buffer.element_size() for buffer in self.buffers)

    def view(self, index: int, kind: str) -> torch.Tensor:
        buffer = self.buffers[index]
        return buffer[:, :, :self.rows] if kind == "kv" else buffer

    def reusable(self, caches, rows: int) -> bool:
        """Its buffers fit another prefix of these caches."""
        return rows <= self.capacity and len(caches) == len(self.sources) and all(
            tuple(tensor.shape) == shape for (_, tensor), (_, _, _, shape) in zip(caches, self.sources))

    def fits(self, caches) -> bool:
        """The caches it came from are the ones in use (not reallocated since)."""
        return len(caches) == len(self.sources) and all(
            tensor is source and tensor.data_ptr() == pointer and tuple(tensor.shape) == shape
            for (_, tensor), (_, source, pointer, shape) in zip(caches, self.sources))

    def restore(self) -> None:
        for index, (kind, tensor, _, _) in enumerate(self.sources):
            prefix_part(kind, tensor, self.rows).copy_(self.view(index, kind), non_blocking=True)


def prefix_cut(prompt: torch.Tensor, im_start: int, semantic: tuple[int, int], align: int) -> int | None:
    """Where the kept prefix ends: the last multiple of `align` at or before the end of the
    system turn (the row before the first user turn); None when the system turn holds no
    reference worth keeping before it."""
    starts = torch.nonzero(prompt[:, 0] == im_start).flatten().tolist()
    if len(starts) < 2 or starts[0] != 0:
        return None
    cut = starts[1] // max(align, 1) * max(align, 1)
    if cut + 2 > len(prompt):  # the label row and at least one more row after the prefix
        return None
    audio = int(((prompt[:cut, 0] >= semantic[0]) & (prompt[:cut, 0] <= semantic[1])).sum())
    return cut if audio >= MIN_AUDIO_ROWS else None


def install() -> None:
    """Wrap the engine (CUDA backends) so a cacheable prompt's prefix is read once."""
    import drama3_engine as de

    base = de.Engine

    class PrefixCacheEngine(base):
        saved: collections.OrderedDict | None = None

        @torch.inference_mode()
        def prefill(self, prompt, request):
            tower = self.model.audio_decoder
            cut = prefix_cut(prompt, self.tokenizer.token_to_id("<|im_start|>"),
                             (tower.semantic_begin, tower.semantic_end), ALIGN)
            if cut is None:
                return super().prefill(prompt, request)
            if self.saved is None:
                self.saved = collections.OrderedDict()
            model = self.model
            backbone, tower = model.backbone, model.audio_decoder
            graphs = self.decode_graphs()
            prompt_rows = len(prompt)
            capacity = prompt_rows + request.max_new_tokens
            if capacity > graphs.graph_capacity(model):
                raise de.RequestError("the prompt and audio limit exceed the context")
            needed = next(b for b in graphs.buckets if b >= capacity)
            for bucket in graphs.buckets:
                if prompt_rows + 1 <= bucket <= needed:
                    self.graph_for(bucket)
            device = tower.device
            frames = prompt.to(device)
            positions = torch.arange(prompt_rows, device=device)
            caches = model_caches(model)
            key_rows = prompt[:cut + 1].cpu()
            key = (de.PREFILL_CHUNK_ROWS, hashlib.sha1(key_rows.contiguous().numpy().tobytes()).hexdigest())
            entry = self.saved.get(key) if ENABLED else None
            if entry is not None and not (entry.fits(caches) and torch.equal(entry.key_rows, key_rows)):
                del self.saved[key]
                entry = None
            began = time.perf_counter()
            if entry is not None:
                entry.restore()
                self.saved.move_to_end(key)
            hidden = None
            # Chunks as an uncached prefill makes them (from row 0, PREFILL_CHUNK_ROWS long),
            # also cut at `cut`: a multiple of the delta rule's block, so no row is read differently
            cuts = sorted({cut, *range(0, prompt_rows, de.PREFILL_CHUNK_ROWS), prompt_rows})
            for start, end in zip(cuts, cuts[1:]):
                if entry is not None and end <= cut:
                    continue
                chunk = frames[start:end]
                embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
                hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[start:end])[0]
                paired_end = min(end, prompt_rows - 1)
                if paired_end > start:
                    tower.prefill(hidden[: paired_end - start], positions[start:paired_end],
                                  frames[start:paired_end], frames[start + 1: paired_end + 1])
                if end == cut and entry is None and caches and ENABLED:
                    spare = self.saved.popitem(last=False)[1] if len(self.saved) >= ENTRIES else None
                    self.saved[key] = saved = Saved(caches, cut, key_rows, spare)
                    reused = spare is not None and saved.buffers is spare.buffers
                    log(f"reference kept: {cut} of {prompt_rows} rows, {saved.bytes / 2**20:.0f} MiB in RAM"
                        f"{' (an older voice’s buffers)' if reused else ''}")
            if entry is not None:
                torch.cuda.synchronize()
                log(f"reference reused: {cut} of {prompt_rows} rows, the rest read in "
                    f"{time.perf_counter() - began:.2f}s")
            graphs.start(hidden[-1:], prompt_rows - 1, frames, request.temperature, request.top_k, request.top_p)
            return graphs

    PrefixCacheEngine.__name__ = base.__name__
    de.Engine = PrefixCacheEngine
