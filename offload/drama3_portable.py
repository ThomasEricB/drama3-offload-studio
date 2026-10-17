"""EXPERIMENTAL: Drama 3 without CUDA (Apple Silicon via MPS, or CPU).

The Space's engine needs CUDA: block-FP8 MoE experts run on a Triton kernel, and decode
replays CUDA graphs with CUDA events and streams. This module swaps in an engine that
keeps the Space's prompts, sampling, model code and codec, and replaces only what is
CUDA-specific:

- Decode runs eagerly, one step at a time, with the same operations as the Space's
  graph (candidate lm_head rows -> decode_candidates -> backbone).
- Vocoding runs synchronously on the same device instead of on a side stream.
- MPS has no float8 dtype, so the FP8 experts are kept as their raw bytes (uint8) and
  dequantized per forward through a 256-entry table, with the checkpoint's block scales
  (the same values as fish-speech's dequantize_fp8_blocks). Experts stay 1 byte per
  weight: ~30 GiB instead of 60 GiB in bfloat16.
- No torch.compile.

It needs ~40 GiB of memory for the weights alone: a Mac with 64 GB of unified memory at
the very least. Speed on Apple Silicon is unknown; it was developed and checked on
Linux (CPU, and CUDA in eager mode), not on a Mac.

Environment:
  DRAMA3_DEVICE   "mps" (default when available) or "cpu"
"""

from __future__ import annotations

import collections
import os
import time

import torch
import torch.nn.functional as F

import stall_guard

FP8_BLOCK = 128
GATHERED_MAX_PAIRS = 64  # as fish-speech's MoEExperts: up to this many pairs, one batched matmul


def log(message: str) -> None:
    print(f"[portable] {message}", flush=True)


def portable_device() -> torch.device:
    name = os.environ.get("DRAMA3_DEVICE", "").strip().lower()
    if not name:
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DRAMA3_DEVICE=mps, but this PyTorch build or Mac has no MPS (Apple Silicon, macOS 14+)")
    return torch.device(name)


# ─── FP8 experts as bytes ─────────────────────────────────────────────────────


def fp8_table() -> torch.Tensor:
    """float32 value of every float8_e4m3fn bit pattern (built on CPU, which has the dtype)."""
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()


def dequantize_bytes(weight: torch.Tensor, scale_inv: torch.Tensor, table: torch.Tensor,
                     dtype: torch.dtype) -> torch.Tensor:
    """uint8 FP8 bytes [..., out, in] with fp32 block scales [..., out/128, in/128] -> dtype."""
    *lead, rows, cols = weight.shape
    row_blocks, col_blocks = scale_inv.shape[-2:]
    values = table[weight.int()]
    blocks = values.view(*lead, row_blocks, FP8_BLOCK, col_blocks, FP8_BLOCK)
    scaled = blocks * scale_inv[..., :, None, :, None]
    return scaled.view(*lead, rows, cols).to(dtype)


def byte_expert_weights(self, index, dtype: torch.dtype):
    """MoEExperts.expert_weights for experts stored as uint8 FP8 bytes."""
    return (
        dequantize_bytes(self.w13_weight[index], self.w13_scale_inv[index], self.fp8_table, dtype),
        dequantize_bytes(self.w2_weight[index], self.w2_scale_inv[index], self.fp8_table, dtype),
    )


def byte_experts_forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """MoEExperts.forward without the CUDA FP8 kernel (fish-speech's own fallback paths)."""
    if topk_ids.numel() <= GATHERED_MAX_PAIRS:
        return self.forward_gathered(x, topk_ids, topk_weights)
    return self.forward_grouped(x, topk_ids, topk_weights)


def store_experts_as_bytes(model: torch.nn.Module) -> int:
    """Turn every FP8 expert buffer into a uint8 view (no copy) and route the experts
    through the byte path. Run on CPU, before the move to the device."""
    table = fp8_table()
    count = 0
    for layer in model.backbone.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None or not experts.fp8:
            continue
        for name in ("w13_weight", "w2_weight"):
            experts._buffers[name] = experts._buffers[name].view(torch.uint8)
        experts.register_buffer("fp8_table", table.clone(), persistent=False)
        experts.expert_weights = byte_expert_weights.__get__(experts)
        experts.forward = byte_experts_forward.__get__(experts)
        count += 1
    return count


# ─── Eager engine ─────────────────────────────────────────────────────────────


class _Done:
    """Stands in for a CUDA event: the window was vocoded synchronously."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class EagerDecode:
    """The decode state the Space's DecodeGraphs keeps on the GPU, stepped eagerly."""

    def __init__(self, model, candidate_ids, candidate_head, max_frames: int) -> None:
        from fish_speech.models.text2semantic.fish_omni.generate import RAS_WINDOW_FRAMES

        backbone, tower = model.backbone, model.audio_decoder
        device = tower.device
        width = 1 + model.config.audio_decoder.num_codebooks
        self.model = model
        self.candidate_ids = candidate_ids
        self.candidate_head = candidate_head
        self.ras_window = RAS_WINDOW_FRAMES
        self.hidden = torch.zeros((1, model.config.text.hidden_size), dtype=backbone.dtype, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.frame = torch.zeros((1, width), dtype=torch.long, device=device)
        self.history = torch.full((RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device=device)
        self.out_frames = torch.zeros((max_frames + 1, width), dtype=torch.long, device=device)
        self.produced = 0

    def start(self, hidden, position, prompt_frames, temperature, top_k, top_p) -> None:
        self.hidden.copy_(hidden)
        self.position.fill_(position)
        self.frame.copy_(prompt_frames[-1:])
        window = prompt_frames[-self.ras_window:, 0]
        self.history.fill_(-1)
        self.history[self.ras_window - len(window):].copy_(window)
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        self.produced = 0
        self.silent_run = 0

    def step(self) -> tuple[int, bool]:
        """One frame, as DecodeGraphs.run: returns (main token, logits finite)."""
        import drama3_engine
        import pause_fix

        backbone, tower = self.model.backbone, self.model.audio_decoder
        logits = F.linear(self.hidden, self.candidate_head)
        finite = bool(torch.isfinite(logits).all())
        if pause_fix.SILENCE_CODES and pause_fix.MAX_PAUSE_FRAMES > 0 and self.silent_run >= pause_fix.MAX_PAUSE_FRAMES:
            if getattr(self, "silence_candidates", None) is None:
                self.silence_candidates = torch.tensor([1 + c for c in pause_fix.SILENCE_CODES], device=logits.device)
            logits = logits.index_fill(-1, self.silence_candidates, float("-inf"))  # pause fix: speak or end
        sampled = drama3_engine.decode_candidates(
            tower, self.candidate_ids, logits, self.hidden, self.position, self.frame, self.history,
            **self.sampling,
        )
        main = int(sampled[0, 0])
        silent = int(sampled[0, 1]) in pause_fix.SILENCE_SET
        self.silent_run = self.silent_run + 1 if silent else 0
        if main != self.model.config.eos_token_id:
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            hidden = backbone(inputs_embeds=embeddings[None], input_pos=self.position + 1)
            self.out_frames[self.produced] = sampled[0]
            self.produced += 1
            self.hidden.copy_(hidden[0])
            self.frame.copy_(sampled)
            self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
            self.position.add_(1)
        return main, finite


def make_portable_engine():
    import drama3_engine
    from fish_speech.models.text2semantic.fish_omni.generate import ensure_drama3_caches

    class PortableEngine(drama3_engine.Engine):
        """The Space's Engine with eager decode and synchronous vocoding."""

        def __init__(self, weights) -> None:  # no torch.compile, no CUDA graphs
            self.weights = weights
            self.model = weights.model
            self.config = weights.config
            self.codec = weights.codec
            self.tokenizer = self.model.tokenizer
            self.reference_codes = collections.OrderedDict()
            self.graphs = None
            self.vocoder_stream = None
            self.decode = None

        @torch.inference_mode()
        def warm_up(self, reference) -> None:
            started = time.perf_counter()
            body = {"input": "Places, everyone.", "references": [reference] if reference else None,
                    "max_new_tokens": 2 * drama3_engine.STREAM_STRIDE, "temperature": 1.0, "top_p": 0.9,
                    "top_k": 4097}
            frames = 0
            for event in self.speech(body):
                if event["type"] == "speech.audio.done":
                    frames = event["frames"]
            elapsed = time.perf_counter() - started
            log(f"warmed up on {self.device} in {elapsed:.1f}s ({frames} frames)")

        @torch.inference_mode()
        def prefill(self, prompt: torch.Tensor, request) -> EagerDecode:
            model = self.model
            backbone, tower = model.backbone, model.audio_decoder
            prompt_rows = len(prompt)
            capacity = prompt_rows + request.max_new_tokens
            if capacity > model.config.text.max_position_embeddings:
                raise drama3_engine.RequestError("the prompt and audio limit exceed the context")
            ensure_drama3_caches(model, capacity)
            if self.decode is None:
                self.decode = EagerDecode(model, self.weights.candidate_ids, self.weights.candidate_head,
                                          drama3_engine.MAX_NEW_TOKENS_CAP)
            device = tower.device
            frames = prompt.to(device)
            positions = torch.arange(prompt_rows, device=device)
            for start in range(0, prompt_rows, drama3_engine.PREFILL_CHUNK_ROWS):
                end = min(start + drama3_engine.PREFILL_CHUNK_ROWS, prompt_rows)
                chunk = frames[start:end]
                embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
                hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[start:end])[0]
                paired_end = min(end, prompt_rows - 1)
                if paired_end > start:
                    tower.prefill(hidden[: paired_end - start], positions[start:paired_end],
                                  frames[start:paired_end], frames[start + 1: paired_end + 1])
            self.decode.start(hidden[-1:], prompt_rows - 1, frames, request.temperature, request.top_k,
                              request.top_p)
            return self.decode

        @torch.inference_mode()
        def frames(self, prompt: torch.Tensor, request):
            import pause_fix

            decode = self.prefill(prompt, request)
            eos = self.config.eos_token_id
            expected = pause_fix.expected_frames(request.text)
            guard = bool(pause_fix.SILENCE_CODES) and pause_fix.MAX_PAUSE_FRAMES > 0
            stalls = stall_guard.StallGuard()
            for step in range(request.max_new_tokens):
                main, finite = decode.step()
                if not finite:
                    raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
                if main == eos:
                    yield from stalls.release(step)
                    return "stop"
                stalled, counts = stalls.add(main, step + 1)
                if stalled:
                    log(f"stall: a drone from frame {stalls.released}; the take ends there")
                    return "stall"
                yield from counts
                if guard and step + 1 >= int(pause_fix.CAP_RATIO * expected):
                    yield from stalls.release(step + 1)
                    return "stop"  # length guard
                if guard and decode.silent_run >= pause_fix.MAX_PAUSE_FRAMES and step + 1 >= int(pause_fix.END_RATIO * expected):
                    yield from stalls.release(step + 1)
                    return "stop"  # a pause after the text should be done
            yield from stalls.release(request.max_new_tokens)
            return "length"

        @torch.inference_mode()
        def queue_window(self, start: int, end: int, emitted: int):
            codes = self.decode.out_frames[start:end, 1:].T.contiguous()
            audio = self.vocode(codes)[(emitted - start) * drama3_engine.FRAME_SAMPLES:]
            pcm = (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu()
            return drama3_engine.PendingAudio(_Done(), pcm)

    return PortableEngine


# ─── Installation ─────────────────────────────────────────────────────────────


def install() -> torch.device:
    """Load on CPU, store the experts as bytes, move to the portable device, and make
    the backend build PortableEngine instead of the Space's CUDA Engine."""
    import drama3_engine

    target = portable_device()
    original = drama3_engine.load_weights

    def load_weights(checkpoint_dir=None, device: str = "cuda"):  # the Space asks for "cuda"
        weights = original(checkpoint_dir, device="cpu")
        layers = store_experts_as_bytes(weights.model)
        started = time.perf_counter()
        # Module.to replaces one tensor at a time, so the CPU copy is released as it goes
        weights.model.to(target)
        weights.codec.to(target)
        log(f"{layers} expert layers kept as FP8 bytes; weights on {target} in "
            f"{time.perf_counter() - started:.0f}s")
        return weights

    drama3_engine.load_weights = load_weights
    drama3_engine.Engine = make_portable_engine()
    log(f"EXPERIMENTAL eager engine on {target} (no CUDA graphs, Triton or torch.compile)")
    return target
