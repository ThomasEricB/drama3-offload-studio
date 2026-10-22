"""Drama 3 on Apple Silicon through MLX (fcmeyer/drama-3-MLX-4bit-HQ or -3bit-HQ, run by mlx-audio).

The Space's engine needs CUDA, and the eager PyTorch fallback (drama3_portable) is very
slow on a Mac. This engine keeps everything above the model as it is (the Space's
prompts, server and UI, and our pause fix, stall guard, long takes and infill edits)
and runs the model and the codec in MLX instead:

- The weights are mlx-audio conversions of the official checkpoint (install_mlx.js):
  4-bit HQ, routed experts 4-bit and the rest bfloat16 (24.5 GB, peak ~39 GB), or 3-bit HQ,
  experts 3-bit (the most sensitive layers 4-bit) and attention / tower 8-bit (19.7 GB,
  peak ~34 GB); each plus its 1.6 GB codec.
- A take's prompt rows are the Space's own (drama3_engine.segment_rows), fed to
  mlx-audio's backbone and tower; each step samples the main token with repetition-aware
  sampling and the tower's codebooks as mlx-audio does, with the pause fix's mask applied
  inside the step, so step n + 1 is queued before step n is read.
- All MLX work runs on one thread of its own: MLX streams belong to the thread that
  made them, and the server runs every take in a new thread.

Selected with DRAMA3_BACKEND=mlx (start.js, from the MLX menu entries); the default
backend on a Mac stays the eager PyTorch engine on the official checkpoint.

Environment:
  DRAMA3_MLX_DIR   the MLX model folder (models/drama-3-mlx-4bit or models/drama-3-mlx-3bit)
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import stall_guard

MODEL_FILES = ("config.json", "tokenizer.json", "codec.safetensors", "model-00001.safetensors")
# mlx-audio's repetition-aware sampling: a main token that repeats one of the last
# RAS_WINDOW main tokens is drawn again with these settings
RAS_WINDOW, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P = 16, 1.0, 30, 0.95


def log(message: str) -> None:
    print(f"[mlx] {message}", flush=True)


def model_dir() -> Path:
    return Path(os.environ.get("DRAMA3_MLX_DIR", "")).expanduser()


def available() -> str | None:
    """Why the MLX backend cannot run here, or None when it can."""
    try:
        import mlx.core  # noqa: F401
        import mlx_audio.tts.models.fish_drama3  # noqa: F401
    except ImportError as error:
        return f"mlx-audio is not installed ({error.name})"
    folder = model_dir()
    if not os.environ.get("DRAMA3_MLX_DIR"):
        return "DRAMA3_MLX_DIR is not set"
    missing = [name for name in MODEL_FILES if not (folder / name).is_file()]
    if missing:
        return f"the MLX model is incomplete in {folder} (missing {', '.join(missing)}); run its Install"
    return None


def model_name() -> str:
    """"MLX 4-bit HQ" or "MLX 3-bit HQ": the bits most routed experts have (the 3-bit model's
    default is the 8 bits of its attention layers)."""
    try:
        with open(model_dir() / "config.json", encoding="utf-8") as f:
            quantization = json.load(f)["quantization"]
        experts = [v["bits"] for k, v in quantization.items() if "switch_mlp" in k and isinstance(v, dict)]
        bits = collections.Counter(experts).most_common(1)[0][0] if experts else quantization["bits"]
        return f"MLX {bits}-bit HQ"
    except (OSError, KeyError, ValueError):
        return "MLX"


class MlxThread:
    """Runs every MLX call on one thread, in order."""

    def __init__(self) -> None:
        self.pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="drama3-mlx")

    def __call__(self, fn, *args, **kwargs):
        return self.pool.submit(fn, *args, **kwargs).result()


MLX = MlxThread()
ACTIVE = False  # set by install(): the MLX backend runs the takes (seeds.seed_rng seeds MLX too)


def seed(value: int) -> None:
    """Seed MLX's global random state, which sampling draws from (on the MLX thread)."""
    import mlx.core as mx

    MLX(mx.random.seed, value)


# ─── Weights ──────────────────────────────────────────────────────────────────


class ModelShim:
    """What the Space's Engine reads from a model: its tokenizer (the MLX model's)."""

    def __init__(self, mlx_model) -> None:
        self.mlx = mlx_model
        self.tokenizer = mlx_model.tokenizer


class Weights:
    """drama3_engine.Weights for the MLX model: the Space's config, the MLX model and codec."""

    def __init__(self, config, mlx_model) -> None:
        self.config = config
        self.model = ModelShim(mlx_model)
        self.codec = mlx_model.codec


def space_config(folder: Path):
    """The Space's FishOmniConfig from the MLX model's config.json (the same fields; its
    model_type names mlx-audio's model)."""
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig

    with open(folder / "config.json", encoding="utf-8") as f:
        data = json.load(f)
    data = {key: value for key, value in data.items() if key != "quantization"}
    return FishOmniConfig.from_dict({**data, "model_type": "fish_omni"})


def load_weights(checkpoint_dir=None, device: str = "cuda") -> Weights:  # the Space asks for "cuda"
    def load():
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        started = time.perf_counter()
        model = load_model(model_dir())
        mx.eval(model.parameters())
        log(f"{model_name()} loaded on {mx.default_device()} in {time.perf_counter() - started:.0f}s "
            f"({mx.get_active_memory() / 2**30:.1f} GiB)")
        return model

    config = space_config(model_dir())
    model = MLX(load)
    if model.tokenizer.token_to_id("<|im_end|>") != config.eos_token_id:
        raise RuntimeError("the MLX model's tokenizer does not match the Space's config")
    return Weights(config, model)


# ─── Decode ───────────────────────────────────────────────────────────────────


def sample(logits, temperature: float, top_k: int, top_p: float):
    from mlx_audio.tts.models.fish_drama3.drama3 import sample_top_k_top_p

    return sample_top_k_top_p(logits, temperature, top_k, top_p)


class _Done:
    """Stands in for a CUDA event: the window was vocoded synchronously."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class MlxDecode:
    """One take's decode state. out_frames holds the frames on the host, as the Space's
    DecodeGraphs.out_frames (rows [main token, c0..c9])."""

    def __init__(self, model, config, max_frames: int) -> None:
        import pause_fix

        self.model = model
        self.eos = config.eos_token_id
        self.width = 1 + config.audio_decoder.num_codebooks
        self.out_frames = torch.zeros((max_frames + 1, self.width), dtype=torch.long)
        self.produced = 0
        self.silent_run = 0
        self.pause_frames = pause_fix.MAX_PAUSE_FRAMES if pause_fix.SILENCE_CODES else 0
        self.silence = sorted(pause_fix.SILENCE_SET) if self.pause_frames > 0 else []
        self.pending = None

    def start(self, prompt: torch.Tensor, request) -> None:
        """Prefill the prompt rows and queue the first step (on the MLX thread)."""
        import mlx.core as mx

        model = self.model
        frames = mx.array(prompt.numpy().astype(np.int32))
        self.caches = (model.backbone.make_cache(), model.tower.make_cache())
        hidden = model._prefill(frames, self.caches, 0, frames.shape[0])
        history = np.full(RAS_WINDOW, -1, dtype=np.int32)
        window = prompt[-RAS_WINDOW:, 0].numpy()
        history[RAS_WINDOW - len(window):] = window
        band = model.tower.vocab_size
        mask = np.zeros(band + 1, dtype=bool)  # semantic_head rows: the band, then end of speech
        mask[self.silence] = True
        self.silence_mask = mx.array(mask)
        self.sampling = (request.temperature, request.top_k, request.top_p)
        self.pending = self._step(hidden, frames[None, -1:], mx.array(history), mx.array(0))
        mx.async_eval(self.pending)

    def _step(self, hidden, frame, history, silent_run):
        """mlx-audio's Model._step with the pause fix: after pause_frames silent frames,
        silence is masked from the main token's draw. Everything stays lazy."""
        import mlx.core as mx

        model = self.model
        tower = model.tower
        temperature, top_k, top_p = self.sampling
        logits = model.semantic_head(hidden[:, 0]).astype(mx.float32)
        finite = mx.isfinite(logits).all()
        if self.pause_frames > 0:
            masked = (silent_run >= self.pause_frames) & self.silence_mask
            logits = mx.where(masked, -mx.inf, logits)
        first = model._candidate_ids[sample(logits, temperature, top_k, top_p)]
        retry = model._candidate_ids[sample(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)]
        repeated = (history[None, :] == first[:, None]).any(axis=-1)
        main = mx.where(repeated & tower.is_semantic(first), retry, first)
        c0 = mx.clip(main - model.semantic_begin, 0, tower.vocab_size - 1)
        semantic = tower.is_semantic(main)[:, None]

        def draw(head_logits):
            return sample(head_logits[:, 0], temperature, top_k, top_p)[:, None]

        codes = tower.residual_codes(hidden, c0[:, None], semantic, frame, self.caches[1], draw)
        new_frame = mx.concatenate([main[:, None], *codes], axis=-1)[:, None, :].astype(mx.int32)
        next_hidden = model.backbone.model(None, cache=self.caches[0], input_embeddings=model._embed(new_frame))
        history = mx.concatenate([history[1:], main])
        silent = mx.take(self.silence_mask, c0[0]) if self.pause_frames > 0 else mx.array(False)
        silent_run = mx.where(silent, silent_run + 1, 0)
        return new_frame, next_hidden, history, silent_run, finite

    def step(self) -> tuple[int, bool]:
        """Read the queued step, queue the next one, and keep the frame (on the MLX thread)."""
        import mlx.core as mx

        frame, hidden, history, silent_run, finite = self.pending
        self.pending = self._step(hidden, frame, history, silent_run)
        mx.async_eval(self.pending)
        row = np.array(frame[0, 0], dtype=np.int64)
        main, ok = int(row[0]), bool(finite.item())
        self.silent_run = int(silent_run.item())
        if main != self.eos:
            self.out_frames[self.produced] = torch.from_numpy(row)
            self.produced += 1
        return main, ok


def make_mlx_engine():
    import drama3_engine

    class MlxEngine(drama3_engine.Engine):
        """The Space's Engine with the model and codec in MLX."""

        def __init__(self, weights: Weights) -> None:
            self.weights = weights
            self.model = weights.model
            self.config = weights.config
            self.codec = weights.codec
            self.tokenizer = self.model.tokenizer
            self.reference_codes = collections.OrderedDict()
            self.graphs = None
            self.vocoder_stream = None
            self.decode = None

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")  # tensors handed to MLX come from the host

        def encode(self, reference) -> torch.Tensor:
            """Codes [num_codebooks, frames] of a reference clip, cached by content."""
            key = hashlib.sha1(str(reference.get("data") or reference.get("base64")).encode()).hexdigest()
            if (codes := self.reference_codes.get(key)) is not None:
                self.reference_codes.move_to_end(key)
                return codes
            samples = np.clip(drama3_engine.reference_audio(reference), -1.0, 1.0)

            def run():
                import mlx.core as mx

                indices, lengths = self.codec.encode(mx.array(samples)[None, None, :])
                return np.array(indices[0, :, : int(lengths[0].item())], dtype=np.int64)

            codes = torch.from_numpy(MLX(run))
            self.reference_codes[key] = codes
            while len(self.reference_codes) > drama3_engine.REFERENCE_CACHE_SIZE:
                self.reference_codes.popitem(last=False)
            return codes

        def vocode(self, codes: torch.Tensor) -> torch.Tensor:
            """Waveform [samples] of codes [num_codebooks, frames]."""
            indices = codes.cpu().numpy().astype(np.int32)

            def run():
                import mlx.core as mx

                audio, _ = self.codec.decode(mx.array(indices)[None], mx.array([indices.shape[1]]))
                return np.array(audio.astype(mx.float32)).reshape(-1)

            return torch.from_numpy(MLX(run))

        def warm_up(self, reference) -> None:
            started = time.perf_counter()
            body = {"input": "Places, everyone.", "references": [reference] if reference else None,
                    "max_new_tokens": 2 * drama3_engine.STREAM_STRIDE, "temperature": 1.0, "top_p": 0.9,
                    "top_k": 4097}
            frames = 0
            for event in self.speech(body):
                if event["type"] == "speech.audio.done":
                    frames = event["frames"]
            log(f"warmed up in {time.perf_counter() - started:.1f}s ({frames} frames)")

        def prefill(self, prompt: torch.Tensor, request) -> MlxDecode:
            if len(prompt) + request.max_new_tokens > self.config.text.max_position_embeddings:
                raise drama3_engine.RequestError("the prompt and audio limit exceed the context")
            self.decode = MlxDecode(self.model.mlx, self.config, drama3_engine.MAX_NEW_TOKENS_CAP)
            MLX(self.decode.start, prompt, request)
            return self.decode

        def frames(self, prompt: torch.Tensor, request):
            """As drama3_portable's: the pause fix's length guards and the stall guard."""
            import pause_fix

            decode = self.prefill(prompt, request)
            eos = self.config.eos_token_id
            expected = pause_fix.expected_frames(request.text)
            guard = bool(pause_fix.SILENCE_CODES) and pause_fix.MAX_PAUSE_FRAMES > 0
            stalls = stall_guard.StallGuard()
            for step in range(request.max_new_tokens):
                main, finite = MLX(decode.step)
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

        def queue_window(self, start: int, end: int, emitted: int):
            codes = self.decode.out_frames[start:end, 1:].T.contiguous()
            audio = self.vocode(codes)[(emitted - start) * drama3_engine.FRAME_SAMPLES:]
            pcm = (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            return drama3_engine.PendingAudio(_Done(), pcm)

    return MlxEngine


# ─── Installation ─────────────────────────────────────────────────────────────


def install() -> None:
    """Make the backend load the MLX model and build MlxEngine instead of the CUDA Engine."""
    import drama3_engine

    global ACTIVE
    drama3_engine.load_weights = load_weights
    drama3_engine.Engine = make_mlx_engine()
    ACTIVE = True
    log(f"MLX engine: {model_name()} from {model_dir()}")
