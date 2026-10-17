"""Serve the Drama 3 Space locally, on 127.0.0.1.

Run from the Space folder (app/): python ../offload/serve.py <port>
Same as app.py's local launch, plus:
- the model and backend:
  - Start (24+ GB VRAM): the full model. On NVIDIA, the Space's CUDA engine with MoE
    expert offload (drama3_offload); without CUDA, the EXPERIMENTAL eager engine
    (drama3_portable: Apple Silicon / MPS, or CPU). DRAMA3_BACKEND=cuda|portable.
  - Start (MLX 4-bit / 3-bit, Apple Silicon): DRAMA3_BACKEND=mlx, the MLX conversions
    run by mlx-audio (drama3_mlx), from DRAMA3_MLX_DIR.
  - Start (Small GGUF, 16 GB VRAM), when DRAMA3_GGUF is set: drama3-small read from
    its GGUF into the same CUDA engine (gguf_drama3), no audio.cpp.
- the drama3-small pause fix (pause_fix), for both models.
- one take at a time. On ZeroGPU every take ran in its own GPU worker; here all takes
  share one engine (one KV cache), and concurrent takes corrupt each other's audio.
- Windows: the Space's /dev/shm report uses os.statvfs, which Windows lacks.
"""

import os
import sys
import threading

sys.path.insert(0, os.getcwd())  # the Space's modules (app.py, drama3_engine, ...)

# The web server's admission gate: further takes wait in its queue (shown in the UI)
os.environ.setdefault("DEMO_MAX_ACTIVE", "1")
# No ZeroGPU quota locally: don't end takes at the Space's per-call GPU budget (about 43 s
# for the longest line); the server's TAKE_TIMEOUT_S still bounds each take
os.environ.setdefault("ZEROGPU_MAX_DURATION_S", "3600")
os.environ.setdefault("ZEROGPU_MIN_FRAMES_PER_S", "1")

# Apple GPU: run any operation MPS lacks on the CPU instead of failing (read when torch loads)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import warnings  # noqa: E402

# Gradio converts the voice preview to 16-bit itself; the warning repeats on every preview
warnings.filterwarnings("ignore", message="Trying to convert audio automatically from float32")

import torch  # noqa: E402
import drama3_engine  # noqa: E402


def use_installed_checkpoint() -> None:
    """Load the checkpoint Install downloaded (DRAMA3_CHECKPOINT_DIR), never the HF cache.

    The engine otherwise looks in HF_HOME and silently downloads 41 GB when the files
    are not there, e.g. when Pinokio's per-app HF_HOME differs from the one Install used."""
    folder = os.environ.get("DRAMA3_CHECKPOINT_DIR")
    if not folder:
        # No folder given: accept only a checkpoint already in the HF cache, never download
        def cached_only():
            from huggingface_hub import snapshot_download

            try:
                return __import__("pathlib").Path(snapshot_download(
                    drama3_engine.MODEL_ID, revision=drama3_engine.MODEL_REVISION, local_files_only=True,
                    allow_patterns=["*.json", "*.safetensors", "codec.pth"]))
            except Exception as error:
                raise SystemExit("Drama 3 checkpoint not found (set DRAMA3_CHECKPOINT_DIR, as start.js does, "
                                 f"or run Install): {type(error).__name__}") from error

        drama3_engine.checkpoint_path = cached_only
        return
    from pathlib import Path
    import json

    root = Path(folder)
    needed = ["config.json", "codec.pth", "tokenizer.json", "model.safetensors.index.json"]
    if (root / "model.safetensors.index.json").is_file():
        with open(root / "model.safetensors.index.json", encoding="utf-8") as f:
            needed += sorted(set(json.load(f)["weight_map"].values()))
    missing = [name for name in needed if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"Drama 3 checkpoint incomplete in {root} (missing {', '.join(missing[:4])}"
                         f"{'...' if len(missing) > 4 else ''}). Run Install to download it.")
    drama3_engine.checkpoint_path = lambda: root


GGUF_PATH = os.environ.get("DRAMA3_GGUF", "")  # set by start_small.js
PREVIEW = os.environ.get("DEMO_PREVIEW") == "1"  # the Space's UI-only mode: no model, no engine

if GGUF_PATH:
    backend = "small"
    if not PREVIEW:
        if not torch.cuda.is_available():
            raise SystemExit("Start (Small GGUF) needs an NVIDIA GPU (RTX 40-series or newer); none is visible to PyTorch")
        if not os.path.isfile(GGUF_PATH):
            raise SystemExit(f"drama3-small not found at {GGUF_PATH}. Run “Install Small GGUF”.")
        import gguf_drama3  # noqa: E402

        gguf_drama3.install(GGUF_PATH)  # the engine loads the GGUF; references are encoded on the CPU
else:
    backend = os.environ.get("DRAMA3_BACKEND", "auto").strip().lower() or "auto"
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "portable"
    if backend != "mlx":  # MLX reads its own model folder (DRAMA3_MLX_DIR)
        use_installed_checkpoint()
if backend == "small":
    pass
elif backend == "cuda":
    import drama3_offload  # noqa: E402  (imports Triton, which only exists with CUDA)

    drama3_offload.install()
elif backend == "mlx":
    import drama3_mlx  # noqa: E402

    if not PREVIEW and (reason := drama3_mlx.available()):
        raise SystemExit(f"DRAMA3_BACKEND=mlx: {reason}")
    os.environ.setdefault("ENGINE_START_TIMEOUT_S", "1800")  # loading 26 GB from disk
    drama3_mlx.install()
elif backend == "portable":
    import drama3_portable  # noqa: E402

    # Eager decode is much slower than the Space's GPU: wait longer for audio, and skip
    # the warm-up take (there is nothing to compile or capture)
    os.environ.setdefault("ENGINE_START_TIMEOUT_S", "1800")
    os.environ.setdefault("TAKE_TIMEOUT_S", "3600")
    os.environ.setdefault("DRAMA3_WARM_UP", "0")
    drama3_portable.install()
else:
    raise SystemExit(f"DRAMA3_BACKEND must be auto, cuda, mlx or portable; got {backend!r}")

import uvicorn  # noqa: E402

import branding  # noqa: E402
import long_takes  # noqa: E402
import pause_fix  # noqa: E402
import studio_extras  # noqa: E402

# The drama3-small pause fix, for the full model too: silence is masked from the next draw
# after DRAMA3_MAX_PAUSE_FRAMES silent frames, and a length guard ends takes that never stop
SILENCE_CODES = os.environ.get("DRAMA3_SILENCE_CODES", "")
if not PREVIEW:
    if SILENCE_CODES and os.path.isfile(SILENCE_CODES):
        if backend in ("cuda", "small"):
            pause_fix.install(SILENCE_CODES)
        else:
            pause_fix.load_codes(SILENCE_CODES)  # the portable and MLX engines apply it in their own step
        print(f"[pause] pause fix on: {len(pause_fix.SILENCE_CODES)} silence codes, "
              f"{pause_fix.MAX_PAUSE_FRAMES} frames", flush=True)
    else:
        print("[pause] pause fix off: silence codes not found (run Install)", flush=True)

# Long scripts are chunked inside the engine (wraps whichever Engine the backend set up)
long_takes.install_engine()
# A take can now be minutes long: the per-take limit is no longer a useful guard
# (a stalled engine is still caught by the server's 15 s idle timeout)
os.environ.setdefault("TAKE_TIMEOUT_S", "86400")

with branding.rebranded_ui(backend), studio_extras.extended_studio():
    from app import app  # noqa: E402  (loads Drama 3, then builds the UI)
import server  # noqa: E402

if not PREVIEW:
    import zerogpu_backend  # noqa: E402

branding.apply_to_server(server, app, backend)
long_takes.install_server(server)  # no script limit, empty-note edits, 20 / 60 s references

import reference_trim  # noqa: E402

reference_trim.install(server)  # long reference clips are cut at the latest pause before the limit

if not hasattr(os, "statvfs"):
    server.shared_memory_report = lambda: "unavailable"
if backend == "portable":
    server.ENGINE_IDLE_TIMEOUT_S = max(server.ENGINE_IDLE_TIMEOUT_S, 600.0)  # a constant in the Space
elif backend == "mlx":
    server.ENGINE_IDLE_TIMEOUT_S = max(server.ENGINE_IDLE_TIMEOUT_S, 120.0)  # a long prompt's prefill on a small Mac


ENGINE_LOCK = threading.Lock()  # one take at a time; also held while the experts are re-placed


def serialize_engine() -> None:
    """Hold one lock per take, so the startup warm-up (which bypasses the admission
    gate) never overlaps a visitor's take. The lock is released when the take's
    stream ends or is closed."""
    original = zerogpu_backend.gpu_speech

    def gpu_speech(body, budget_s):
        with ENGINE_LOCK:
            compile_settings()
            if backend == "cuda" and zerogpu_backend.WARM_UP not in body:
                # The memory layout follows the take's voice: a clone voice longer than 20 s
                # gets the 60 s layout, any other take the 20 s one (no-op when unchanged)
                drama3_offload.rebalance(getattr(zerogpu_backend, "_engine", None),
                                         bool(body.get(long_takes.LONG_VOICE_KEY)))
            yield from original(body, budget_s)

    zerogpu_backend.gpu_speech = gpu_speech


def compile_settings() -> None:
    """torch.compile settings for the thread that runs this take.

    The Space sets them once (drama3_fast.compile_layers), in the thread that builds
    the engine. In this torch they are per thread, and every take runs in a new one:
    a take that compiled something new (the 4096-row decode graph that long prompts,
    such as 60 s references, need) did so without emulate_precision_casts and with
    the default recompile limit of 8, which it hit, leaving the rest of the audio
    decoder's layers uncompiled. Applied before every take, warm-up included."""
    if os.environ.get("DRAMA3_COMPILE", "1") == "0":
        return
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    inductor_config.emulate_precision_casts = True
    # Decode graphs exist for 4 KV buckets; layers recompile per bucket and layer kind
    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)


if not PREVIEW:
    serialize_engine()
    # Capture the 4096-row decode graph at startup too: 60 s references and long histories
    # need it on their first take, which would otherwise wait for it to compile
    import drama3_engine as _engine

    if backend != "small":  # drama3-small fits 16 GB with the Space's own buckets (1024 / 2048)
        _engine.WARM_UP_BUCKETS = tuple(sorted({*_engine.WARM_UP_BUCKETS, 4096}))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="info", access_log=False)
