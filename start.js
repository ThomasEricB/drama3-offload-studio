module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // faster-whisper (reference-clip transcription, CPU) caches its model here instead of ~/.cache/whisper
          WHISPER_DIR: "{{path.resolve(cwd, 'cache/whisper')}}",
          // Whisper runs on the CPU (reference transcripts, infill edits, subtitles); the Space uses 2 threads
          WHISPER_CPU_THREADS: "{{Math.max(2, Math.min(8, os.cpus().length))}}",
          // The checkpoint install.js downloaded; serve.py refuses to start (instead of downloading 41 GB) if it is incomplete
          DRAMA3_CHECKPOINT_DIR: "{{path.resolve(cwd, 'models/drama-3')}}",
          // Apple Silicon: the MLX entries of the menu pass args.mlx ("4" or "3") to run the MLX model
          // install_mlx.js downloaded (offload/drama3_mlx.py). Without it, the backend is chosen as before:
          // the CUDA engine on NVIDIA, else the eager PyTorch engine (MPS or CPU) on the checkpoint above.
          DRAMA3_BACKEND: "{{args.mlx ? 'mlx' : 'auto'}}",
          DRAMA3_MLX_DIR: "{{args.mlx ? path.resolve(cwd, 'models/drama-3-mlx-' + args.mlx + 'bit') : ''}}",
          // Pause fix (from drama3-small): after this many silent frames (10 = 0.47 s) silence is masked
          // from the next draw, and a length guard ends takes that never stop; 0 = off
          DRAMA3_SILENCE_CODES: "{{path.resolve(cwd, 'models/pause-fix/silence_codes.json')}}",
          DRAMA3_MAX_PAUSE_FRAMES: "10",
          // Stall guard (offload/stall_guard.py): a take that turns into a low drone is ended where the drone
          // began, before it is heard (a long script's chunk is made again once); 0 = off
          DRAMA3_STALL_GUARD: "1",
          // MoE expert offload (offload/drama3_offload.py): "auto" keeps as many expert layers on the GPU as fit,
          // the rest in pinned RAM. A number forces that many layers to RAM; 0 is the Space's all-on-GPU behaviour.
          DRAMA3_OFFLOAD_LAYERS: "auto",
          // VRAM that "auto" leaves free for the engine's working memory and the desktop, per reference limit
          // (measured working memory ~6.5 GiB with 20 s references, ~8.7 GiB with 60 s ones; ticking the
          // 60 s box moves experts between GPU and RAM to switch between the two)
          DRAMA3_VRAM_RESERVE_GB: "9",
          DRAMA3_VRAM_RESERVE_60S_GB: "12",
          // torch.compile speeds decode ~25%, but on Windows it can need the MSVC build tools; off there by default
          DRAMA3_COMPILE: "{{platform === 'win32' ? '0' : '1'}}",
          // Initial state of the Studio checkboxes (also the setting for API-only use):
          // reference clips up to 60 s instead of 20 s, experimental Retake / Edit of multi-chunk takes,
          // long-script chunks voiced knowing the next one, and empty-note edits by infilling the changed words
          DRAMA3_REF_UNLOCKED: "0",
          DRAMA3_LONG_RETAKE: "0",
          DRAMA3_LOOKAHEAD: "1",
          DRAMA3_INFILL: "1"
        },
        path: "app",
        // offload/serve.py picks the backend (NVIDIA: CUDA engine + expert offload; Mac: experimental eager MPS engine),
        // then imports app.py (loads Drama 3) and serves it on 127.0.0.1. start_small.js runs drama3-small instead.
        message: [
          "python ../offload/serve.py {{port}}",
        ],
        on: [{
          // Uvicorn prints its URL only after the weights are loaded and the lifespan has started
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
