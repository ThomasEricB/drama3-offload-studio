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
          // drama3-small, as install_small.js downloaded it (serve.py refuses to start, instead of downloading, if missing)
          DRAMA3_GGUF: "{{path.resolve(cwd, 'models/drama3-small/drama3-small-IQ3_XXS.gguf')}}",
          // Pause fix (from drama3-small): after this many silent frames (10 = 0.47 s) silence is masked
          // from the next draw, and a length guard ends takes that never stop; 0 = off
          DRAMA3_SILENCE_CODES: "{{path.resolve(cwd, 'models/drama3-small/silence_codes.json')}}",
          DRAMA3_MAX_PAUSE_FRAMES: "10",
          // Stall guard (offload/stall_guard.py): a take that turns into a low drone is ended where the drone
          // began, before it is heard (a long script's chunk is made again once); 0 = off
          DRAMA3_STALL_GUARD: "1",
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
        // offload/serve.py loads drama3-small from the GGUF (DRAMA3_GGUF) into the Space's CUDA engine and serves
        // it on 127.0.0.1: no audio.cpp, same Studio
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
