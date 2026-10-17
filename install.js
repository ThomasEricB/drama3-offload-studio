module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Unsupported machines stop here with a message instead of a failed 40 GB install
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "Drama 3 needs an Apple Silicon Mac (PyTorch 2.13 has no Intel macOS build)."
      },
      next: null
    },
    {
      when: "{{platform !== 'darwin' && gpu !== 'nvidia'}}",
      method: "notify",
      params: {
        html: "Drama 3 needs an NVIDIA GPU (RTX 40-series or newer, 24 GB or more for this model; 16 GB: use Install Small GGUF) on Windows and Linux."
      },
      next: null
    },
    {
      when: "{{platform === 'darwin' && os.totalmem() < 64 * 1024 ** 3}}",
      method: "notify",
      params: {
        html: "Warning: Drama 3 needs about 45 GB of memory for its weights; this Mac has less than 64 GB and will likely run out."
      }
    },
    // The app and its Python environment (shared with Install Small GGUF)
    {
      method: "script.start",
      params: {
        uri: "setup.js"
      }
    },
    // The checkpoint (~41 GB) at the revision drama3_engine.py pins (MODEL_REVISION), into models/drama-3.
    // start.js points the engine at this folder (DRAMA3_CHECKPOINT_DIR), so it never depends on HF_HOME,
    // which Pinokio may set per app; re-running skips files that are already complete.
    {
      method: "hf.download",
      params: {
        "_": [
          "fishaudio/drama-3",
          "codec.pth",
          "config.json",
          "model-00001-of-00008.safetensors",
          "model-00002-of-00008.safetensors",
          "model-00003-of-00008.safetensors",
          "model-00004-of-00008.safetensors",
          "model-00005-of-00008.safetensors",
          "model-00006-of-00008.safetensors",
          "model-00007-of-00008.safetensors",
          "model-00008-of-00008.safetensors",
          "model.safetensors.index.json",
          "special_tokens_map.json",
          "tokenizer.json",
          "tokenizer_config.json"
        ],
        revision: "e0a9218aa48888270f06ef1ed9ea67622c0e17d2",
        "local-dir": "models/drama-3"
      }
    }
  ]
}
