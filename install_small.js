module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // drama3-small runs on the Space's CUDA engine: an FP8-capable NVIDIA GPU (16 GB or more)
    {
      when: "{{platform === 'darwin'}}",
      method: "notify",
      params: {
        html: "The Small GGUF needs an NVIDIA GPU (Linux or Windows). On an Apple Silicon Mac, use Install MLX 4-bit (64 GB) or Install MLX 3-bit (48 GB)."
      },
      next: null
    },
    {
      when: "{{gpu !== 'nvidia'}}",
      method: "notify",
      params: {
        html: "The Small GGUF needs an NVIDIA GPU (RTX 40-series or newer, 16 GB or more)."
      },
      next: null
    },
    // The app and its Python environment (shared with Install (24+ GB VRAM))
    {
      method: "script.start",
      params: {
        uri: "setup.js"
      }
    },
    // drama3-small (9.7 GiB) at the commit this launcher was tested with; re-running skips complete files
    {
      method: "hf.download",
      params: {
        "_": [
          "pentacoxian-dev/drama3-small-GGUF",
          "drama3-small-IQ3_XXS.gguf",
          "silence_codes.json",
          "LICENSE",
          "NOTICE"
        ],
        revision: "4a3078e885344bb9b24d511cec389d2726854d15",
        "local-dir": "models/drama3-small"
      }
    }
  ]
}
