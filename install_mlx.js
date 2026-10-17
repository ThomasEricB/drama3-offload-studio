// The full model on Apple Silicon through MLX (offload/drama3_mlx.py, run by mlx-audio).
// pinokio.js passes args.bits: "4" (fcmeyer/drama-3-MLX-4bit-HQ) or "3" (fcmeyer/drama-3-MLX-3bit-HQ).
module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // MLX runs on Apple Silicon Macs only
    {
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "The MLX models need an Apple Silicon Mac. On NVIDIA GPUs, use Install (24+ GB VRAM) or Install Small GGUF (16 GB VRAM)."
      },
      next: null
    },
    {
      when: "{{args.bits === '4' && os.totalmem() < 64 * 1024 ** 3}}",
      method: "notify",
      params: {
        html: "Warning: the MLX 4-bit model peaks at about 39 GB of memory and needs a Mac with 64 GB (48 GB only with a raised GPU memory limit, see the README). The 3-bit model fits 48 GB."
      }
    },
    {
      when: "{{args.bits === '3' && os.totalmem() < 48 * 1024 ** 3}}",
      method: "notify",
      params: {
        html: "Warning: the MLX 3-bit model peaks at about 34 GB of memory and needs a Mac with 48 GB; this Mac has less and will likely run out."
      }
    },
    // The app and its Python environment (on macOS it includes MLX and mlx-audio)
    {
      method: "script.start",
      params: {
        uri: "setup.js"
      }
    },
    // Experts 4-bit, the rest bfloat16: 24.5 GB + 1.6 GB codec, into models/drama-3-mlx-4bit
    {
      when: "{{args.bits === '4'}}",
      method: "hf.download",
      params: {
        "_": [
          "fcmeyer/drama-3-MLX-4bit-HQ",
          "config.json",
          "tokenizer.json",
          "tokenizer_config.json",
          "special_tokens_map.json",
          "chat_template.jinja",
          "codec.safetensors",
          "model-00001.safetensors",
          "model-00002.safetensors",
          "model-00003.safetensors",
          "model-00004.safetensors",
          "model-00005.safetensors",
          "model-00006.safetensors",
          "LICENSE",
          "NOTICE"
        ],
        revision: "fd137b475117e6d776e4db72aced62be82361743",
        "local-dir": "models/drama-3-mlx-4bit"
      }
    },
    // Experts 3-bit (the most sensitive layers 4-bit), attention and tower 8-bit: 19.7 GB + 1.6 GB codec,
    // into models/drama-3-mlx-3bit
    {
      when: "{{args.bits === '3'}}",
      method: "hf.download",
      params: {
        "_": [
          "fcmeyer/drama-3-MLX-3bit-HQ",
          "config.json",
          "tokenizer.json",
          "tokenizer_config.json",
          "special_tokens_map.json",
          "chat_template.jinja",
          "codec.safetensors",
          "model-00001.safetensors",
          "model-00002.safetensors",
          "model-00003.safetensors",
          "model-00004.safetensors",
          "model-00005.safetensors",
          "LICENSE",
          "NOTICE"
        ],
        revision: "fb68eb7bb1786ed7781e5b87d0d2a381f3e7d089",
        "local-dir": "models/drama-3-mlx-3bit"
      }
    }
  ]
}
