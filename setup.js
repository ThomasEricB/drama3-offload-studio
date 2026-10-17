// Shared by install.js (full model) and install_small.js (drama3-small): the app and its
// Python environment, without any model. Safe to re-run (update.js does).
module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // The Space itself: Studio + Workflow UI, the in-process Drama 3 engine and the vendored fish_speech model code
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://huggingface.co/spaces/fishaudio/Drama-3 app",
        ]
      }
    },
    // offload/ hooks into the Space's engine internals: use the Space commit it was tested with.
    // This commit pins checkpoint revision e0a9218 (MODEL_REVISION).
    {
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "git fetch origin",
          "git checkout --detach c368729ab9347dda6635c5272ad31e6521114b79"
        ]
      }
    },
    // requirements.txt pins torch==2.13.0. Linux: the PyPI wheel is the CUDA 13 build and bundles triton.
    // gguf reads drama3-small. torch.js is not used: it would replace the torch pin with an older torch.
    {
      when: "{{platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt gguf==0.18.0"
        ]
      }
    },
    // Windows: PyPI's torch is CPU-only, so 2.13.0+cu130 comes from the PyTorch index; Triton is the
    // triton-windows build matching the triton 3.7.1 that torch 2.13 uses on Linux
    {
      when: "{{platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt gguf==0.18.0 triton-windows==3.7.1.post27 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match"
        ]
      }
    },
    // macOS (Apple Silicon, macOS 14+): MLX and mlx-audio run the model (offload/drama3_mlx.py); torch (with MPS)
    // still runs the Space and is the fallback engine (offload/drama3_portable.py). mlx-audio is pinned to the
    // commit the MLX model was published for (its Drama 3 port).
    {
      when: "{{platform === 'darwin'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt gguf==0.18.0 mlx==0.32.3 \"mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@dad2087f790dfe96b07d5aac1405db4b578b411f\""
        ]
      }
    },
    // The pause fix's silence codes (813 bytes), published with drama3-small: codec codes, so
    // they apply to both models (same codec)
    {
      method: "hf.download",
      params: {
        "_": ["pentacoxian-dev/drama3-small-GGUF", "silence_codes.json", "LICENSE", "NOTICE"],
        revision: "4a3078e885344bb9b24d511cec389d2726854d15",
        "local-dir": "models/pause-fix"
      }
    }
  ]
}
