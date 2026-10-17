const path = require('path')
module.exports = {
  version: "7.0",
  title: "Drama 3 Offload Studio",
  description: "Fish Audio Drama 3: expressive multi-speaker TTS with voice cloning, voice design from a description and directing notes. Models: the full model for 24+ GB NVIDIA GPUs (part of it in system RAM), the Small GGUF (drama3-small, Japanese and English) for 16 GB NVIDIA GPUs, and MLX 4-bit / 3-bit conversions of the full model for Apple Silicon (64 / 48 GB; untested on a Mac). https://huggingface.co/fishaudio/drama-3",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("app/env")
    // A model counts as installed once its last file is in place
    const full = info.exists("models/drama-3/model.safetensors.index.json") && info.exists("models/drama-3/codec.pth")
    // Apple Silicon: the MLX conversions of the full model (install_mlx.js), by their last weight file
    const mac = kernel.platform === "darwin"  // install_mlx.js refuses Intel Macs
    const mlx = {
      "4": mac && info.exists("models/drama-3-mlx-4bit/model-00006.safetensors") && info.exists("models/drama-3-mlx-4bit/codec.safetensors"),
      "3": mac && info.exists("models/drama-3-mlx-3bit/model-00005.safetensors") && info.exists("models/drama-3-mlx-3bit/codec.safetensors"),
    }
    const installMlx = {
      "4": {
        icon: "fa-solid fa-plug",
        text: "<div><strong>Install MLX 4-bit (Apple Silicon, 64 GB)</strong><div>Full model on MLX, 26 GB download, needs ~39 GB of unified memory</div></div>",
        href: "install_mlx.js",
        params: { bits: "4" },
      },
      "3": {
        icon: "fa-solid fa-plug",
        text: "<div><strong>Install MLX 3-bit (Apple Silicon, 48 GB)</strong><div>Full model on MLX, 21 GB download, needs ~34 GB of unified memory</div></div>",
        href: "install_mlx.js",
        params: { bits: "3" },
      },
    }
    const small = info.exists("models/drama3-small/drama3-small-IQ3_XXS.gguf")
    const running = {
      install: info.running("install.js"),
      install_mlx: info.running("install_mlx.js"),
      install_small: info.running("install_small.js"),
      start: info.running("start.js"),
      start_small: info.running("start_small.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    const busy = (text, href) => [{ default: true, icon: "fa-solid fa-terminal", text, href }]
    const installFull = {
      icon: "fa-solid fa-plug",
      text: "<div><strong>Install (24+ GB VRAM)</strong><div>Full model, 41 GB download</div></div>",
      href: "install.js",
    }
    const installSmall = {
      icon: "fa-solid fa-plug",
      text: "<div><strong>Install Small GGUF (16 GB VRAM)</strong><div>drama3-small, 10 GB download, Japanese and English</div></div>",
      href: "install_small.js",
    }
    if (running.install) return busy("Installing (24+ GB VRAM)", "install.js")
    if (running.install_small) return busy("Installing Small GGUF", "install_small.js")
    if (running.install_mlx) return busy("Installing MLX", "install_mlx.js")
    if (running.update) return busy("Updating", "update.js")
    if (running.reset) return busy("Resetting", "reset.js")
    for (const script of ["start.js", "start_small.js"]) {
      if (!info.running(script)) continue
      const local = info.local(script)
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url,
        }, {
          icon: "fa-solid fa-diagram-project",
          text: "Workflow",
          href: local.url + "/workflow/",
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: script,
        }]
      }
      return busy("Terminal", script)
    }
    if (!installed || (!full && !small && !mlx["4"] && !mlx["3"])) {
      // Nothing to start yet: choose the model for your GPU (or Mac)
      return mac ? [installMlx["4"], installMlx["3"], installFull] : [installFull, installSmall]
    }
    const items = []
    for (const bits of ["4", "3"]) {
      if (mlx[bits]) items.push({
        default: !items.length, icon: "fa-solid fa-power-off",
        text: `Start (MLX ${bits}-bit, Apple Silicon ${bits === "4" ? "64" : "48"} GB)`,
        href: "start.js", params: { mlx: bits },
      })
    }
    if (full) items.push({ default: !items.length, icon: "fa-solid fa-power-off", text: "Start (24+ GB VRAM)", href: "start.js" })
    if (small) items.push({ default: !full, icon: "fa-solid fa-power-off", text: "Start (Small GGUF, 16 GB VRAM)", href: "start_small.js" })
    if (mac) {
      for (const bits of ["4", "3"]) if (!mlx[bits]) items.push(installMlx[bits])
    }
    if (!full) items.push(installFull)
    if (!small && !mac) items.push(installSmall)
    items.push({
      icon: "fa-solid fa-plug",
      text: "Update",
      href: "update.js",
    }, {
      icon: "fa-regular fa-circle-xmark",
      text: "<div><strong>Reset</strong><div>Revert to pre-install state (the downloaded models are kept)</div></div>",
      href: "reset.js",
      confirm: "Are you sure you wish to reset the app?"
    })
    return items
  }
}
