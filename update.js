module.exports = {
  run: [{
    // Only when the launcher itself was installed from a git repository
    when: "{{exists('.git')}}",
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    // The app and its Python environment: checks out the Space commit the launcher pins
    method: "script.start",
    params: {
      uri: "setup.js"
    }
  }, {
    // Re-verify only the models that are installed (complete files are skipped, nothing new is downloaded)
    when: "{{exists('models/drama-3/model.safetensors.index.json')}}",
    method: "script.start",
    params: {
      uri: "install.js"
    }
  }, {
    when: "{{exists('models/drama3-small/drama3-small-IQ3_XXS.gguf')}}",
    method: "script.start",
    params: {
      uri: "install_small.js"
    }
  }, {
    when: "{{exists('models/drama-3-mlx-4bit/config.json')}}",
    method: "script.start",
    params: {
      uri: "install_mlx.js",
      params: { bits: "4" }
    }
  }, {
    when: "{{exists('models/drama-3-mlx-3bit/config.json')}}",
    method: "script.start",
    params: {
      uri: "install_mlx.js",
      params: { bits: "3" }
    }
  }]
}
