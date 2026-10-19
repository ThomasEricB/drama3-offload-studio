"""Drama 3 Offload Studio: the Space's UI, labelled for a local run.

The Space says "running on ZeroGPU" in its header and shows a "ZeroGPU" badge (its
server reports no local device, because ZeroGPU attaches one per take). Here the model
runs on this machine, so the header, badge and page titles name the local device and
the offload split instead. The Space's files are not modified: the header is rewritten
as the UI reads it, and the badge comes from a replaced server.gpu_info().
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path

STUDIO_NAME = "Drama 3 Offload Studio"
SPACE_TITLE = "Fish Audio Drama 3"


def device_name(backend: str) -> str:
    """The local device, for the header (the split can change while running)."""
    import torch

    if backend in ("cuda", "small"):
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return device_label(backend)


def device_label(backend: str) -> str:
    """The badge: the device and, on NVIDIA, the model and where it sits."""
    if backend == "small":
        return f"{device_name(backend)} · drama3-small (Small GGUF)"
    return _full_label(backend)


def _full_label(backend: str) -> str:
    """The local device and, on NVIDIA, how many expert layers live in RAM."""
    import torch

    if backend == "cuda":
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        import drama3_offload

        offloaded, layers = drama3_offload.PLACEMENT
        return f"{name} · {offloaded}/{layers} expert layers in RAM" if offloaded else name
    if backend == "mlx":
        import platform

        import drama3_mlx

        return f"{'Apple Silicon' if platform.machine() == 'arm64' else platform.machine()} · {drama3_mlx.model_name()}"
    import drama3_portable

    device = drama3_portable.portable_device()
    return f"{'Apple GPU (MPS)' if device.type == 'mps' else device.type.upper()} · experimental"


def rebrand_page(text: str, device: str) -> str:
    """hero.html (Studio) or index.html (/legacy) with the local name and device."""
    return (text
            .replace("<h1>🎭 Drama 3</h1>", f"<h1>🎭 {STUDIO_NAME}</h1>")
            .replace("by Fish Audio · running on ZeroGPU", f"Model by Fish Audio · running locally on {device}")
            .replace(f"<title>{SPACE_TITLE}</title>", f"<title>{STUDIO_NAME}</title>"))


@contextlib.contextmanager
def rebranded_ui(backend: str):
    """While the Space loads the model and builds its UI: rewrite hero.html as it is
    read, and rename the Gradio page title. Both patches are removed afterwards. The
    device label is computed when hero.html is read, after the weights (and so the
    offload split) are placed."""
    import gradio as gr

    read_text, blocks_init = Path.read_text, gr.Blocks.__init__

    def patched_read_text(self, *args, **kwargs):
        if not args:  # the Space's files are UTF-8; Windows would read them as cp1252
            kwargs.setdefault("encoding", "utf-8")
        text = read_text(self, *args, **kwargs)
        return rebrand_page(text, device_name(backend)) if self.name == "hero.html" else text

    @functools.wraps(blocks_init)
    def patched_blocks_init(self, *args, **kwargs):
        if kwargs.get("title") == SPACE_TITLE:
            kwargs["title"] = STUDIO_NAME
        blocks_init(self, *args, **kwargs)

    Path.read_text, gr.Blocks.__init__ = patched_read_text, patched_blocks_init
    try:
        yield
    finally:
        Path.read_text, gr.Blocks.__init__ = read_text, blocks_init


class _RebrandedPage:
    """Stands in for server.INDEX_HTML, which /legacy reads on every request."""

    def __init__(self, path: Path, device: str) -> None:
        self.path, self.device = path, device

    def read_text(self, *args, **kwargs) -> str:
        return rebrand_page(self.path.read_text(*args, **kwargs), self.device)


def apply_to_server(server, app, backend: str) -> None:
    """After the server is built: the badge (recomputed, as the split can change), the
    /legacy page and the API title."""
    server.gpu_info = lambda: {"name": device_label(backend), "remote": False}
    server.INDEX_HTML = _RebrandedPage(server.INDEX_HTML, device_name(backend))
    app.title = STUDIO_NAME
