"""Studio controls for long scripts, long-take edits and 60 s references.

The Space builds its Studio in gradio_ui.build_apps; these hooks add to it while it
is built, without changing the Space's files:

- Clone panel: the "trimmed to 20 s" hint follows the reference limit, with a
  checkbox that unlocks clips up to 60 s.
- Under the stage: "Retake / Edit long takes (experimental)", and a line that says
  how the script will be chunked and which chunks the next Retake / Edit regenerates.
- The script's character counter shows the chunk count instead of "/ 600", and the
  language / duration hints.
- In the same card: "Read ahead across chunks" (lookahead), "Edit by infilling", and
  subtitles: import an .srt as a timed script, download the .srt of the current take.
- A box of its own between the Voice card and Sampling: "Check every word before
  playing (Quality Mode)" (word_check.py), and folded under it like Sampling, its
  "Voice guidance" strength (tag_guidance.py; full model only), greyed out while the
  check is off.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys

import chunking
import long_takes
import subtitles

REF_HINT_SPACE = "Clips are trimmed to 20 s and transcribed for you."
SMALL = bool(os.environ.get("DRAMA3_GGUF"))  # Start (Small GGUF): drama3-small
UNLOCK_INFO = (
    "Applies to clips added after ticking; they are marked “· 60 s” in the voice list and keep working after "
    "you untick. Longer clips make a longer prompt: more GPU memory and a slower first audio. drama3-small was "
    "tuned with 10-30 s references." if SMALL else
    "Applies to clips added after ticking; they are marked “· 60 s” in the voice list and keep working after "
    "you untick. A take with such a voice uses a memory layout with more of the model in RAM (a little slower); "
    "the next take with a shorter voice goes back to the faster one. Switching happens between takes, ~1 s.")
# drama3-small's recommended sampling (its model card); the Space's are 1.0 / 0.9 / all
SAMPLING_DEFAULTS = {"Temperature": 0.7, "Top-p": 0.7, "Top-k": 30} if SMALL else {}
NOTE_PLACEHOLDER_SPACE = "e.g. more amused than before; drop the hurt"
NOTE_PLACEHOLDER = "e.g. more amused than before · or leave empty and edit the script to re-read it"


def server():
    """The Space's server module (imported while the Studio is built, used after)."""
    return sys.modules["server"]


def ref_hint() -> str:
    seconds = int(long_takes.reference_limit())
    return (f"Clips are trimmed to {seconds} s (at a pause) and transcribed for you. "
            "Only clone voices you have permission to use.")


HINTS = "Hints: `[language: fr]` sets the main language, `[audio tokens: 65]` asks for about 3 s."


def counter(text: str) -> str:
    if subtitles.is_timed(text or ""):
        cues = len(subtitles.timed_lines(text))
        return (f"Tags in `[ ]` and `< >` are direction, not words. · **{cues} subtitle cues**, each voiced at "
                "its time and length")
    chunks = len(chunking.split(text or ""))
    extra = f" · {chunks} chunks, generated one after another" if chunks > 1 else ""
    return f"Tags in `[ ]` and `< >` are direction, not words. {HINTS} · **{len(text or '')} characters**{extra}"


def import_srt(path) -> str:
    """The script box's new text: the uploaded .srt as a timed script."""
    import gradio as gr

    with open(path, encoding="utf-8-sig", errors="replace") as f:
        script = subtitles.from_srt(f.read())
    if not script:
        raise gr.Error("No subtitle cues found in that file.")
    return script


def export_srt(take_id):
    """An .srt file of the selected take (Whisper on its audio, aligned to its script)."""
    import tempfile

    import gradio as gr

    take = server().store.take(take_id) if take_id else None
    if take is None:
        raise gr.Error("Make or select a take first.")
    if server().transcriber.state != "ready":
        raise gr.Error("Whisper is still loading; try again in a moment.")
    import infill

    words = infill.transcribe_words(server().transcriber, take.pcm)
    srt = subtitles.to_srt(long_takes.parse_retake(take.line)[1], words, take.seconds)
    path = os.path.join(tempfile.mkdtemp(prefix="drama3-srt-"), f"drama3-{take_id}.srt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt)
    return gr.File(value=path, visible=True)


def explain(script: str, note: str, current, enabled: bool):
    """The line under the stage, and whether to show it."""
    import gradio as gr

    lines = []
    chunks = chunking.split(script or "")
    if subtitles.is_timed(script or ""):
        lines.append(f"Timed script: {len(subtitles.timed_lines(script))} subtitle cues, each voiced as its own chunk "
                     "with a length hint and placed at its start time.")
    elif len(chunks) > 1:
        lines.append(f"This script is {len(chunks)} chunks (about {chunking.CHUNK_BYTES} bytes each); "
                     "they play as one take, keeping the voice from chunk to chunk.")
    take = server().store.take(current) if current else None
    previous = long_takes.long_take(take.turn_text) if take is not None else None
    if previous is not None:
        if not enabled:
            lines.append(f"The selected take has {len(previous.chunks)} chunks: tick the box to Retake or Edit it.")
        else:
            planned = chunking.plan_edit(previous.line, [c.text for c in previous.chunks], script or "")
            lines.append(chunking.describe(planned, len(previous.chunks), note=(note or "").strip() or None))
    return gr.Markdown(value="\n\n".join(lines), visible=bool(lines))


@contextlib.contextmanager
def extended_studio():
    """While the Space builds its UI: add the controls above (hooks removed after)."""
    import gradio as gr

    found = {}
    markdown_init, textbox_init, state_init, audio_init = (
        gr.Markdown.__init__, gr.Textbox.__init__, gr.State.__init__, gr.Audio.__init__)
    textbox_input = gr.Textbox.input
    slider_init, button_click = gr.Slider.__init__, gr.Button.click
    accordion_init = gr.Accordion.__init__

    def guidance_classes(checked):
        return ["voice-guidance"] + ([] if checked else ["voice-guidance-off"])

    def word_check_controls():
        """"Check every word" and its voice guidance (folded away, like Sampling)."""
        # (a folded Accordion's contents are not rendered, so the style is outside it)
        gr.HTML("<style>.voice-guidance-off { opacity: 0.4; } #voice-guidance-style { display: none; }</style>",
                padding=False, container=False, elem_id="voice-guidance-style")
        found["word_box"] = gr.Checkbox(
            value=long_takes.SETTINGS["word_check"], label="Check every word before playing (Quality Mode)",
            info="**DISABLES STREAMING AND REAL-TIME GENERATION!** Each take is held until Whisper confirms every "
                 "word was said, and made again (up to twice) if words are missing or a direction was read aloud. "
                 "Takes then play once made instead of streaming. Off: takes stream, and only their opening is "
                 "checked.")
        with gr.Accordion("Voice guidance", open=False, visible=not SMALL,
                          elem_classes=guidance_classes(long_takes.SETTINGS["word_check"])) as box:
            found["guidance_box"] = box
            found["voice_slider"] = gr.Slider(
                0, 3, value=long_takes.SETTINGS["voice_guidance"], step=0.5, label="Voice guidance",
                show_label=False,
                info="With “Check every word” on: a cloned voice (and its Retakes) is made closer to its clip, "
                     "at about 1.5x the time. 2 measured best; 0 = off. A clip's background music is copied "
                     "more too.")

    @functools.wraps(accordion_init)
    def on_accordion(self, *args, **kwargs):
        label = args[0] if args else kwargs.get("label")
        if label == "Sampling" and "word_box" not in found:
            # A box of its own between the Voice card and Sampling
            with gr.Column(elem_classes="card", elem_id="word-check"):
                word_check_controls()
        accordion_init(self, *args, **kwargs)

    @functools.wraps(slider_init)
    def on_slider(self, *args, **kwargs):
        if kwargs.get("label") in SAMPLING_DEFAULTS:
            if len(args) >= 3:
                args = (*args[:2], SAMPLING_DEFAULTS[kwargs["label"]], *args[3:])
            else:
                kwargs["value"] = SAMPLING_DEFAULTS[kwargs["label"]]
        slider_init(self, *args, **kwargs)

    @functools.wraps(button_click)
    def on_click(self, fn=None, *args, **kwargs):
        if getattr(self, "value", None) == "Reset sampling" and getattr(fn, "__name__", "") == "<lambda>":
            fn = lambda: tuple(SAMPLING_DEFAULTS.values())  # noqa: E731
        return button_click(self, fn, *args, **kwargs)


    @functools.wraps(markdown_init)
    def on_markdown(self, value=None, *args, **kwargs):
        if isinstance(value, str) and value.startswith(REF_HINT_SPACE):
            markdown_init(self, ref_hint(), *args, **kwargs)
            unlock = gr.Checkbox(value=long_takes.SETTINGS["ref_unlocked"], label="Allow reference clips up to 60 s",
                                 info=UNLOCK_INFO)

            def toggle(unlocked):
                long_takes.set_ref_unlocked(server(), unlocked)
                return ref_hint()

            unlock.change(toggle, inputs=unlock, outputs=self, queue=False, api_visibility="private")
            found["unlock"], found["ref_hint"] = unlock, self
            return
        markdown_init(self, value, *args, **kwargs)

    @functools.wraps(textbox_init)
    def on_textbox(self, *args, **kwargs):
        if kwargs.get("placeholder") == NOTE_PLACEHOLDER_SPACE:
            kwargs["placeholder"] = NOTE_PLACEHOLDER
        textbox_init(self, *args, **kwargs)
        if kwargs.get("elem_id") in ("script", "stage-note"):
            found[kwargs["elem_id"]] = self

    @functools.wraps(state_init)
    def on_state(self, *args, **kwargs):
        state_init(self, *args, **kwargs)
        found.setdefault("current", self)  # build_apps creates `current` first

    @functools.wraps(textbox_input)
    def on_input(self, fn=None, *args, **kwargs):
        if self is found.get("script") and getattr(fn, "__name__", "") == "<lambda>":
            fn = counter  # the Space's counter says "/ 600"
        return textbox_input(self, fn, *args, **kwargs)

    @functools.wraps(audio_init)
    def on_audio(self, *args, **kwargs):
        audio_init(self, *args, **kwargs)
        if kwargs.get("streaming") and kwargs.get("visible") is False and "long" not in found:
            found["long"] = True
            with gr.Column(elem_classes="card", elem_id="long-takes"):
                enabled = gr.Checkbox(value=long_takes.SETTINGS["long_retake"],
                                      label="Retake / Edit long takes (experimental)",
                                      info="For takes generated in several chunks. With an empty note, Retake becomes an "
                                           "edit: only the chunks whose words changed are regenerated, the rest keep "
                                           "their audio. With a note, every chunk is redone.")
                plan = gr.Markdown(visible=False, elem_classes="hint")
                lookahead = gr.Checkbox(value=long_takes.SETTINGS["lookahead"], label="Read ahead across chunks",
                                        info="Each chunk of a long script is voiced knowing the next one (Drama 3's "
                                             "streaming-input format), for smoother joins.")
                infill_box = gr.Checkbox(value=long_takes.SETTINGS["infill"], label="Edit by infilling",
                                         info="With an empty note, an edit regenerates only the words that changed "
                                              "and keeps the audio around them; large edits redo the chunk.")
                if "word_box" not in found:  # the Space's UI had no Sampling box to put them before
                    word_check_controls()
                word_box, voice_slider = found["word_box"], found["voice_slider"]
                with gr.Row():
                    srt_in = gr.UploadButton("Import subtitles (.srt)", file_types=[".srt"], size="sm")
                    srt_out = gr.Button("Subtitles of this take (.srt)", size="sm")
                srt_file = gr.File(visible=False, label="Subtitles", interactive=False)
            script, note, current = found["script"], found["stage-note"], found["current"]

            def setting(name):
                def store(value):
                    long_takes.SETTINGS[name] = bool(value)
                return store

            lookahead.change(setting("lookahead"), inputs=lookahead, queue=False, api_visibility="private")
            infill_box.change(setting("infill"), inputs=infill_box, queue=False, api_visibility="private")
            word_box.change(setting("word_check"), inputs=word_box, queue=False, api_visibility="private")

            def guidance_state(checked):  # greyed out while the word check is off: it does nothing then
                return (gr.Accordion(elem_classes=guidance_classes(checked)),
                        gr.Slider(interactive=bool(checked)))

            guidance = [found["guidance_box"], voice_slider]
            word_box.change(guidance_state, inputs=word_box, outputs=guidance, queue=False, api_visibility="private")

            def set_voice_guidance(value):
                long_takes.SETTINGS["voice_guidance"] = float(value)

            voice_slider.change(set_voice_guidance, inputs=voice_slider, queue=False, api_visibility="private")
            srt_in.upload(import_srt, inputs=srt_in, outputs=script, api_visibility="private")
            srt_out.click(export_srt, inputs=current, outputs=srt_file, api_visibility="private")

            def toggle(value):
                long_takes.SETTINGS["long_retake"] = bool(value)

            def update(text, note_text, take_id, value):
                return explain(text, note_text, take_id, value)

            enabled.change(toggle, inputs=enabled, queue=False, api_visibility="private")

            # On every page load, show the server's current settings (not the ones it started with)
            from gradio.context import Context

            def current_settings():
                return (gr.Checkbox(value=long_takes.SETTINGS["ref_unlocked"]), ref_hint(),
                        gr.Checkbox(value=long_takes.SETTINGS["long_retake"]),
                        gr.Checkbox(value=long_takes.SETTINGS["lookahead"]),
                        gr.Checkbox(value=long_takes.SETTINGS["infill"]),
                        gr.Checkbox(value=long_takes.SETTINGS["word_check"]),
                        gr.Slider(value=long_takes.SETTINGS["voice_guidance"]))

            Context.root_block.load(current_settings,
                                    outputs=[found["unlock"], found["ref_hint"], enabled, lookahead, infill_box, word_box,
                                             voice_slider],
                                    queue=False, api_visibility="private")
            Context.root_block.load(lambda: guidance_state(long_takes.SETTINGS["word_check"]), outputs=guidance,
                                    queue=False, api_visibility="private")
            inputs = [script, note, current, enabled]
            for event in (enabled.change, script.change, note.change, current.change):
                event(update, inputs=inputs, outputs=plan, queue=False, api_visibility="private")

    gr.Markdown.__init__, gr.Textbox.__init__, gr.State.__init__, gr.Audio.__init__ = (
        on_markdown, on_textbox, on_state, on_audio)
    gr.Textbox.input = on_input
    gr.Accordion.__init__ = on_accordion
    if SAMPLING_DEFAULTS:
        gr.Slider.__init__, gr.Button.click = on_slider, on_click
    try:
        yield
    finally:
        gr.Markdown.__init__, gr.Textbox.__init__, gr.State.__init__, gr.Audio.__init__ = (
            markdown_init, textbox_init, state_init, audio_init)
        gr.Textbox.input = textbox_input
        gr.Accordion.__init__ = accordion_init
        gr.Slider.__init__, gr.Button.click = slider_init, button_click
