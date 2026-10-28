"""Studio controls for long scripts, long-take edits and 60 s references.

The Space builds its Studio in gradio_ui.build_apps; these hooks add to it while it
is built, without changing the Space's files:

- Clone panel: the "trimmed to 20 s" hint follows the reference limit, with a
  checkbox that unlocks clips up to 60 s.
- Under the stage: "Retake / Edit long takes (experimental)", and a line that says
  how the script will be chunked and which chunks the next Retake / Edit regenerates.
- The script's character counter shows the chunk count instead of "/ 600", and the
  language / duration hints.
- The stage's take history: a click on a take's row selects it, without playing it.
- The script box's text cursor shows on its highlighted [cues] and <tags> too, and stays
  on its words in a script longer than the box (EXTRAS_CSS).
- In the same card: "Read ahead across chunks" (lookahead), "Trim chunk tails" and
  "Crossfade chunk joins" (how chunks join), "Edit by infilling", and subtitles: import an
  .srt as a timed script, download the .srt of the current take.
- Between the Voice card and Sampling, an "Experimental" card: "Enable Experimental
  Changes" (upstream_sampling.py, register_guard.py; off by default) and its sampler (the
  community fixes or Fish reference sampling), greyed out while the box is off. Under it,
  a box of its own: "Length Hint" (off by
  default), "Check every word before playing (Quality Mode)" (word_check.py), and folded
  under it like Sampling, its "Voice guidance" strength (tag_guidance.py; full model
  only), greyed out while the check is off.
- Sampling: the seed (seeds.py), a field that opens empty: a new random seed per take
  (a Retake keeps its take's); a number in it (typed, the last take's, or a new one) is
  kept for every take. Each take shows its seed in the stage: a click on it keeps it.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
from pathlib import Path

import chunking
import long_takes
import seeds
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
# The script's highlighting is a mirror of the textarea, whose own text is transparent.
# - The caret showed only where the mirror is transparent, never on a [cue] or <tag> chip:
#   the textarea goes on top instead, its background transparent so the chips show through.
# - Past the box's height the caret drifted from the words, further on every line: the
#   mirror wrapped wider than the textarea, which had a scrollbar (13 px) and, at a browser
#   zoom, a fractional width the mirror's whole-pixel one rounded off. Both now keep the
#   same thin scrollbar gutter, and the mirror spans its wrapper (the textarea's exact width).
EXTRAS_CSS = """
#studio-extras-style { display: none; }
#experimental .exp-sampler { gap: 6px !important; padding: 10px 2px 0; border-top: 1px solid var(--border); }
#experimental .exp-sampler-off { opacity: 0.4; }
#experimental .exp-label { font-size: 12px; font-weight: 600; color: var(--muted); letter-spacing: .02em; }
#experimental-sampler .wrap { display: flex; flex-wrap: nowrap; background: var(--soft); border-radius: 999px;
  padding: 3px; gap: 3px; }
#experimental-sampler label { flex: 1; justify-content: center; border: 0; border-radius: 999px; background: transparent;
  padding: 7px 6px; font-size: 13px; white-space: nowrap; }
#experimental-sampler input { position: absolute; opacity: 0; width: 1px; height: 1px; }
#experimental-sampler label.selected, #experimental-sampler label:has(input:checked) { background: var(--card);
  box-shadow: 0 1px 2px #14141414; }
@media (max-width: 560px) { #experimental-sampler .wrap { flex-wrap: wrap; border-radius: 16px; } }
#experimental-sampler-group .hint { padding: 0 8px; }
#experimental-details .hint li, #experimental-details .hint p { color: var(--muted); font-size: 12.5px; }
.voice-guidance-off { opacity: 0.4; }
#script textarea.drama-highlighted { position: relative; z-index: 3; background: transparent !important; }
.stage-item[role="button"] { cursor: pointer; }
.stage-item[role="button"]:hover { border-color: var(--muted); }
.stage-item.current[role="button"]:hover { border-color: var(--ink); }
#script textarea.drama-highlighted, .gradio-container .drama-script-mirror {
  scrollbar-gutter: stable; scrollbar-width: thin; }
.gradio-container .drama-script-mirror {
  left: 0 !important; right: 0 !important; width: auto !important; scrollbar-color: transparent transparent; }
.stage-seed { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer;
  text-decoration: underline dotted; text-underline-offset: 2px; }
.stage-seed:hover { color: var(--ink); }
#seed-box { display: none !important; }
#seed-widget .seed-ui { display: grid; gap: 10px; padding: 12px 14px; border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg); background: var(--block-background-fill); color: var(--body-text-color); }
#seed-widget .seed-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
#seed-widget .seed-text { min-width: 0; }
#seed-widget .seed-title { display: block; font-weight: 600; }
#seed-widget .seed-hint { display: block; margin-top: 2px; font-size: 12px; color: var(--body-text-color-subdued); }
#seed-widget .seed-badge { flex: none; padding: 3px 10px; border: 1px solid var(--border-color-primary);
  border-radius: 999px; font-size: 12px; color: var(--body-text-color-subdued); white-space: nowrap; }
#seed-widget .seed-ui[data-mode="fixed"] .seed-badge { background: var(--body-text-color);
  border-color: var(--body-text-color); color: var(--background-fill-primary); font-weight: 600; }
#seed-widget .seed-row { display: flex; gap: 6px; align-items: stretch; }
#seed-widget .seed-value { flex: 1 1 auto; min-width: 0; padding: 6px 10px; border: 1px solid var(--border-color-primary);
  border-radius: 8px; background: var(--input-background-fill); color: var(--body-text-color);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 15px; letter-spacing: .05em; }
#seed-widget .seed-value::placeholder { color: var(--body-text-color-subdued); font-family: inherit; letter-spacing: 0; }
#seed-widget .seed-value:focus { outline: none; border-color: var(--body-text-color-subdued); }
#seed-widget .seed-act { flex: none; padding: 6px 12px; border: 1px solid var(--border-color-primary); border-radius: 8px;
  background: none; color: var(--body-text-color); font: inherit; font-size: 12px; cursor: pointer; white-space: nowrap; }
#seed-widget .seed-act:hover:not(:disabled) { border-color: var(--body-text-color-subdued); }
#seed-widget .seed-act:disabled { opacity: .4; cursor: default; }
"""
# The stage's take history (stage.js) selected a take only through its ▶ Play button:
# a click anywhere on its row selects it now (it becomes the take that Retake, an edit
# and Keep this voice act on, its line back in the script) without playing it.
STAGE_PLAY = """    play.addEventListener('click', () => {
      if (state.busy) return;
      player.reset(); state.current = take; state.take = take;
      $('audio').src = take.url; $('audio').hidden = false; show(take); slate();
      player.ensure(); $('audio').play().catch(() => {});
      if (take.id && !take.partial) trigger('click', {action: 'select', take_id: take.id});
    });
"""
STAGE_SELECT = """    play.addEventListener('click', event => {
      event.stopPropagation();
      if (state.busy) return;
      player.reset(); state.current = take; state.take = take;
      $('audio').src = take.url; $('audio').hidden = false; show(take); slate();
      player.ensure(); $('audio').play().catch(() => {});
      if (take.id && !take.partial) trigger('click', {action: 'select', take_id: take.id});
    });
    const select = () => {
      if (state.busy || take === state.current) return;
      player.reset(); state.current = take; state.take = take;
      $('audio').src = take.url; $('audio').hidden = false; show(take); slate(); syncControls();
      if (take.id && !take.partial) trigger('click', {action: 'select', take_id: take.id});
    };
    item.tabIndex = 0; item.setAttribute('role', 'button');
    item.title = 'Select this take (Retake, edits and Keep this voice act on it)';
    item.addEventListener('click', select);
    item.addEventListener('keydown', event => {
      if (event.target === item && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); select(); }
    });
"""


# Each take's seed (every event's "seed", seeds.py, so a stopped take has it too): in
# its metrics, and in its row
# as a button that puts it in the Sampling box's Seed field (opening the box first:
# a folded Accordion's contents are not rendered)
STAGE_SEED = [
    ("'trimmed_s', 'finish_reason']) {", "'trimmed_s', 'finish_reason', 'seed']) {"),
    ("    if (event.ttfa_ms != null) take.metrics.ttfa_ms = event.ttfa_ms;\n",
     "    if (event.ttfa_ms != null) take.metrics.ttfa_ms = event.ttfa_ms;\n"
     "    if (event.seed != null) take.metrics.seed = event.seed;\n"),
    ("  if (m.audio_s != null) add('length', `${m.audio_s.toFixed(1)} s`);\n",
     "  if (m.audio_s != null) add('length', `${m.audio_s.toFixed(1)} s`);\n"
     "  if (m.seed != null) add('seed', String(m.seed));\n"),
    ("function slate() {\n", """const SEED_HINTS = {
  random: 'Empty: a new random seed every take. Each take shows its own; click one to keep it.',
  fixed: 'Every take uses this seed: same script, voice and settings, same take. Clear it for random.'};
const SEED_BADGES = {random: 'Random', fixed: 'Fixed'};
function seedField() { return document.querySelector('#seed-box input'); }
function setSeed(value) {
  const input = seedField(); if (!input) return;
  input.value = value == null ? '' : String(value);
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  seedWidget();
}
function lastSeed() {
  for (let i = state.takes.length - 1; i >= 0; i--) if (state.takes[i].metrics.seed != null) return state.takes[i].metrics.seed;
  return null;
}
function newSeed() { return crypto.getRandomValues(new Uint32Array(1))[0]; }
function seedWidget() {
  const ui = document.querySelector('#seed-widget .seed-ui'), input = seedField();
  if (!ui || !input) return;
  const value = ui.querySelector('.seed-value');
  if (!ui.dataset.bound) {
    ui.dataset.bound = '1';
    ui.querySelector('[data-act="last"]').addEventListener('click', () => { const s = lastSeed(); if (s != null) setSeed(s); });
    ui.querySelector('[data-act="new"]').addEventListener('click', () => setSeed(newSeed()));
    ui.querySelector('[data-act="clear"]').addEventListener('click', () => setSeed(null));
    value.addEventListener('input', () => { value.value = value.value.replace(/\\D/g, '').slice(0, 10); });
    const commit = () => {  // empty: random; a number over the range is not taken
      const n = Number(value.value);
      if (value.value === '') setSeed(null); else if (n <= 4294967295) setSeed(n); else seedWidget();
    };
    value.addEventListener('change', commit);
    value.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); value.blur(); } });
  }
  const seed = input.value.trim(), mode = seed === '' ? 'random' : 'fixed';
  ui.dataset.mode = mode;
  ui.querySelector('.seed-badge').textContent = SEED_BADGES[mode];
  ui.querySelector('.seed-hint').textContent = SEED_HINTS[mode];
  if (document.activeElement !== value) value.value = seed;
  const last = lastSeed();
  ui.querySelector('[data-act="last"]').disabled = last == null || String(last) === seed;
  ui.querySelector('[data-act="clear"]').disabled = seed === '';
}
setInterval(seedWidget, 400);  // the Sampling box renders its contents only when opened
function useSeed(value) {
  if (!seedField()) {
    [...document.querySelectorAll('button.label-wrap')].find(b => b.textContent.trim().startsWith('Sampling'))?.click();
  }
  let tries = 0;
  const fill = () => {
    if (!seedField()) { if (++tries < 40) setTimeout(fill, 50); return; }
    setSeed(value);
    document.querySelector('#seed-widget')?.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  };
  fill();
}
function slate() {
"""),
    ("    body.append(title, sub);\n", """    if (take.metrics.seed != null) {
      const seed = document.createElement('button'); seed.type = 'button'; seed.className = 'stage-seed';
      seed.textContent = `seed ${take.metrics.seed}`; seed.title = 'Use this seed for the next take (Sampling → Seed)';
      seed.addEventListener('click', event => { event.stopPropagation(); useSeed(take.metrics.seed); });
      sub.append(' · ', seed);
    }
    body.append(title, sub);
"""),
]


def stage_script(text: str) -> str:
    """stage.js with selectable take rows and take seeds (each part left out, with a
    note, if the Space's code differs)."""
    if STAGE_PLAY in text:
        text = text.replace(STAGE_PLAY, STAGE_SELECT)
    else:
        print("[studio] take rows: stage.js changed, rows select through ▶ Play only", flush=True)
    if all(text.count(old) == 1 for old, _ in STAGE_SEED):
        for old, new in STAGE_SEED:
            text = text.replace(old, new)
    else:
        print("[studio] take seeds: stage.js changed, takes do not show their seed", flush=True)
    return text


# The Experimental card's sampler switch: short names, and a line about the one picked
SAMPLER_LABELS = {"fixes": "Community fixes", "reference": "Fish reference", "reference_ras": "Fish reference + RAS"}
SAMPLER_HINTS = {
    "fixes": "The tested default: the changes under *What changes*. 70 drones instead of 136 on pause-heavy lines.",
    "reference": "Fish Audio's own rule (SGLang Omni #3278): epsilon sampling on the main token, and a repeated draw "
                 "becomes the most probable one. Here it made more drones (7 to 1 in 10 takes) and lost blind, 3 to 5.",
    "reference_ras": "Fish reference, with a repeated draw redrawn at temperature 1.0, top-p 0.95, top-k 30 instead "
                     "of the most probable one. Not tested here.",
}
EXPERIMENTAL_DETAILS = """**From the next take, with the box on:**

- The main token is drawn with no top-p cut: top-p 0.9 cut a drone's way out, the end of speech \
  ([torenyx/drama3-24gb](https://github.com/torenyx/drama3-24gb)).
- A remake after a stall also loosens its repeat-avoiding redraw (same source).
- The 9 acoustic codebooks draw from at most 30 candidates, and the redraw keeps the token that crosses top-p \
  (SGLang Omni [#2810](https://github.com/sgl-project/sglang-omni/pull/2810), \
  [#2890](https://github.com/sgl-project/sglang-omni/pull/2890)).
- With Quality Mode on, a take whose phrase jumps far above the voice's pitch is made once more \
  ([drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold)'s register guard).
- The sampler above applies them with the community fixes or with Fish Audio's reference sampling.

**Always, box on or off:** no acoustic code the codec lacks is drawn (it has 1,024 per codebook, the model scores 4,096).

Tested on pause-heavy lines (two cloned voices, 80 takes a side): 70 drones caught instead of 136, 61 takes right first
time instead of 51."""


def sampler_classes(on: bool) -> list[str]:
    return ["exp-sampler"] + ([] if on else ["exp-sampler-off"])


SAMPLING_DEFAULTS = {"Temperature": 0.7, "Top-p": 0.7, "Top-k": 30} if SMALL else {}
NOTE_PLACEHOLDER_SPACE = "e.g. more amused than before; drop the hurt"
NOTE_PLACEHOLDER = "e.g. more amused than before · or leave empty and edit the script to re-read it"
# The seed control: a field that starts empty (a new random seed every take); a number in
# it is kept for every take (typed, the last take's, or a new one). stage.js (STAGE_SEED)
# wires it to the hidden #seed-box field
SEED_WIDGET = """<div class="seed-ui" data-mode="random">
  <div class="seed-head">
    <div class="seed-text"><span class="seed-title">Seed</span><span class="seed-hint"></span></div>
    <span class="seed-badge"></span>
  </div>
  <div class="seed-row">
    <input class="seed-value" placeholder="Random" inputmode="numeric" autocomplete="off" spellcheck="false"
           aria-label="Seed (empty: a random seed every take)">
    <button type="button" class="seed-act" data-act="last" title="The seed of your latest take">↺ Last take</button>
    <button type="button" class="seed-act" data-act="new" title="A fresh random seed, kept for every take">🎲 New</button>
    <button type="button" class="seed-act" data-act="clear" title="Empty: a new random seed every take">✕ Clear</button>
  </div>
</div>"""


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
    read_text = Path.read_text

    @functools.wraps(read_text)
    def on_read_text(self, *args, **kwargs):
        text = read_text(self, *args, **kwargs)
        return stage_script(text) if self.name == "stage.js" else text

    def guidance_classes(checked):
        return ["voice-guidance"] + ([] if checked else ["voice-guidance-off"])

    def experimental_controls(sampling):
        """"Enable Experimental Changes", its sampler (a switch like the Voice card's), a
        line about the sampler picked, and the details folded away."""
        on = bool(sampling and sampling.ENABLED)
        gr.HTML('<div class="card-head"><h2>Experimental</h2></div>')
        found["experimental_box"] = gr.Checkbox(
            value=on, label="Enable Experimental Changes", container=False,
            info="Sampler changes from others' findings, off by default: half as many drones on pause-heavy lines "
                 "in testing. A seed makes a different take with them on.")
        with gr.Column(elem_classes=sampler_classes(on), elem_id="experimental-sampler-group") as group:
            found["sampler_group"] = group
            gr.HTML('<span class="exp-label">Sampler</span>', padding=False, container=False)
            found["sampler_box"] = gr.Radio(
                [(SAMPLER_LABELS[name], name) for name in sampling.SAMPLERS] if sampling else [],
                value=sampling.SAMPLER if sampling else None, show_label=False, container=False,
                interactive=on, elem_id="experimental-sampler")
            found["sampler_hint"] = gr.Markdown(SAMPLER_HINTS.get(sampling.SAMPLER if sampling else "", ""),
                                                elem_classes="hint")
        with gr.Accordion("What changes", open=False, elem_id="experimental-details"):
            gr.Markdown(EXPERIMENTAL_DETAILS, elem_classes="hint")

    def word_check_controls():
        """Length Hint, "Check every word" and its voice guidance (folded away, like Sampling)."""
        # (a folded Accordion's contents are not rendered, so the style is outside it)
        gr.HTML(f"<style>{EXTRAS_CSS}</style>", padding=False, container=False, elem_id="studio-extras-style")
        found["hint_box"] = gr.Checkbox(
            value=long_takes.SETTINGS["length_hint"], label="Length Hint (Might improve short generations)",
            info="Each line tells the model about how long it should take, from its words (Drama 3's "
                 "[audio tokens: N]). Steadier pacing, preferred blind on long takes. Off: the model picks "
                 "the length itself.")
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
            # Cards of their own between the Voice card and Sampling: the experimental
            # sampling (not with the MLX backend), then Length Hint and Quality Mode
            sampling = sys.modules.get("upstream_sampling")
            with gr.Column(elem_classes="card", elem_id="experimental", visible=sampling is not None):
                experimental_controls(sampling)
            with gr.Column(elem_classes="card", elem_id="word-check"):
                word_check_controls()
        accordion_init(self, *args, **kwargs)

    def seed_controls():
        """The seed, in Sampling after Top-k (seeds.py): the SEED_WIDGET control (driven by
        stage.js, STAGE_SEED) over a hidden number field that holds the setting (empty:
        a random seed per take)."""
        from gradio.context import Context

        gr.HTML(SEED_WIDGET, elem_id="seed-widget", padding=False)
        seed = gr.Number(value=long_takes.SETTINGS["seed"], label="Seed", precision=0, minimum=0,
                         maximum=seeds.SEED_LIMIT - 1, elem_id="seed-box")
        found["seed"] = seed

        def set_seed(value):
            long_takes.SETTINGS["seed"] = seeds.checked(value)

        seed.change(set_seed, inputs=seed, queue=False, api_visibility="private")
        def fresh():  # a page always opens with the field empty: a random seed per take
            long_takes.SETTINGS["seed"] = None
            return gr.Number(value=None)

        Context.root_block.load(fresh, outputs=seed, queue=False, api_visibility="private")

    @functools.wraps(slider_init)
    def on_slider(self, *args, **kwargs):
        if kwargs.get("label") in SAMPLING_DEFAULTS:
            if len(args) >= 3:
                args = (*args[:2], SAMPLING_DEFAULTS[kwargs["label"]], *args[3:])
            else:
                kwargs["value"] = SAMPLING_DEFAULTS[kwargs["label"]]
        slider_init(self, *args, **kwargs)
        if kwargs.get("label") == "Top-k" and "seed" not in found:
            seed_controls()

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
                tidy_box = gr.Checkbox(value=long_takes.SETTINGS["tail_tidy"], label="Trim chunk tails",
                                       info="With crossfading off: each chunk is held 80 ms past its last word, "
                                            "faded out and cut, and the join is silent, dropping the hiss the model "
                                            "trails. Off keeps every chunk as made, breaths and soft word endings "
                                            "included.")
                crossfade_box = gr.Checkbox(
                    value=long_takes.SETTINGS["crossfade"], label="Crossfade chunk joins",
                    info="Each chunk's trailing breath and hiss fade out under the start of the next one instead "
                         "of a trimmed end and a silent pause; no word is faded. Preferred blind at 5 of 6 joins. "
                         "Off: trimmed ends (\u201cTrim chunk tails\u201d) and silent pauses.")
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
            tidy_box.change(setting("tail_tidy"), inputs=tidy_box, queue=False, api_visibility="private")
            crossfade_box.change(setting("crossfade"), inputs=crossfade_box, queue=False, api_visibility="private")
            word_box.change(setting("word_check"), inputs=word_box, queue=False, api_visibility="private")
            experimental_box, hint_box, sampler_box = found["experimental_box"], found["hint_box"], found["sampler_box"]
            hint_box.change(setting("length_hint"), inputs=hint_box, queue=False, api_visibility="private")

            sampler_group, sampler_hint = found["sampler_group"], found["sampler_hint"]

            def experimental_state():
                """The switch greyed out while the box is off: the sampler is used with it on only."""
                sampling = sys.modules.get("upstream_sampling")
                on, name = bool(getattr(sampling, "ENABLED", False)), getattr(sampling, "SAMPLER", None)
                return (gr.Checkbox(value=on), gr.Radio(value=name, interactive=on),
                        gr.Column(elem_classes=sampler_classes(on)), SAMPLER_HINTS.get(name, ""))

            def set_experimental(value):
                if (sampling := sys.modules.get("upstream_sampling")) is not None:
                    sampling.set_enabled(bool(value))
                return experimental_state()[1:]

            experimental_outputs = [sampler_box, sampler_group, sampler_hint]
            experimental_box.change(set_experimental, inputs=experimental_box, outputs=experimental_outputs,
                                    queue=False, api_visibility="private")

            def set_sampler(value):
                if (sampling := sys.modules.get("upstream_sampling")) is not None and value:
                    sampling.set_sampler(value)
                return SAMPLER_HINTS.get(value, "")

            sampler_box.change(set_sampler, inputs=sampler_box, outputs=sampler_hint, queue=False,
                               api_visibility="private")

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
                        gr.Checkbox(value=long_takes.SETTINGS["tail_tidy"]),
                        gr.Checkbox(value=long_takes.SETTINGS["crossfade"]),
                        gr.Checkbox(value=long_takes.SETTINGS["word_check"]),
                        gr.Slider(value=long_takes.SETTINGS["voice_guidance"]),
                        gr.Checkbox(value=long_takes.SETTINGS["length_hint"]))

            Context.root_block.load(current_settings,
                                    outputs=[found["unlock"], found["ref_hint"], enabled, lookahead, infill_box, tidy_box,
                                             crossfade_box, word_box, voice_slider, hint_box],
                                    queue=False, api_visibility="private")
            Context.root_block.load(experimental_state, outputs=[experimental_box, *experimental_outputs],
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
    Path.read_text = on_read_text
    gr.Slider.__init__ = on_slider
    if SAMPLING_DEFAULTS:
        gr.Button.click = on_click
    try:
        yield
    finally:
        gr.Markdown.__init__, gr.Textbox.__init__, gr.State.__init__, gr.Audio.__init__ = (
            markdown_init, textbox_init, state_init, audio_init)
        gr.Textbox.input = textbox_input
        gr.Accordion.__init__ = accordion_init
        Path.read_text = read_text
        gr.Slider.__init__, gr.Button.click = slider_init, button_click
