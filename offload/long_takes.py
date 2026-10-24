"""Long scripts, edits without a note, and 20 / 60 s reference clips.

- Long scripts: the engine splits text longer than one chunk (chunking.py), generates
  the chunks one after another and streams them as one take. Each chunk sees the first
  chunk and the latest one as history (user text + its audio codes), the layout Fish
  Audio uses in fish-speech, so the voice stays the same across chunks.
- Edits: a Retake with an empty note re-reads the edited script with the earlier
  takes as context (Drama 3's "editing as continuation"), instead of being refused.
- Retake / Edit of a multi-chunk take (EXPERIMENTAL, off by default): an edit
  regenerates only the chunks whose text changed and keeps the others' audio; a note
  redoes every chunk, each guided by its old version.
- A Direction the user writes ("Direction: …\nLine: …") holds for the whole take: every
  chunk is sent under it with the Line's own [cues] kept (a Retake note, instead, replaces
  the cues; see parse_turn).
- Reference clips: 20 s by default, as the Space; up to 60 s (the engine's limit)
  when unlocked.
- Joins: each held chunk's trailing breath and hiss fade out under the start of the next
  (speech-safe crossfade, crossfade_shape), or, with that off, end trimmed before a silent
  pause (tidy_tail).
- A take or chunk that ends before its last words, or stalls, gets them made as a
  continuation of it (make_rest): when streamed, once it is made; in Quality Mode, before
  it is made again from scratch.
- Line breaks inside a line are sent as spaces (one_line), and an optional length hint
  (length_hint) asks for about as long as the words need.

Short single-chunk lines take the Space's own code path unchanged.
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
import re
import time
from dataclasses import dataclass

import torch

import chunking
import infill
import register_guard
import seeds
import subtitles
import word_check

REF_DEFAULT_SECONDS = 20.0
REF_UNLOCKED_SECONDS = 60.0  # drama3_engine.MAX_REFERENCE_SECONDS
TRANSCRIPT_CHARS = 1500  # a 60 s clip says ~900 characters; the Space allows 600
CACHE_TAKES = 32
PACKET_BYTES = 44100  # 0.5 s of 16-bit mono, for audio replayed from the cache


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


# The Studio's checkboxes (one local user) and the env defaults for API-only use.
SETTINGS = {
    "ref_unlocked": _flag("DRAMA3_REF_UNLOCKED"),
    "long_retake": _flag("DRAMA3_LONG_RETAKE"),
    "lookahead": os.environ.get("DRAMA3_LOOKAHEAD", "1").strip().lower() in {"1", "true", "yes", "on"},
    "infill": os.environ.get("DRAMA3_INFILL", "1").strip().lower() in {"1", "true", "yes", "on"},
    # hold each take until its words are checked (word_check.py); off: takes stream
    "word_check": _flag("DRAMA3_WORD_CHECK"),
    # those held takes' voice guidance (tag_guidance.py; 0 = off): the Studio's slider
    "voice_guidance": float(os.environ.get("DRAMA3_VOICE_GUIDANCE", "2")),
    # the Studio's Seed box (None: each take gets a random seed; see seeds.py)
    "seed": None,
    # the Studio's "Length hint" box: "[audio tokens: N]" on every turn (see length_hint)
    "length_hint": _flag("DRAMA3_LENGTH_HINT"),
    # with crossfading off, held chunks end shortly after their last word, then silence (tidy_tail)
    "tail_tidy": os.environ.get("DRAMA3_TAIL_TIDY", "1").strip().lower() in {"1", "true", "yes", "on"},
    # the Studio's "Crossfade chunk joins" box: joins as speech-safe crossfades (see crossfade_shape)
    "crossfade": os.environ.get("DRAMA3_CROSSFADE", "1").strip().lower() in {"1", "true", "yes", "on"},
}
# Drama 3's streaming-input format (technical report, Figure 10): each chunk's turn also
# shows the chunk that follows, after <|endofturn|>, so the model voices the current
# chunk knowing what comes next (an empty next chunk marks the last one).
END_OF_TURN = "<|endofturn|>"
STREAM_INSTRUCTION = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"
KEEPALIVE_S = 2.0  # empty packets while a chunk is held back (the server's idle limit is 15 s)
# A take's first windows are decoded after its reference clip's codes, not cold: cold, the
# codec's long memory (a window differs from a continuous decode for 2-3 s) started a cloned
# take up to 4.5 dB loud (Joker clone: opening +3.3 dB over its steady level, +-1.2 warm;
# a voice chat on Drama 3 measured +5.0 -> +0.85 dB, gabacool/bidirectional-voice-claude #26).
# Off by default: blind, the warm start won all 3 Joker openings and lost all 3 Ramsay ones,
# and some first words came out distorted (the reference's own sound carries into them).
WARM_START = os.environ.get("DRAMA3_WARM_START", "0").strip().lower() in {"1", "true", "yes", "on"}


def log(message: str) -> None:
    print(f"[long] {message}", flush=True)


def spoken(text: str) -> str:
    """Words only: no [cues] or <tags>, white space collapsed (for comparisons)."""
    return " ".join(re.sub(r"\[[^\[\]]*\]|<[^<>]*>", " ", text).split())


# Also as pasted by hand: blank lines between the two, the Direction wrapped over several
# lines, the Line starting on the line after "Line:". Unmatched, the Direction was read as
# the first chunk's words, and with the lookahead the model voiced the next chunk there
# (the Line's opening, then said again by the chunk that held it).
RETAKE = re.compile(r"\s*Direction:[ \t]*(?P<note>.+?)[ \t]*\n\s*Line:\s*(?P<line>.*)", re.S)


def parse_turn(text: str) -> tuple[str | None, str | None, str]:
    """(note, direction, words) of a turn.

    note: a Retake the Space's retake_text wrote (every sentence opens with [note]);
    its words lose the acting cues, which the note replaces. direction: a
    "Direction: …\nLine: …" the user wrote; its Line keeps its own cues, and every
    chunk of it is sent under the same Direction."""
    match = RETAKE.fullmatch(text)
    if not match:
        return None, None, text
    note, line = " ".join(match.group("note").split()), match.group("line")
    if line.startswith(f"[{note}] "):
        return note, None, line.replace(f"[{note}] ", "")
    return None, note, line


def parse_retake(text: str) -> tuple[str | None, str]:
    """(note, words) of a turn (see parse_turn); a user's Direction is not a note."""
    note, _, words = parse_turn(text)
    return note, words


# Line breaks inside a line are sent as spaces (the chunker still cuts at them): with blank
# lines in its text Drama 3 stopped emitting end of speech and drew ~47 s of silence
# (gabacool/bidirectional-voice-claude #24). DRAMA3_ONE_LINE=0 sends them as written.
ONE_LINE = os.environ.get("DRAMA3_ONE_LINE", "1").strip().lower() in {"1", "true", "yes", "on"}
# The length hint (Drama 3's "[audio tokens: N]", technical report Table 10) on every
# turn, at LENGTH_HINT_RATIO of the pause fix's estimate (finished takes ran a median 0.67
# of it): said to keep takes from stalling after a sentence (appautomaton/mlx-speech #49).
# With lines split by blank lines (joined, ONE_LINE) it caused no more stalls, but held
# each take to 0.76-0.83 of the estimate (0.66-0.98 without), and blind it was preferred
# in all 4 long takes (the 2 short ones a toss-up). Off by default: the Studio's "Length
# Hint" box (SETTINGS["length_hint"]; DRAMA3_LENGTH_HINT=1 starts with it on).
LENGTH_HINT_RATIO = float(os.environ.get("DRAMA3_LENGTH_HINT_RATIO", 0.8))


def one_line(text: str) -> str:
    return re.sub(r"[ \t]*\n\s*", " ", text).strip() if ONE_LINE else text


def length_hint(text: str) -> str:
    """" [audio tokens: N]" for a turn's words, or "" when the hint is off."""
    if not SETTINGS["length_hint"] or "[audio tokens" in text.lower():
        return ""
    import pause_fix

    return f" [audio tokens: {max(1, round(LENGTH_HINT_RATIO * pause_fix.expected_frames(text)))}]"


def directed_turn(direction: str | None, text: str) -> str:
    """One chunk of a user-directed Line, under the take's Direction."""
    return f"Direction: {direction}\nLine: {text}" if direction else text


def prefixed_turn(direction: str, text: str) -> str:
    """A user's Direction as a [style prefix] (a first take; see needs_prefix)."""
    return f"[{direction}] {text}"


def with_turn(text: str, turn: str) -> str:
    """An engine input (the turn, or an envelope holding it) with its turn replaced."""
    import json

    import drama3_engine as de

    if not text.startswith(de.ENVELOPE_MARKER):
        return turn
    spec = json.loads(text[len(de.ENVELOPE_MARKER):])
    return de.ENVELOPE_MARKER + json.dumps(dict(spec, text=turn), ensure_ascii=False)


def retake_turn(note: str, words: str) -> str:
    """The Space's retake_text layout for one chunk of words."""
    sentences = re.split(r"(?<=[.!?。！？])\s+", words)
    return f"Direction: {note}\nLine: " + " ".join(f"[{note}] {sentence}" for sentence in sentences)


# ─── Generated chunks, by the turn text the server sends back as history ──────


@dataclass
class ChunkAudio:
    text: str  # the chunk of the script (what edits are aligned against)
    turn: str  # the text the model was given for it (with its lookahead, if any)
    codes: torch.Tensor  # [num_codebooks, frames] on the CPU
    pcm: bytes  # 16-bit mono at 44.1 kHz, as streamed
    words: list | None = None  # Whisper word timestamps of pcm (for infill edits), once needed


@dataclass
class LongTake:
    line: str
    chunks: list[ChunkAudio]
    direction: str | None = None  # the user's Direction (all chunks were made under it)


CACHE: collections.OrderedDict[str, LongTake] = collections.OrderedDict()


def remember(turn_text: str, take: LongTake) -> None:
    CACHE[turn_text] = take
    CACHE.move_to_end(turn_text)
    while len(CACHE) > CACHE_TAKES:
        CACHE.popitem(last=False)


def long_take(turn_text: str | None) -> LongTake | None:
    take = CACHE.get(turn_text) if turn_text else None
    return take if take is not None and len(take.chunks) > 1 else None


# ─── Engine ───────────────────────────────────────────────────────────────────


SILENCE_RMS = 0.003  # as the Space's SilenceGuard (50 ms frames)


def edge_silence(pcm: bytes, rate: int = 44100) -> tuple[float, float]:
    """Seconds of silence at the start and the end of 16-bit mono PCM."""
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    frame = int(0.05 * rate)
    count = len(samples) // frame
    if not count:
        return 0.0, 0.0
    loud = np.sqrt((samples[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) >= SILENCE_RMS
    if not loud.any():
        return count * 0.05, count * 0.05
    return float(np.argmax(loud)) * 0.05, float(np.argmax(loud[::-1])) * 0.05


# The pause between two chunks. Each chunk is its own generation, and the model ends it
# right on the last syllable and starts the next one with no lead-in, so joined as is the
# next line follows with ~0-0.1 s of silence. The pause is sized like the model's own
# pauses inside a chunk (median 0.25-0.45 s at a sentence end) by how the chunk ends,
# counting the silence it already has; below the server's 0.75 s pause cap.
JOIN_PAUSE_S = {"sentence": 0.45, "clause": 0.25, "word": 0.12}
SENTENCE_END = ".!?…。！？"
CLAUSE_END = ",;:—–-、，；："


def pause_after(text: str) -> float:
    """Seconds of pause after a chunk that ends on text (JOIN_PAUSE_S by its last mark)."""
    end = re.sub(r"(\s*</[^<>]*>)+$", "", text.rstrip())  # "reef.</menacing>" ends a sentence
    end = end.rstrip().rstrip("\"'”’)]»」』").rstrip()
    return JOIN_PAUSE_S["sentence" if end[-1:] in SENTENCE_END else "clause" if end[-1:] in CLAUSE_END else "word"]


def join_pause(text: str, pcm: bytes, rate: int = 44100, tone: list[bytes] | None = None) -> bytes:
    """The pause to put after a chunk that ends on text, whose audio is pcm: room tone
    of the chunks in `tone` (see room_tone), or silence."""
    missing = pause_after(text) - edge_silence(pcm, rate)[1]
    return room_tone(tone or [pcm], int(max(0.0, missing) * rate))


# Room tone: a cloned voice keeps its recording's background (hiss, hum, room), and a
# pause of digital silence between two chunks makes that background drop out and come
# back. The pause is filled instead with noise of the background's spectrum and level,
# measured on the quietest stretches of the take's audio. Takes whose background is
# below ROOM_TONE_MIN_RMS keep silent pauses.
# Off by default (DRAMA3_ROOM_TONE=1 turns it on): in a listening test of one join, the
# pause filled with this noise, with the take's own background copied from its natural
# pauses, or with the generated gap 10 dB down all still carried the hiss the joins were
# made to lose; only a fade to digital silence did not.
ROOM_TONE = os.environ.get("DRAMA3_ROOM_TONE", "0").strip().lower() in {"1", "true", "yes", "on"}
ROOM_TONE_FRAME = 4096  # samples per analysis frame (93 ms: 11 Hz bins, so a mains hum stays near its pitch)
ROOM_TONE_CEILING = 0.02  # RMS above which a frame is never background (-34 dBFS)
ROOM_TONE_MIN_RMS = 0.0005  # quieter backgrounds are left silent (-66 dBFS)
ROOM_TONE_MIN_FRAMES = 4  # background needed to measure it (0.37 s)
# The quiet frames' level still counts breath tails and word edges, and a take ended
# below it (the background a chunk's last word fades into: -46 dBFS where the frames
# gave -41): a pause at that level was a step up into flat hiss, heard as a burst of
# static between two chunks, worst with clean voices. A pause is never louder than the
# audio it follows ends, and fades in and out.
ROOM_TONE_EDGE_S = 0.2  # the end of the audio before a pause, whose level caps it
ROOM_TONE_FADE_S = 0.01


def background(pcms: list[bytes]):
    """(magnitude spectrum, RMS) of the quietest frames of 16-bit PCM, or None when there
    is too little background or it is below ROOM_TONE_MIN_RMS."""
    import numpy as np

    frames = []
    for pcm in pcms:
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
        count = len(samples) // ROOM_TONE_FRAME
        if count:
            frames.append(samples[: count * ROOM_TONE_FRAME].reshape(count, ROOM_TONE_FRAME))
    if not frames:
        return None
    frames = np.concatenate(frames)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    # the quietest frames, up to twice the 10th percentile: pauses and breaths' tails, not speech
    quiet = frames[(rms <= min(ROOM_TONE_CEILING, 2 * np.percentile(rms, 10))) & (rms > 0)]
    if len(quiet) < ROOM_TONE_MIN_FRAMES:
        return None
    level = float(np.sqrt((quiet ** 2).mean()))
    if level < ROOM_TONE_MIN_RMS:
        return None
    window = np.hanning(ROOM_TONE_FRAME).astype(np.float32)
    spectrum = np.sqrt((np.abs(np.fft.rfft(quiet * window, axis=1)) ** 2).mean(axis=0))
    return spectrum, level


def room_tone(pcms: list[bytes], samples: int) -> bytes:
    """`samples` of 16-bit noise shaped like the background of pcms, or silence. Seeded by
    the audio, so a take replays with the same pauses."""
    import zlib

    import numpy as np

    if samples <= 0:
        return b""
    if not ROOM_TONE:
        return bytes(2 * samples)
    measured = background(pcms)
    if measured is None:
        return bytes(2 * samples)
    spectrum, level = measured
    edge = np.frombuffer(pcms[-1][-int(ROOM_TONE_EDGE_S * 44100) * 2:], dtype="<i2").astype(np.float32) / 32768
    if edge.size:
        level = min(level, float(np.sqrt((edge ** 2).mean())))
    if level < ROOM_TONE_MIN_RMS:
        return bytes(2 * samples)
    rng = np.random.default_rng(zlib.crc32(pcms[-1][-8192:]) + samples)
    white = np.fft.rfft(rng.standard_normal(samples))
    bins = np.linspace(0, len(spectrum) - 1, len(white))
    shaped = np.fft.irfft(white * np.interp(bins, np.arange(len(spectrum)), spectrum), n=samples)
    shaped *= level / max(float(np.sqrt((shaped ** 2).mean())), 1e-9)
    fade = min(int(ROOM_TONE_FADE_S * 44100), samples // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        shaped[:fade] *= ramp
        shaped[samples - fade:] *= ramp[::-1]
    return (np.clip(shaped, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# A held take that runs on after its last word (voice guidance copies a clip's music bed,
# and twice in 48 takes the bed went on for 17 s after the line) is cut TAIL_KEEP_S after
# the last word Whisper places, when more than TAIL_MAX_S follow it. Not when the line's
# last word was not heard (it may be whispered) or the script ends on a [cue] (a laugh or
# a sigh is not a word Whisper places). Over 358 test takes the runaways left 5-17 s;
# with clean clips no take had more than 2.7 s, often a voiced breath or hum, kept.
TAIL_KEEP_S = 0.4
TAIL_MAX_S = 4.0  # as the server's stall limit
TAIL_FADE_S = 0.05
ENDS_ON_CUE = re.compile(r"\[[^\[\]]*\]\s*$")


def trim_tail(script: str, heard: list, pcm: bytes, codes):
    """(pcm, codes) of a held take without what runs on after its last word."""
    import numpy as np

    if not heard or ENDS_ON_CUE.search(script):
        return pcm, codes
    words = infill.align(script, heard)
    if not words or not words[-1].heard or words[-1].end is None:
        return pcm, codes
    keep = words[-1].end + TAIL_KEEP_S
    if len(pcm) / 88200 - keep <= TAIL_MAX_S:
        return pcm, codes
    samples = np.frombuffer(pcm[: int(keep * 44100) * 2], dtype="<i2").astype(np.float32)
    fade = min(int(TAIL_FADE_S * 44100), len(samples))
    samples[len(samples) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    log(f"cut {len(pcm) / 88200 - keep:.1f}s after the last word ({words[-1].text!r} ends at {words[-1].end:.1f}s)")
    if codes is not None:
        codes = codes[:, : min(codes.shape[1], int(np.ceil(keep * infill.FPS)))]
    return samples.astype("<i2").tobytes(), codes


# A chunk's tail (the end nudge's fallback; pause_fix.py). After its last word the model
# lets the audio decay for ~0.3 s, then goes on with breath and room noise for ~0.6 s
# more (-45..-50 dB, a third to half of its frames silence codes) before it ends the
# chunk; at a join that played as a burst of hiss ahead of the join's own pause. A held
# chunk's audio fades out over TAIL_FADE_OUT_S from its last speech frame (a frame
# within TAIL_SPEECH_DB of the chunk's loud level: word-final consonants are kept, and
# never before the end of the last word Whisper heard) and ends there; what the model
# made before its first speech frame fades in over TAIL_FADE_IN_S. With the join's
# silent pause (ROOM_TONE), a join is the word's own decay, then digital silence: in a
# listening test of one join, keeping 0.28 s of decay, the codec's own floor, room tone,
# real background or the gap 10 dB down all still carried the hiss; only this did not.
# It can clip a breath or a soft word ending: SETTINGS["tail_tidy"] (the Studio's "Trim
# chunk tails" box, DRAMA3_TAIL_TIDY) turns it off.
TAIL_SPEECH_DB = 24.0
TAIL_FADE_OUT_S = float(os.environ.get("DRAMA3_TAIL_FADE_S", 0.25))
# full level kept this long past the speech end before the fade: a word's last consonant
# sits at the level of the hiss that follows (both 24-35 dB under the loud level), so no
# threshold tells them apart, and a fade from the speech end can soften it. In a blind
# test of six joins, holding 80 ms then fading over 0.25 s was preferred in five (a
# fade from the speech end over 0.2 s in one), neither ever picked as clipped or hissing.
TAIL_HOLD_S = float(os.environ.get("DRAMA3_TAIL_HOLD_S", 0.08))
TAIL_FADE_IN_S = 0.06


def speech_span(pcm: bytes, frames: int, heard: list | None = None, script: str = "") -> tuple[int, float] | None:
    """(first speech frame, end of speech in seconds) of a chunk's audio, or None."""
    import numpy as np

    samples = np.frombuffer(pcm[: frames * 2048 * 2], dtype="<i2").astype(np.float32) / 32768
    count = min(frames, len(samples) // 2048)
    if count < 8:
        return None
    level = 20 * np.log10(np.sqrt((samples[: count * 2048].reshape(count, 2048) ** 2).mean(1)) + 1e-9)
    speech = np.where(level >= np.percentile(level, 90) - TAIL_SPEECH_DB)[0]
    if not len(speech):
        return None
    end = (int(speech[-1]) + 1) / infill.FPS
    if heard and script:
        words = infill.align(script, heard)
        if words and words[-1].heard and words[-1].end is not None:
            end = max(end, words[-1].end)
    return int(speech[0]), end


def shape_tail(samples, first: int, end: float):
    """A chunk's samples (float, 44.1 kHz) held TAIL_HOLD_S past the speech end at `end`
    seconds, faded out over TAIL_FADE_OUT_S and cut, with what comes before its first
    speech frame faded in; or None when the chunk ends on its last word."""
    import numpy as np

    begin = min(samples.size, int((end + TAIL_HOLD_S) * 44100))
    stop = min(samples.size, int((end + TAIL_HOLD_S + TAIL_FADE_OUT_S) * 44100))
    if stop - int(end * 44100) < int((TAIL_HOLD_S + TAIL_FADE_OUT_S) * 44100) // 2:
        return None
    samples = samples[:stop].copy()
    samples[begin:] *= np.linspace(1.0, 0.0, stop - begin, dtype=np.float32)
    head = first * 2048
    if head > 0:
        rise = min(int(TAIL_FADE_IN_S * 44100), head)
        samples[head - rise: head] *= np.linspace(0.0, 1.0, rise, dtype=np.float32)
        samples[: head - rise] = 0.0
    return samples


# Speech-safe crossfade at a join (SETTINGS["crossfade"], on by default; off: the trim and
# silent pause above). A chunk keeps full level TAIL_HOLD_S past its last word, then only
# its tail (breath and hiss) fades out smoothly to the chunk's end, uncut; the next chunk
# starts at full level CROSSFADE_S before that end (never sooner than CROSSFADE_GAP_S after
# the hold), so the fading hiss runs under its first words, which mask it. Blind, over
# six joins of the same takes: plain crossfades beat trim + silence (250 ms: best 4 of 6;
# trim + silence: worst 4 of 6, never best), and this speech-safe one, which never fades a
# word in or out, beat the plain 250 ms crossfade (best 5 of 6, never worst; 3 at 350 ms).
CROSSFADE_S = float(os.environ.get("DRAMA3_CROSSFADE_S", 0.35))
CROSSFADE_GAP_S = 0.15


def crossfade_shape(samples, first: int, end: float, pause: float = 0.0):
    """(samples, start) of a held chunk for a crossfade join: float samples (44.1 kHz) with
    the tail faded out from TAIL_HOLD_S past the speech end at `end` seconds and what comes
    before the first speech frame faded in; `start` is where the next chunk begins, at
    least `pause` seconds after the speech end. A chunk that ends soon after its last word
    (the end nudge often stops it within 0.2 s) has too little tail to overlap: it fades to
    silence and is padded with silence up to `start`."""
    import numpy as np

    samples = samples.copy()
    size = samples.size
    fade_from = min(size, int((end + TAIL_HOLD_S) * 44100))
    if size > fade_from:
        samples[fade_from:] *= np.cos(np.linspace(0.0, np.pi / 2, size - fade_from, dtype=np.float32)) ** 2
    start = max(size - int(CROSSFADE_S * 44100), fade_from + int(CROSSFADE_GAP_S * 44100), int((end + pause) * 44100))
    if start > size:
        samples = np.concatenate((samples, np.zeros(start - size, dtype=np.float32)))
    head = first * 2048
    if head > 0:
        rise = min(int(TAIL_FADE_IN_S * 44100), head)
        samples[head - rise: head] *= np.linspace(0.0, 1.0, rise, dtype=np.float32)
        samples[: head - rise] = 0.0
    return samples, start


def keepalive(fn, *args):
    """fn(*args) in a worker thread, sending empty audio packets meanwhile: the server
    ends a take that sends nothing for 15 s (CPU Whisper on a long chunk can take that)."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        while True:
            try:
                return future.result(timeout=KEEPALIVE_S)
            except concurrent.futures.TimeoutError:
                yield {"type": "speech.audio.delta", "pcm": b""}


def with_stream_instruction(segments: list) -> list:
    """The Space's prompt segments with the streaming-lookahead instruction in the system
    turn: in place of the plain TTS one, or after the reference blocks / character
    descriptions (where the report's line-separated format puts its instruction)."""
    import drama3_engine as de

    segments = list(segments)
    if segments[1] == de.TTS_INSTRUCTION:
        segments[1] = STREAM_INSTRUCTION
    else:
        segments.insert(segments.index(de.MESSAGE_END), "\n\n" + STREAM_INSTRUCTION)
    return segments


def make_engine(base):
    """base (the Space's Engine, or the portable one) with chunked generation."""
    import drama3_engine as de

    class LongEngine(base):
        _prompt_override = None

        def prompt(self, request):
            if self._prompt_override is not None:
                return self._prompt_override
            return super().prompt(request)

        take_seed = None  # this take's seed (seeds.py), None: the generator is left unseeded
        _seed_now = None  # the seed the next prefill applies
        _stalled = False  # the last generation ended in a stall
        _relax_now = False  # the next one remakes it: its RAS redraw is uncut (upstream_sampling)

        def use_seed(self, attempt: int) -> None:
            """The next generation is the take's remake number `attempt` (0: its own seed)."""
            self._seed_now = None if self.take_seed is None else seeds.attempt_seed(self.take_seed, attempt)
            self._relax_now = attempt > 0 and self._stalled

        def frames(self, prompt, request):
            """Every backend's frame loop; a remake of one that stalled draws as
            upstream_sampling.set_relaxed says."""
            sampling = sys.modules.get("upstream_sampling")
            relaxed, self._stalled = self._relax_now and getattr(sampling, "ENABLED", False), False
            if relaxed:
                log("remaking a stalled generation: its RAS redraw is uncut")
                sampling.set_relaxed(True)
            try:
                reason = yield from super().frames(prompt, request)
            finally:
                if relaxed:
                    sampling.set_relaxed(False)
            self._stalled = reason == "stall"
            return reason

        def prefill(self, prompt, request):
            """Every backend's frame loop starts with this: the seed applies to the
            sampling that follows it (the prefill and graph capture use random numbers too)."""
            decode = super().prefill(prompt, request)
            if self._seed_now is not None:
                seeds.seed_rng(self._seed_now)
            return decode

        def speech(self, body, *, deadline=None):
            self.take_seed = body.get(seeds.SEED_KEY)
            self.use_seed(0)
            try:
                request = de.parse_request(body)
                plan = self.plan(request)
            except de.RequestError as error:
                yield {"type": "error", "message": f"Drama 3 request: {error}"}
                return
            if plan is None:  # one chunk: the Space's own path
                note, direction, words = parse_turn(request.text)
                line = words
                if note is None and not subtitles.is_timed(words):
                    line = one_line(words) + length_hint(words)
                if self.needs_prefix(request, direction):
                    body = dict(body, input=prefixed_turn(direction, line))
                elif direction is not None:  # as the model knows it, however it was pasted
                    body = dict(body, input=with_turn(body["input"], directed_turn(direction, line)))
                elif line != words:
                    body = dict(body, input=with_turn(body["input"], line))
                check = word_check.checker(words, request.max_new_tokens)
                if check is not None and SETTINGS["word_check"]:
                    yield from self.checked_speech(body, deadline, request, check, direction, words)
                    return
                base_speech = super().speech
                events = (self.opening_checked(lambda: base_speech(body, deadline=deadline), check)
                          if check is not None and word_check.OPENING else base_speech(body, deadline=deadline))
                pcm = bytearray()
                with self.warm_start(request):  # (decoding runs while the events are read)
                    for event in events:
                        if event["type"] == "speech.audio.delta":
                            pcm += event["pcm"]
                        elif event["type"] == "speech.audio.done" and event.get("frames"):
                            codes = self.last_codes(event["frames"])
                            if check is not None and word_check.ENDING:
                                rest = yield from self.finish_line(request, body["input"], direction, check,
                                                                   bytes(pcm), codes, event)
                                if rest is not None:
                                    more, more_codes, event = rest
                                    pcm += more
                                    codes = torch.cat((codes, more_codes), dim=1)
                            # kept (codes + audio) so an edit of this line can be infilled
                            remember(request.text, LongTake(line=words, direction=direction, chunks=[
                                ChunkAudio(words, request.text, codes, bytes(pcm))]))
                        yield event
                return
            try:
                yield from self.long_speech(request, *plan)
            except de.RequestError as error:
                yield {"type": "error", "message": f"Drama 3 request: {error}"}

        # A Direction on a first take ------------------------------------------------
        #
        # "Direction: …\nLine: …" is Drama 3's directing format, learned as a conversation
        # about an earlier take (the Space uses it so for retakes, which send that take as
        # context). On a first take the model often lost its place in it: "Direction: Donald
        # Trump is having a chat with Elizabeth Holmes from Theranos\nLine: <whispers>
        # Okay... Alright... Good chat. </whispers> Are you being real, right now?" read the
        # direction aloud or skipped the whispered span: right in 1 of 12 takes with a
        # random voice, 3 of 10 with a clone (tag guidance on). A first take is therefore
        # sent as the line under a [style prefix], which Drama 3 never voices: 12 of 12 and
        # 9 of 10 right (with the word check), the span whispered and the rest voiced. The
        # take keeps its own text everywhere else (history, retakes, edits).

        def voice_guided(self, events, request):
            """events of a take the word check holds, made with voice guidance
            (tag_guidance.py) at the Studio's strength. Not for a Surprise / Design retake:
            its voice reference is an earlier take, which its history holds too, so the
            prompt without the reference differs little, and guidance on that difference
            ended the line after its first sentence (6 of 8 takes; unguided 1 of 4)."""
            own = request.voice_count and any(
                spoken(request.references[0].get("text") or "") == spoken(parse_turn(turn)[2])
                for turn in request.history)
            self.voice_weight = 0.0 if own else SETTINGS["voice_guidance"]
            try:
                return (yield from events)
            finally:
                self.voice_weight = 0.0

        def needs_prefix(self, request, direction) -> bool:
            return direction is not None and not request.history and request.character is None

        def opening_checked(self, start, check):
            """A streamed take whose first word_check.OPENING_S are held (the server holds
            its first 2.5 s anyway) and checked against the start of the line: started
            again (at most word_check.RETRIES times) when they do not match, else the rest
            streams as it is made. start() makes the take's events; its return value (a
            chunk's frames, reason and audio) is passed on."""
            need = int(word_check.OPENING_S * 88200)
            for attempt in range(1 + word_check.RETRIES):
                self.use_seed(attempt)
                events, held, buffered, result, finished = start(), 0, [], None, False
                try:
                    while held < need:
                        event = next(events)
                        buffered.append(event)
                        if event["type"] == "speech.audio.delta":
                            held += len(event["pcm"])
                        elif event["type"] == "error":
                            break
                except StopIteration as stop:
                    result, finished = stop.value, True
                pcm = b"".join(e["pcm"] for e in buffered if e["type"] == "speech.audio.delta")
                failed = any(e["type"] == "error" for e in buffered)
                score = None
                if not failed and len(pcm) > 0.3 * 88200:
                    heard = yield from keepalive(check.hear, pcm)
                    score = check.opening(heard)
                    word_check.log(f"opening {attempt + 1}: {score if score is None else round(score, 2)} heard "
                                   f"{word_check.heard_text(heard)[:60]!r}")
                if not failed and score is not None and score < word_check.OPENING_PASS and attempt < word_check.RETRIES:
                    events.close()
                    continue
                yield from buffered
                if finished or failed:
                    events.close()
                    return result
                return (yield from events)

        def finish_line(self, request, turn, direction, check, pcm, codes, done):
            """A streamed take that ended before its last words: those words are made as a
            continuation (the take so far as history, as a long script's next chunk) and
            streamed after a pause. Returns (audio, codes, done event) of the addition, or
            None when the take said everything."""
            heard = yield from keepalive(check.hear, pcm)
            tail = word_check.missing_tail(check, heard, stalled=done.get("finish_reason") == "stall")
            if tail is None:
                return None
            word_check.log(f"the take ended before {tail!r}; making the rest")
            if turn.startswith(de.ENVELOPE_MARKER):
                turn = request.text
            rest = (prefixed_turn(direction, tail) if turn.startswith("[") and direction
                    else directed_turn(direction, tail))
            rows = self.fit_prompt(rest, self.voices(request, None), [(turn, codes)], request.character)
            pause = join_pause(word_check.heard_text(heard), pcm)  # the pause after what was said
            if pause:
                yield {"type": "speech.audio.delta", "pcm": pause}
            self.use_seed(0)  # the rest is made as a long script's next chunk is
            with self.joined_after(codes):
                frames, reason, more = yield from self.generate(rows, request, rest, live=True)
            if not frames:
                return None
            done = dict(done, frames=done["frames"] + frames, finish_reason=reason)
            return pause + more, self.last_codes(frames), done

        def make_rest(self, request, turn, rest, voices, history, codes, *, live):
            """The words a take or chunk ended without, made as a continuation of it: its
            own turn and codes are the latest history turn, and the rest is decoded right
            after them. Returns (audio, codes, finish reason) of the addition, or None.
            Making the rest keeps what was said and adds only what is missing (a voice
            chat on Drama 3 resumed 25 of 27 stalls this way, none lost or repeated, where
            a whole remake repeated words: gabacool/bidirectional-voice-claude #26)."""
            rows = self.fit_prompt(rest, voices, list(history) + [(turn, codes)], request.character)
            self.use_seed(0)  # as a long script's next chunk
            with self.joined_after(codes):
                frames, reason, more = yield from self.generate(rows, request, rest, live=live)
            if not frames:
                return None
            return more, self.last_codes(frames), reason

        def checked_speech(self, body, deadline, request, check, direction, words):
            """The Space's path, held until the take is made and its words checked;
            made again (at most word_check.RETRIES more times) when words are missing,
            and the best take is sent."""
            best, register_held = None, None
            register = register_guard.active()
            tries, attempt = 1 + word_check.RETRIES, -1
            while (attempt := attempt + 1) < tries:
                self.use_seed(attempt)
                pcm, done, beat = bytearray(), None, time.perf_counter()
                with self.warm_start(request):  # (decoding runs while the events are read)
                    for event in self.voice_guided(super().speech(body, deadline=deadline), request):
                        if event["type"] == "speech.audio.delta":
                            pcm += event["pcm"]
                            if time.perf_counter() - beat > KEEPALIVE_S:
                                beat = time.perf_counter()
                                yield {"type": "speech.audio.delta", "pcm": b""}
                        elif event["type"] == "speech.audio.done":
                            done = event
                        else:
                            yield event
                            return
                if done is None:
                    break
                codes = self.last_codes(done["frames"]) if done.get("frames") else None
                heard = yield from keepalive(check.hear, bytes(pcm))
                score = check.score(heard)
                word_check.log(f"take {attempt + 1}: {score:.2f} heard {word_check.heard_text(heard)[:80]!r}")
                if codes is not None and score < word_check.PASS and register_held is None and (
                        tail := word_check.missing_tail(check, heard, done.get("finish_reason") == "stall")):
                    word_check.log(f"take {attempt + 1} ended before {tail!r}; making the rest")
                    turn = request.text if body["input"].startswith(de.ENVELOPE_MARKER) else body["input"]
                    rest = (prefixed_turn(direction, tail) if turn.startswith("[") and direction
                            else directed_turn(direction, tail))
                    added = yield from self.make_rest(request, turn, rest, self.voices(request, None), [], codes,
                                                      live=False)
                    if added is not None:
                        more, more_codes, reason = added
                        pcm += more
                        codes = torch.cat((codes, more_codes), dim=1)
                        done = dict(done, frames=codes.shape[1], finish_reason=reason)
                        heard = yield from keepalive(check.hear, bytes(pcm))
                        score = check.score(heard)
                        word_check.log(f"take {attempt + 1} with the rest: {score:.2f} heard "
                                       f"{word_check.heard_text(heard)[:80]!r}")
                jump = register_guard.pcm_jump(bytes(pcm)) if register and score >= word_check.PASS else None
                if register_held is not None:  # the register guard's remake
                    if score >= word_check.PASS and (jump or 0.0) < register_held:
                        best = (score, bytes(pcm), done, codes, heard)
                        log(f"the remake's jump is {jump or 0.0:.1f}; it is kept")
                    else:
                        log("the remake is no better; the first take is kept")
                    break
                if best is None or score > best[0]:
                    best = (score, bytes(pcm), done, codes, heard)
                if score >= word_check.PASS:
                    if jump:
                        log(f"a phrase opens {jump:.1f} semitones above the others (register guard), "
                            f"making it once more")
                        best = (score, bytes(pcm), done, codes, heard)
                        register_held, tries = jump, attempt + 2
                        continue
                    break
            if best is None:
                return
            score, pcm, done, codes, heard = best
            pcm, codes = trim_tail(words, heard, pcm, codes)
            if codes is not None:
                done = dict(done, frames=codes.shape[1])
            for offset in range(0, len(pcm), PACKET_BYTES):
                yield {"type": "speech.audio.delta", "pcm": pcm[offset:offset + PACKET_BYTES]}
            if codes is not None:
                # kept (codes + audio) so an edit of this line can be infilled
                remember(request.text, LongTake(line=words, direction=direction, chunks=[
                    ChunkAudio(words, request.text, codes, pcm)]))
            yield done

        # Planning ---------------------------------------------------------------

        def plan(self, request):
            note, direction, words = parse_turn(request.text)
            if subtitles.is_timed(words):
                return self.plan_timed(request, note, words)
            previous = long_take(request.history[-1]) if request.history else None
            single = CACHE.get(request.history[-1]) if request.history else None
            if (previous is None and single is not None and note is None and SETTINGS["infill"]
                    and direction == single.direction and spoken(single.line) != spoken(words)):
                # An edit of a one-chunk take: its changed words can be infilled
                planned = chunking.plan_edit(single.line, [c.text for c in single.chunks], words)
                if planned:
                    return note, words, planned, single
            if previous is not None:
                if not SETTINGS["long_retake"]:
                    raise de.RequestError(
                        f"This take was generated in {len(previous.chunks)} chunks. Tick “Retake / Edit long "
                        "takes (experimental)” under the stage to retake or edit it")
                planned = chunking.plan_edit(previous.line, [c.text for c in previous.chunks], words)
                if note or direction != previous.direction:  # a new direction redoes every chunk
                    for chunk in planned:
                        chunk.changed = True
                if not planned:
                    raise de.RequestError("the edited script is empty")
                return note, words, planned, previous
            chunks = chunking.split(words)
            if len(chunks) <= 1:
                return None
            return note, words, [chunking.PlannedChunk(text) for text in chunks], None

        def plan_timed(self, request, note, words):
            """A subtitle script: one chunk per cue. An edit keeps the cues whose line is
            unchanged and redoes (or infills) a changed cue in place of the old one at the
            same start time."""
            lines = subtitles.timed_lines(words)
            previous = CACHE.get(request.history[-1]) if request.history else None
            if previous is not None and not subtitles.is_timed(previous.line):
                previous = None
            if previous is not None and len(previous.chunks) > 1 and not SETTINGS["long_retake"]:
                raise de.RequestError(
                    f"This take was generated in {len(previous.chunks)} chunks. Tick “Retake / Edit long "
                    "takes (experimental)” under the stage to retake or edit it")
            old = [c.text for c in previous.chunks] if previous is not None else []
            planned = []
            for line in lines:
                if line in old:
                    planned.append(chunking.PlannedChunk(line, [old.index(line)], changed=bool(note)))
                    continue
                start = subtitles.cue(line).start
                same = [i for i, o in enumerate(old) if start is not None and subtitles.cue(o).start == start]
                planned.append(chunking.PlannedChunk(line, same[:1]))
            return note, words, planned, previous

        # Generation -------------------------------------------------------------

        def voices(self, request, previous):
            references = request.references[:request.voice_count]
            if previous is not None and references and spoken(references[0].get("text") or "") == spoken(previous.line):
                # Surprise / Design retakes use the take itself as the voice: its first chunk
                # says the same, and may be longer than the engine's reference limit.
                first = previous.chunks[0]
                return [de.Reference(transcript=spoken(first.text) or first.text, codes=first.codes)]
            return [de.Reference(transcript=r.get("text"), codes=self.encode(r)) for r in references]

        def fit_prompt(self, turn, voices, history, character, streaming=False):
            """The prompt with as much history as fits, dropping the oldest turns first."""
            for start in range(len(history) + 1):
                segments = de.build_segments(turn, voices, history[start:], character)
                if streaming:
                    segments = with_stream_instruction(segments)
                rows = de.segment_rows(segments, self.tokenizer, self.config)
                if len(rows) <= de.MAX_PROMPT_ROWS:
                    return rows
            raise de.RequestError(f"one chunk's prompt has {len(rows)} rows, over {de.MAX_PROMPT_ROWS}")

        def last_codes(self, frames: int) -> torch.Tensor:
            source = self.decode.out_frames if getattr(self, "decode", None) is not None else self.graphs.out_frames
            return source[:frames, 1:].T.contiguous().cpu()

        def generate(self, rows, request, turn: str, *, live: bool = True):
            """One chunk through the Space's own decode and streaming; returns
            (frames, finish_reason, pcm). `turn` is the chunk's own text: the prompt is
            built here, but the frame loop reads the text (the pause fix's length guard).
            Not live: the audio is held back (with empty keepalive packets) and returned
            for the caller to stream."""
            body = {"input": turn, "temperature": request.temperature, "top_p": request.top_p,
                    "top_k": request.top_k, "max_new_tokens": chunking.CHUNK_MAX_FRAMES}
            pcm, done = bytearray(), None
            beat = time.perf_counter()
            self._prompt_override = rows
            self.end_nudge = True  # pause_fix: a chunk ends near its last word, not ~1 s of hiss later
            try:
                for event in base.speech(self, body):
                    if event["type"] == "speech.audio.delta":
                        pcm += event["pcm"]
                        if live:
                            yield event
                        elif time.perf_counter() - beat > KEEPALIVE_S:
                            beat = time.perf_counter()
                            yield {"type": "speech.audio.delta", "pcm": b""}
                    elif event["type"] == "speech.audio.done":
                        done = event
                    else:
                        yield event
                        raise de.RequestError(event.get("message") or "a chunk failed")
            finally:
                self._prompt_override = None
                self.end_nudge = False
            return done["frames"], done["finish_reason"], bytes(pcm)

        def try_infill(self, previous, chunk, note, request):
            """The chunk as an infill edit of the old chunk it replaces, or None (then it
            is regenerated whole): only for edits (no note) inside one old chunk."""
            if (note is not None or previous is None or len(chunk.old) != 1 or not SETTINGS["infill"]):
                return None
            transcriber = getattr(sys.modules.get("server"), "transcriber", None)
            if transcriber is None or transcriber.state != "ready":
                return None
            old = previous.chunks[chunk.old[0]]
            old_text, new_text = subtitles.cue(old.text).text, subtitles.cue(chunk.text).text
            if old.words is None:
                old.words = yield from keepalive(infill.transcribe_words, transcriber, old.pcm)
            window = infill.window(old_text, new_text, infill.align(old_text, old.words), old.codes.shape[1])
            if window is None:
                infill.log("edit not infillable (alignment, size or no spoken change); regenerating the chunk")
                return None
            rows = de.segment_rows(infill.segments(new_text, window, old.codes, de.with_speaker_tag),
                                   self.tokenizer, self.config)
            for attempt in range(2):
                self.use_seed(attempt)
                frames, reason, _ = yield from self.generate(rows, request, window.new_text, live=False)
                if not frames or reason == "stall":
                    continue
                codes = torch.cat((old.codes[:, :window.start], self.last_codes(frames), old.codes[:, window.end:]), 1)
                pcm = self.vocode_pcm(codes)
                heard = yield from keepalive(infill.transcribe_words, transcriber, pcm)
                share = infill.heard_share(new_text, window, heard)
                whole, before = infill.match_ratio(new_text, heard), infill.match_ratio(old_text, old.words)
                infill.log(f"{window.old_text!r} -> {window.new_text!r}: frames {window.start}-{window.end} "
                           f"({window.end - window.start}) -> {frames} (hint {window.frames_hint}), heard {share:.0%}, "
                           f"whole chunk {whole:.2f} (was {before:.2f})")
                if share >= infill.VERIFY_MIN and whole >= before - infill.WHOLE_SLACK:
                    return ChunkAudio(chunk.text, directed_turn(parse_turn(request.text)[1], chunk.text), codes, pcm, heard)
            infill.log("the infilled chunk did not pass (new words not heard, or the chunk matches its script "
                       "worse than before); regenerating the chunk")
            return None

        @torch.inference_mode()
        def vocode_pcm(self, codes: torch.Tensor) -> bytes:
            audio = self.vocode(codes.to(self.device))  # the codec's device (the host for MLX)
            return (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy().tobytes()

        @torch.inference_mode()
        def tidy_tail(self, script: str, heard, pcm: bytes, codes: torch.Tensor):
            """(pcm, codes) of a held chunk without the hiss before its first and after its
            last word (see speech_span)."""
            import numpy as np

            if not SETTINGS["tail_tidy"] or codes is None:
                return pcm, codes
            span = speech_span(pcm, codes.shape[1], heard, script)
            if span is None:
                return pcm, codes
            first, end = span
            whole = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            samples = shape_tail(whole, first, end)
            if samples is None:  # the chunk ends on its last word
                return pcm, codes
            stop, head = samples.size, first * 2048
            codes = codes[:, : min(codes.shape[1], int(np.ceil(stop / 2048)))]
            log(f"tail: cut {(whole.size - stop) / 44100:.2f}s after the last word, held {TAIL_HOLD_S:.2f}s "
                f"and faded {TAIL_FADE_OUT_S:.2f}s"
                + (f"; {head / 44100:.2f}s before the first faded in" if head else ""))
            return samples.astype("<i2").tobytes(), codes.contiguous()

        # Joins decoded as one take ---------------------------------------------------
        #
        # Each chunk is its own generation, so the stream decoded its first windows with no
        # left context. The codec's receptive field is hundreds of frames (the Space decodes
        # every window with STREAM_OVERLAP_FRAMES = 320 frames before it), and cold, a chunk's
        # first ~0.3 s came out as a loud, garbled burst at every join (0-2 dB SNR against a
        # continuous decode). While a generation follows earlier audio (a long script's next
        # chunk, the rest of a line the word check found missing), its first windows are
        # decoded after that audio's last frames, up to the same 320 frames of context: as
        # inside one take, in windows no larger than the Space's own (no extra memory).

        _join_codes = None  # [codebooks, frames] the generation being decoded follows
        _window = (1, 0)  # (start, emitted) of the stream window being decoded

        @contextlib.contextmanager
        def joined_after(self, codes):
            before, self._join_codes = self._join_codes, codes
            try:
                yield
            finally:
                self._join_codes = before

        def warm_start(self, request, voices=None):
            """A take's first windows decoded after its reference clip's codes (WARM_START),
            as a long script's later chunks are decoded after the chunk before them."""
            if not WARM_START:
                return contextlib.nullcontext()
            if voices is None:
                voices = self.voices(request, None) if request.voice_count else []
            codes = voices[0].codes if voices and voices[0].codes is not None else None
            return self.joined_after(codes) if codes is not None else contextlib.nullcontext()

        def queue_window(self, start: int, end: int, emitted: int):
            self._window = (start, emitted)
            return super().queue_window(start, end, emitted)

        def vocode(self, codes):
            start, emitted = self._window
            context = (de.STREAM_OVERLAP_FRAMES - emitted) if start == 0 and self._join_codes is not None else 0
            context = min(context, self._join_codes.shape[1]) if context > 0 else 0
            self._window = (1, 0)  # only the window queue_window named
            if context <= 0:
                return super().vocode(codes)
            before = self._join_codes[:, -context:].to(device=codes.device, dtype=codes.dtype)
            return super().vocode(torch.cat((before, codes), dim=1))[context * de.FRAME_SAMPLES:]

        def long_speech(self, request, note, words, planned, previous):
            started = time.perf_counter()
            voices = self.voices(request, previous)
            direction = parse_turn(request.text)[1]
            prefix = previous is None and self.needs_prefix(request, direction)
            # Earlier short takes the server sends (a fresh long line retaken from a short take)
            extra = []
            if previous is None:
                for turn, reference in zip(request.history, request.references[request.voice_count:]):
                    extra.append((turn, self.encode(reference)))
            regenerate = sum(chunk.changed for chunk in planned)
            log(f"{len(planned)} chunks, {regenerate} to generate"
                + (f", {len(planned) - regenerate} kept" if previous is not None else "")
                + (f", note “{note}”" if note else ""))
            records: list[ChunkAudio] = []
            register_before: list[float] = []  # phrase openings of the chunks so far (register_guard)
            frames_total, finish, first_audio = 0, "stop", None
            timed = subtitles.is_timed(words)
            position = 0  # bytes of audio sent so far (the subtitle timeline)

            def plain(text: str) -> str:  # a chunk's words without its subtitle timing
                return subtitles.cue(text).text if timed else text

            crossfade = SETTINGS["crossfade"] and not timed  # (timed cues keep their exact start)
            pending = None  # the last chunk's held tail (float samples), mixed into the next one's start

            def shaped(pcm: bytes, frames: int, script: str, heard=None):
                """(samples, start) of a chunk for a crossfade join, or None."""
                import numpy as np

                span = speech_span(pcm, frames, heard, script) if crossfade else None
                if span is None:
                    return None
                return crossfade_shape(np.frombuffer(pcm, dtype="<i2").astype(np.float32), *span, pause_after(script))

            def send(pcm: bytes, shape=None):
                """A chunk's audio packets, the held tail before it mixed into its start; with a
                shape, its own tail from the shape's start is held for the next chunk."""
                nonlocal pending
                import numpy as np

                data = pcm
                if shape is not None or pending is not None:
                    if shape is not None:
                        samples, start = shape
                    else:
                        samples, start = np.frombuffer(pcm, dtype="<i2").astype(np.float32), None
                    if pending is not None:
                        samples = samples.copy()
                        overlap = min(pending.size, samples.size)
                        samples[:overlap] += pending[:overlap]
                        if pending.size > overlap:  # a chunk shorter than the tail before it
                            samples = np.concatenate((samples, pending[overlap:]))
                            start = None
                        pending = None
                    if start is not None:  # (an empty tail still marks the join as made)
                        samples, pending = samples[:start], samples[start:]
                    data = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
                for offset in range(0, len(data), PACKET_BYTES):
                    yield {"type": "speech.audio.delta", "pcm": data[offset:offset + PACKET_BYTES]}

            for index, chunk in enumerate(planned):
                cue = subtitles.cue(chunk.text) if timed else None
                if cue is not None and cue.start is not None:
                    # a subtitle cue starts at its time (or at once, if the one before ran over)
                    pause = room_tone([r.pcm for r in records[-3:]], max(0, int(cue.start * 44100) - position // 2))
                elif records and pending is not None:  # a crossfade join: the held tail is the pause
                    pause = b""
                elif records:  # a pause at the join, kept out of the chunk's cached audio
                    pause = join_pause(plain(records[-1].text), records[-1].pcm,
                                       tone=[r.pcm for r in records[-3:]])
                else:
                    pause = b""
                if pause:
                    position += len(pause)
                    yield {"type": "speech.audio.delta", "pcm": pause}
                if not chunk.changed:
                    record = previous.chunks[chunk.old[0]]
                    yield from send(record.pcm, shaped(record.pcm, record.codes.shape[1], plain(record.text)))
                elif (record := (yield from self.try_infill(previous, chunk, note, request))) is not None:
                    yield from send(record.pcm, shaped(record.pcm, record.codes.shape[1], plain(record.text)))
                else:
                    text = one_line(subtitles.spoken_turn(cue) if cue is not None else chunk.text)
                    if cue is None:
                        text += length_hint(text)
                    turn = (retake_turn(note, spoken(text) or text) if note
                            else prefixed_turn(direction, text) if prefix else directed_turn(direction, text))
                    if note and cue is not None and cue.frames:
                        turn += f" [audio tokens: {cue.frames}]"
                    streaming = SETTINGS["lookahead"]
                    following = one_line(plain(planned[index + 1].text)) if index + 1 < len(planned) else ""
                    prompt_turn = f"{turn}{END_OF_TURN}{following}" if streaming else turn
                    history = list(extra)
                    if records:
                        if len(records) > 1:
                            history.append((records[0].turn, records[0].codes))  # pinned first chunk
                        history.append((records[-1].turn, records[-1].codes))  # the latest one
                    if previous is not None and chunk.old:
                        olds = [previous.chunks[i] for i in chunk.old]
                        history.append((" ".join(plain(o.text) for o in olds), torch.cat([o.codes for o in olds], dim=1)))
                    rows = self.fit_prompt(prompt_turn, voices, history, request.character, streaming)
                    # The first chunk streams as it is made. Later ones are made while the
                    # listener still hears the one before (about 2x real time), so they are
                    # held until complete: a chunk that stalls into a drone is made again
                    # once without any of it having been heard.
                    check = word_check.checker(plain(chunk.text))
                    opening = check if check is not None and not records and word_check.OPENING else None
                    # streamed (no word check): still heard once made, and its missing words made
                    ending = check if word_check.ENDING and not SETTINGS["word_check"] else None
                    if not SETTINGS["word_check"]:
                        check = None
                    live = not records and check is None
                    best, register_held = None, None
                    register = check is not None and not live and register_guard.active()
                    tries, attempt = 1 if live else 1 + max(1, word_check.RETRIES if check else 1), -1
                    while (attempt := attempt + 1) < tries:
                        self.use_seed(attempt)  # every chunk starts from the take's seed
                        joined = self.joined_after(records[-1].codes) if records else self.warm_start(request, voices)
                        with joined:
                            if live and opening is not None:
                                frames, reason, pcm = yield from self.opening_checked(
                                    lambda: self.generate(rows, request, turn, live=True), opening)
                            elif check is not None:  # held and checked: voice guidance
                                frames, reason, pcm = yield from self.voice_guided(
                                    self.generate(rows, request, turn, live=live), request)
                            else:
                                frames, reason, pcm = yield from self.generate(rows, request, turn, live=live)
                        codes = self.last_codes(frames) if frames else None
                        score, heard = (0.0 if reason == "stall" else 1.0), None
                        if check is not None and frames and reason != "stall":
                            heard = yield from keepalive(check.hear, pcm)
                            score = check.score(heard)
                            word_check.log(f"chunk {index + 1} take {attempt + 1}: {score:.2f} heard "
                                           f"{word_check.heard_text(heard)[:80]!r}")
                        listener = check or ending
                        if (listener is not None and frames and register_held is None
                                and (ending is not None or score < word_check.PASS)):
                            if heard is None:
                                heard = yield from keepalive(listener.hear, pcm)
                            tail = word_check.missing_tail(listener, heard, stalled=reason == "stall")
                            if tail is not None:
                                word_check.log(f"chunk {index + 1} take {attempt + 1} ended before {tail!r}; "
                                               f"making the rest")
                                rest = (retake_turn(note, spoken(tail) or tail) if note
                                        else prefixed_turn(direction, tail) if prefix
                                        else directed_turn(direction, tail))
                                added = yield from self.make_rest(request, turn, rest, voices, history, codes,
                                                                  live=live)
                                if added is not None:
                                    more, more_codes, reason = added
                                    pcm, codes = pcm + more, torch.cat((codes, more_codes), dim=1)
                                    frames = codes.shape[1]
                                    heard = yield from keepalive(listener.hear, pcm)
                                    score = check.score(heard) if check is not None else (
                                        0.0 if reason == "stall" else 1.0)
                                    word_check.log(f"chunk {index + 1} take {attempt + 1} with the rest: "
                                                   f"{listener.score(heard):.2f} heard "
                                                   f"{word_check.heard_text(heard)[:80]!r}")
                        jump = (register_guard.pcm_jump(pcm, register_before)
                                if register and frames and score >= word_check.PASS else None)
                        if register_held is not None:  # the register guard's remake
                            if score >= word_check.PASS and (jump or 0.0) < register_held:
                                best = (score, frames, reason, pcm, codes, heard, prompt_turn)
                                log(f"chunk {index + 1}: the remake's jump is {jump or 0.0:.1f}; it is kept")
                            else:
                                log(f"chunk {index + 1}: the remake is no better; the first take is kept")
                            break
                        if best is None or score > best[0]:
                            best = (score, frames, reason, pcm, codes, heard, prompt_turn)
                        if live or score >= word_check.PASS:
                            if jump:
                                log(f"chunk {index + 1}: a phrase opens {jump:.1f} semitones above the others "
                                    f"(register guard), making it once more")
                                best = (score, frames, reason, pcm, codes, heard, prompt_turn)
                                register_held, tries = jump, attempt + 2
                                continue
                            break
                        if reason == "stall":
                            log(f"chunk {index + 1}: stalled after {len(pcm) / 88200:.1f}s, making it again")
                        if check is None and reason != "stall":
                            break
                        if attempt + 1 == tries and check is not None and streaming and following:
                            # Every take missed its words while shown the next chunk: the
                            # model may be voicing that chunk here (then said again by its
                            # own), so one more take without it
                            log(f"chunk {index + 1}: no take passed while reading ahead, making it once without")
                            streaming, prompt_turn, tries = False, turn, tries + 1
                            rows = self.fit_prompt(turn, voices, history, request.character, False)
                    _, frames, reason, pcm, codes, heard, prompt_turn = best
                    if check is not None and frames:
                        pcm, codes = trim_tail(plain(chunk.text), heard, pcm, codes)
                        frames = codes.shape[1]
                    shape = None
                    if not live and frames and reason != "stall":
                        if crossfade:  # the chunk is kept whole; its tail fades under the next one
                            shape = shaped(pcm, codes.shape[1], plain(chunk.text), heard)
                        else:
                            pcm, codes = self.tidy_tail(plain(chunk.text), heard, pcm, codes)
                            frames = codes.shape[1]
                    if not live:
                        yield from send(pcm, shape)
                    if reason in ("length", "stall"):
                        finish = reason
                    if not frames:
                        log(f"chunk {index + 1}: no audio (end of speech at once)")
                        continue
                    record = ChunkAudio(chunk.text, prompt_turn, codes, pcm)
                    lead, tail = edge_silence(pcm)
                    log(f"chunk {index + 1}: {len(pcm) / 88200:.2f}s, silence {lead:.2f}s at the start, "
                        f"{tail:.2f}s at the end, finish {reason}, ends with {chunk.text[-12:]!r}")
                position += len(record.pcm)
                frames_total += record.codes.shape[1]
                first_audio = first_audio or round((time.perf_counter() - started) * 1000)
                records.append(record)
                if register_guard.active():
                    register_before += register_guard.pcm_openings(record.pcm)
            if pending is not None:  # the last chunk's tail, faded out (without the pause padding)
                import numpy as np

                pending = np.trim_zeros(pending, "b")
                yield from send(b"")
            if not records:
                raise de.RequestError("no chunk produced audio")
            remember(request.text, LongTake(line=words, chunks=records, direction=direction))
            total_ms = round((time.perf_counter() - started) * 1000)
            log(f"{len(records)} chunks, {frames_total} frames in {total_ms / 1000:.1f}s")
            yield {"type": "speech.audio.done", "finish_reason": finish, "frames": frames_total,
                   "timings": {"chunks": len(records), "generated": regenerate, "first_audio_ms": first_audio,
                               "total_ms": total_ms}}

    LongEngine.__name__ = f"Long{base.__name__}"
    return LongEngine


# ─── Server ───────────────────────────────────────────────────────────────────


def reference_limit() -> float:
    return REF_UNLOCKED_SECONDS if SETTINGS["ref_unlocked"] else REF_DEFAULT_SECONDS


def set_ref_unlocked(server, unlocked: bool) -> None:
    """The Studio's checkbox: how long *new* reference clips may be. The model's memory
    layout does not follow it; it follows the voice each take uses (see LONG_VOICE_KEY)."""
    SETTINGS["ref_unlocked"] = bool(unlocked)
    server.REF_MAX_SECONDS = reference_limit()


# Set on the engine request of a take whose clone voice is longer than 20 s: that take
# needs the 60 s memory layout (serve.py re-places the experts for it, between takes, and
# back for the next take with a voice of 20 s or less). Not read by the Space's engine.
LONG_VOICE_KEY = "drama3_long_reference"
# Set on the engine request of a subtitle-timed script: its silences are the timeline
TIMED_KEY = "drama3_timed"
LONG_VOICE_TAG = " · 60 s"


def is_long_voice(voice) -> bool:
    return voice is not None and voice.duration > REF_DEFAULT_SECONDS + 0.5


def install_engine() -> None:
    import drama3_engine

    drama3_engine.Engine = make_engine(drama3_engine.Engine)


def install_server(server) -> None:
    """Limits and the empty-note edit. The server looks these names up per request."""
    server.MAX_TEXT_CHARS = 10**9  # no script limit: long scripts are chunked
    server.MAX_TRANSCRIPT_CHARS = TRANSCRIPT_CHARS
    server.REF_MAX_SECONDS = reference_limit()

    clean_text, retake_text = server.clean_text, server.retake_text

    def lenient_clean_text(value, what, limit, *, single_line=False):
        if what == "note" and not (value or "").strip():
            return ""  # an empty note asks for an edit
        return clean_text(value, what, limit, single_line=single_line)

    def edit_or_retake(line: str, note: str) -> str:
        return line if not note else retake_text(line, note)

    server.clean_text = lenient_clean_text
    server.retake_text = edit_or_retake

    engine_request = server.engine_request

    def with_voice_layout(choice, turn_text, history, body):
        """Mark takes whose clone voice is longer than 20 s: they get the 60 s layout, so a
        60 s voice keeps working after the checkbox is turned off, and other takes do not
        pay for it. (A retake keeps its original take's voice, so it follows it too.)"""
        payload = engine_request(choice, turn_text, history, body)
        if choice.mode == "clone" and is_long_voice(getattr(choice, "voice", None)):
            payload[LONG_VOICE_KEY] = True
        if subtitles.is_timed(parse_retake(turn_text)[1]):
            payload[TIMED_KEY] = True
        return payload

    server.engine_request = with_voice_layout

    class TimedSilenceGuard(server.SilenceGuard):
        """The server's silence guard, except for subtitle-timed takes: there it keeps
        the silence before the first cue and between cues (it would cut pauses to
        0.75 s, and end the take after 4 s of silence). The take is identified by the
        engine payload of the take loop that creates the guard (server.py's stream())."""

        def __init__(self, sample_rate: int) -> None:
            super().__init__(sample_rate)
            payload = sys._getframe(1).f_locals.get("payload")
            if isinstance(payload, dict) and payload.get(TIMED_KEY):
                self.max_pause = self.stall_limit = 1 << 40
                self.speech_detected = True

    server.SilenceGuard = TimedSilenceGuard

    from fastapi import HTTPException
    from fastapi.responses import Response

    async def take_subtitles(take_id: str) -> Response:
        """The take's subtitles: its script, timed by Whisper on its audio."""
        take = server.store.take(take_id)
        if take is None:
            raise HTTPException(404, "That take has expired.")
        if server.transcriber.state != "ready":
            raise HTTPException(503, "Whisper is still loading; try again in a moment.")
        import asyncio

        words = await asyncio.to_thread(infill.transcribe_words, server.transcriber, take.pcm)
        srt = subtitles.to_srt(parse_retake(take.line)[1], words, take.seconds)
        return Response(srt, media_type="application/x-subrip", headers={
            "Content-Disposition": f'attachment; filename="drama3-{take_id}.srt"'})

    server.app.add_api_route("/api/takes/{take_id}.srt", take_subtitles, methods=["GET"], response_model=None,
                             dependencies=[server.Depends(server.identify)])  # (no model: /openapi.json)
    server.app.router.routes.insert(0, server.app.router.routes.pop())  # ahead of the Studio's mount at /

    register_voice = server.register_voice

    async def register_marked_voice(wav, sample_rate, label, fallback_text=""):
        """Voices longer than 20 s say so in the voice list (they use the 60 s layout)."""
        result = await register_voice(wav, sample_rate, label, fallback_text)
        voice = server.store.voice(result["id"])
        if is_long_voice(voice) and not voice.label.endswith(LONG_VOICE_TAG):
            voice.label += LONG_VOICE_TAG
            result["label"] = voice.label
        return result

    server.register_voice = register_marked_voice
