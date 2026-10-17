"""Subtitles in and out.

- Import: an .srt file becomes a timed script, one line per cue:
  "{00:01.200 --> 00:03.500} Text of the cue." The braces are not model syntax; the
  engine removes them, voices each cue as its own chunk with Drama 3's duration hint
  ("[audio tokens: N]", technical report Table 10) and places it at its start time.
- Export: the .srt of any take, from the Whisper word timestamps of its audio aligned
  to its script (the infill aligner), grouped into cues at sentence ends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import infill

TIMED = re.compile(
    r"^\{\s*(?P<start>(?:\d+:)?\d+:\d+(?:[.,]\d+)?)\s*(?:-->|→)\s*(?P<end>(?:\d+:)?\d+:\d+(?:[.,]\d+)?)\s*\}\s*(?P<text>.*)$")
SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
CUE_MAX_CHARS = 84  # two lines of 42
CUE_MAX_S = 6.0


@dataclass
class Cue:
    start: float | None  # seconds; None: not timed (follows the cue before)
    end: float | None
    text: str

    @property
    def frames(self) -> int | None:
        if self.start is None or self.end is None or self.end <= self.start:
            return None
        return max(4, int(round((self.end - self.start) * infill.FPS)))


def seconds(stamp: str) -> float:
    parts = stamp.replace(",", ".").split(":")
    value = 0.0
    for part in parts:
        value = value * 60 + float(part)
    return value


def stamp(value: float, srt: bool = False) -> str:
    value = max(0.0, value)
    hours, rest = divmod(value, 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((secs - int(secs)) * 1000))
    if millis == 1000:
        secs, millis = secs + 1, 0
    if srt:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"
    head = f"{int(hours):02d}:" if hours else ""
    return f"{head}{int(minutes):02d}:{int(secs):02d}.{millis:03d}"


def cue(line: str) -> Cue:
    """The cue of a script line (untimed lines: start and end None)."""
    match = TIMED.match(line.strip())
    if not match:
        return Cue(None, None, line.strip())
    return Cue(seconds(match["start"]), seconds(match["end"]), match["text"].strip())


def is_timed(script: str) -> bool:
    return any(TIMED.match(line.strip()) for line in (script or "").splitlines())


def timed_lines(script: str) -> list[str]:
    return [line.strip() for line in script.splitlines() if line.strip()]


def spoken_turn(c: Cue) -> str:
    """What the model is given for a cue: its text, with the duration hint."""
    return f"{c.text} [audio tokens: {c.frames}]" if c.frames else c.text


def from_srt(data: str) -> str:
    """A timed script from SRT text (cue numbers dropped, cue lines joined)."""
    lines, block = [], []
    for raw in (data or "").replace("\r\n", "\n").replace("﻿", "").split("\n") + [""]:
        if raw.strip():
            block.append(raw.strip())
            continue
        if block:
            times = next((i for i, b in enumerate(block) if SRT_TIME.search(b)), None)
            if times is not None:
                m = SRT_TIME.search(block[times])
                start = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
                end = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
                text = " ".join(re.sub(r"<[^<>]*>|\{\\[^{}]*\}", "", b) for b in block[times + 1:]).strip()
                if text:
                    lines.append(f"{{{stamp(start)} --> {stamp(end)}}} {text}")
            block = []
    return "\n".join(lines)


def plain_script(script: str) -> str:
    """The script without timing markers (what was said)."""
    return "\n".join(cue(line).text for line in (script or "").splitlines() if line.strip())


def to_srt(script: str, whisper_words: list[tuple[str, float, float]], duration: float) -> str:
    """Cues for a take: its script's words timed by Whisper, split at sentence ends (or
    at CUE_MAX_CHARS / CUE_MAX_S). Words Whisper missed take their neighbours' times."""
    text = plain_script(script)
    words = infill.align(text, whisper_words)
    if not words:
        return ""
    # Fill the gaps: an unheard word gets the end of the word before / start of the next
    known = [i for i, w in enumerate(words) if w.start is not None]
    if not known:
        return f"1\n{stamp(0, True)} --> {stamp(duration, True)}\n{text}\n"
    for i, w in enumerate(words):
        if w.start is None:
            before = max((k for k in known if k < i), default=None)
            after = min((k for k in known if k > i), default=None)
            w.start = words[before].end if before is not None else words[after].start
            w.end = words[after].start if after is not None else words[before].end
    cues, current = [], []
    for w in words:
        current.append(w)
        joined = " ".join(x.text for x in current)
        if (w.text[-1:] in ".!?…。！？" or len(joined) >= CUE_MAX_CHARS
                or current[-1].end - current[0].start >= CUE_MAX_S):
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    out = []
    for n, group in enumerate(cues, 1):
        start, end = group[0].start, max(group[-1].end, group[0].start + 0.3)
        out.append(f"{n}\n{stamp(start, True)} --> {stamp(min(end, duration), True)}\n{' '.join(w.text for w in group)}\n")
    return "\n".join(out)
