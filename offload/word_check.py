"""Every word read: each take (or chunk) is checked against its script and made again
when words are missing.

The model sometimes loses its place in a line: it skips a span (more often a tagged one,
and more often after a "Direction:"), stops before the end, or reads the direction
aloud. "Direction: Having a chat with Elizabeth Holmes from Theranos\\nLine: <whispers>
Okay... Alright... Good chat. </whispers> Are you being real, right now?" came out
complete in 5 of 10 takes with a random voice; no sampling setting fixed it.

So the audio is held until the take is done, the server's Whisper transcribes it, and
the letters it heard are compared with the script's spoken words (letters, so "all
right" matches "alright"; numbers spelled out). Below PASS it is made again, at most
RETRIES more times, and the best one is kept. Measured on 60 takes: every take missing
a span, an ending or reading the direction scored 0.76 or less, every complete one 0.92
or more; mishearings (Whisper's "good shot") cost a retry at worst.

The cost: a line is heard once it is made and checked, not as it is made (about 1-2 s
later for a short line), plus the retries when they happen, so it is the Studio's
"Check every word before playing" option (off by default; DRAMA3_WORD_CHECK=1 for the API).
Streamed takes get the opening check instead (below).
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import threading
import types
import unicodedata

import infill

RETRIES = int(os.environ.get("DRAMA3_WORD_RETRIES", "2"))
PASS = 0.9
# The opening check, for streamed takes: the server holds the first 2.5 s before any audio
# goes out anyway, so the first OPENING_S are transcribed within that and must match the
# start of the line. A spoken direction or a skipped opening span scored 0.53 or less
# (most 0.43), good openings mostly 0.88-1.0 (a misheard whispered "okay" can cost a
# needless restart, about a second).
OPENING = os.environ.get("DRAMA3_OPENING_CHECK", "1").strip().lower() not in {"0", "false", "no", "off"}
OPENING_S = 2.0
OPENING_PASS = 0.75
MIN_LETTERS = 8  # shorter lines ("Yes.") are too short to judge


def log(message: str) -> None:
    print(f"[words] {message}", flush=True)


def letters(text: str) -> str:
    text = infill.spell_numbers(unicodedata.normalize("NFKC", text).lower()).replace("-", " ")
    return "".join(infill.norm(word) for word in text.split())


def transcriber():
    """Whisper for the checks: a copy on the GPU once loaded (a 2 s opening in ~0.08 s
    instead of ~2 s on the CPU, which would delay every streamed take), else the
    server's CPU Whisper once it is ready, else None (no check)."""
    if GPU["model"] is not None:
        return GPU["model"]
    model = getattr(sys.modules.get("server"), "transcriber", None)
    return model if model is not None and model.state == "ready" else None


GPU = {"model": None}


def load_gpu_whisper() -> None:
    """In the background at startup: the server's Whisper model, on the GPU (~1.5 GB of
    the VRAM reserve). Without CUDA Whisper (e.g. missing CUDA libraries) the checks use
    the CPU one."""
    def load():
        try:
            import ctranslate2
            import torch
            from faster_whisper import WhisperModel

            server = sys.modules.get("server")
            if not torch.cuda.is_available() or ctranslate2.get_cuda_device_count() < 1 or server is None:
                return
            model = WhisperModel(server.WHISPER_MODEL, device="cuda", compute_type="float16",
                                 download_root=server.WHISPER_DIR)
            GPU["model"] = types.SimpleNamespace(model=model, lock=threading.Lock(), state="ready")
            log("word checks use Whisper on the GPU")
        except Exception as error:  # noqa: BLE001  (the CPU Whisper still works)
            log(f"Whisper on the GPU unavailable ({type(error).__name__}: {error}); checks use the CPU one")

    threading.Thread(target=load, name="whisper-gpu", daemon=True).start()


class Check:
    def __init__(self, script: str, model) -> None:
        self.script = script
        self.said = "".join(letters(token) for token in infill.tokens(script))
        self.model = model

    def hear(self, pcm: bytes) -> list[tuple[str, float, float]]:
        return infill.transcribe_words(self.model, pcm)

    def opening(self, heard: list[tuple[str, float, float]]) -> float | None:
        """The share of what was heard that matches the start of the line, or None when
        too little was heard to judge (a whispered or silent opening)."""
        got = "".join(letters(text) for text, _, _ in heard)
        if len(got) < 4:
            return None
        blocks = difflib.SequenceMatcher(None, got, self.said[:len(got) + 6], autojunk=False).get_matching_blocks()
        return sum(block.size for block in blocks) / len(got)

    def score(self, heard: list[tuple[str, float, float]]) -> float:
        got = "".join(letters(text) for text, _, _ in heard)
        return difflib.SequenceMatcher(None, self.said, got, autojunk=False).ratio()


def checker(script: str, max_frames: int | None = None) -> Check | None:
    """A check for this script's take, or None (off, Whisper not ready, too short, or the
    startup warm-up's 20-frame take)."""
    if RETRIES < 1 or (max_frames is not None and max_frames <= 40):
        return None
    model = transcriber()
    if model is None:
        return None
    check = Check(script, model)
    return check if len(check.said) >= MIN_LETTERS else None


ENDING = os.environ.get("DRAMA3_ENDING_CHECK", "1").strip().lower() not in {"0", "false", "no", "off"}


def missing_tail(check: Check, heard: list[tuple[str, float, float]], stalled: bool) -> str | None:
    """The line's last words when a streamed take ended without them, or None. At least
    two missing words (a faded or whispered last word may just be unheard), or one after
    a take the stall guard ended."""
    words = infill.align(check.script, heard)
    said = [i for i, word in enumerate(words) if word.heard]
    if not said:
        return None
    tail = words[said[-1] + 1:]
    if len(tail) >= 2 or (stalled and tail):
        text = " ".join(word.text for word in tail)
        return text if len(letters(text)) >= 4 else None
    return None


# drawled is adapted from latent-variable/shipshow's bin/hearback.py, Copyright (c) 2026
# Lino Valdovinos, under the MIT License
# (https://github.com/latent-variable/shipshow/blob/main/LICENSE).
#
# A take that drags inside a sentence: more than MAX_WORD_GAP s from one word's start to
# the next word's, in one sentence (latent-variable/shipshow #90, its DESIGN D65). Drama 3
# read "Like animals." with 3 s between the two words there; the word check passed it at
# 0.977, and hiss filled the gap, so no silence check saw it. In 1,204 word pairs of their
# good takes the longest such gap was 1.9 s (a year said out loud). A pause between
# sentences is a beat and is never judged; here neither is one the script asks for: a
# [cue] or <tag> between the words, an ellipsis or a dash after the first, or a word given
# as phonemes (not in the words Whisper is matched to). 0 = off.
MAX_WORD_GAP = float(os.environ.get("DRAMA3_MAX_WORD_GAP", "2.5"))
SENTENCE_END = re.compile(r"[.!?。！？]+[\"'”’)\]]*$")
PAUSE_AFTER = re.compile(r"(\.\.\.|…|—|–|-)[\"'”’)]*$")
WRITTEN_PAUSE = re.compile(r"\[[^\[\]]*\]|<[^<>]*>|<\|phoneme_start\|>.*?<\|phoneme_end\|>", re.S)


def drawled(script: str, heard: list[tuple[str, float, float]]) -> str | None:
    """'<gap> s between "<a>" and "<b>"' for the longest gap over MAX_WORD_GAP between two
    words of one sentence, or None."""
    if MAX_WORD_GAP <= 0 or not heard:
        return None
    words, paused = [], []  # the script's spoken words; a pause written before each
    for number, piece in enumerate(WRITTEN_PAUSE.split(infill.PHONEME.sub("[ ]", script))):
        for index, token in enumerate(infill.tokens(piece)):
            words.append(token)
            paused.append(index == 0 and number > 0)
    timed = infill.align(" ".join(words), heard)
    if len(timed) != len(words):
        return None
    worst = None
    for (a, b), written in zip(zip(timed, timed[1:]), paused[1:]):
        if written or a.start is None or b.start is None or SENTENCE_END.search(a.text) or PAUSE_AFTER.search(a.text):
            continue
        gap = b.start - a.start
        if gap > MAX_WORD_GAP and (worst is None or gap > worst[0]):
            worst = (gap, a.text, b.text)
    return None if worst is None else f'{worst[0]:.1f} s between "{worst[1]}" and "{worst[2]}"'


def heard_text(heard: list[tuple[str, float, float]]) -> str:
    return " ".join(text.strip() for text, _, _ in heard)
