"""Edits by infilling: regenerate only the words that changed.

Drama 3 edits a recording by infilling (technical report, section 5.5 and Figure 14):
the source audio is aligned to its transcript, the edit window runs from the first to
the last word that differs (a deletion also covers the unchanged word before it, or
after it at the start), and the model regenerates the window from a prompt holding the
full new text and the audio before and after it. The whole token sequence, with the
new tokens inside the window, is then decoded to audio, so the seams are the codec's
own. The report aligns with Qwen3-ForcedAligner; this uses the word timestamps of the
Whisper model the Studio already runs (on the CPU) for reference transcripts.

Pure functions here; long_takes.py runs the generation.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

FPS = 44100 / 2048  # codec frames per second
INSTRUCTION = ("fill in the missing part of the speech; the suffix follows <|fim_suffix|> "
               "and the prefix follows <|fim_prefix|>")
MAX_WINDOW_SHARE = 0.75  # larger edits regenerate the whole chunk
VERIFY_MIN = 0.5  # share of the new window's letters Whisper must hear afterwards (a wrong word ~0.2, a name spelled differently ~0.6)
# Unchanged words an insertion's window also covers: "both" (the words on both sides) or
# "before" (the word before; the one after at the start). On 16 edits x 2 voices, "both"
# avoided a repeated phrase that "before" made (word match 0.994 vs 0.985).
INSERT_CONTEXT = "both"
WHOLE_SLACK = 0.05  # the edited chunk may match its script this much worse than the old one did
# A word given as phonemes (Arpabet, pinyin, romaji or IPA): its spelling is not in the
# script, so the checks and alignments here leave it out
PHONEME = re.compile(r"<\|phoneme_start\|>([^<>|\n]+?)<\|phoneme_end\|>")


def log(message: str) -> None:
    print(f"[infill] {message}", flush=True)


def _cjk(char: str) -> bool:
    return unicodedata.category(char) == "Lo" and ord(char) >= 0x2E80


def tokens(text: str) -> list[str]:
    """Spoken words: [cues], <tags> and words given as phonemes removed; CJK characters
    are words of their own."""
    text = re.sub(r"\[[^\[\]]*\]|<[^<>]*>", " ", PHONEME.sub(" ", text))
    out: list[str] = []
    for part in text.split():
        word = ""
        for char in part:
            if _cjk(char):
                if word:
                    out.append(word)
                    word = ""
                out.append(char)
            elif unicodedata.category(char) in ("Po", "Pe", "Pf") and not word and out:
                out[-1] += char  # punctuation closes the word before it
            else:
                word += char
        if word:
            out.append(word)
    merged: list[str] = []
    for word in out:  # punctuation on its own (CJK) belongs to the word before
        if merged and not norm(word):
            merged[-1] += word
        else:
            merged.append(word)
    return merged


def norm(word: str) -> str:
    """Letters and digits of a word, lowercased, numbers spelled out (see spell_numbers)."""
    return "".join(c for c in spell_numbers(unicodedata.normalize("NFKC", word).lower()) if c.isalnum())


# Whisper writes numbers as digits ("7", "$5", "50%", "1999") where a script spells them
# out, and the other way round; both sides are compared spelled out, in English.
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}
CURRENCIES = {"$": "dollars", "£": "pounds", "€": "euros"}
NUMBER = re.compile(r"(?P<cur>[$£€])?(?P<int>\d+(?:,\d{3})*)(?P<dec>\.\d+)?(?P<ord>st|nd|rd|th)?(?P<pct>%)?")


def _cardinal(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _cardinal(n % 100))
    for scale, name in SCALES:
        if n >= scale:
            rest = n % scale
            return f"{_cardinal(n // scale)} {name}" + ("" if rest == 0 else " " + _cardinal(rest))
    raise AssertionError(n)


def number_words(digits: str) -> str:
    """English words for a whole number; years are read as years ("1999": nineteen ninety nine)."""
    n = int(digits.replace(",", ""))
    if n >= 10**12:
        return " ".join(ONES[int(d)] for d in digits if d.isdigit())
    if "," not in digits and len(digits) == 4 and (1100 <= n <= 1999 or 2010 <= n <= 2099) and n % 100:
        return f"{_cardinal(n // 100)} {_cardinal(n % 100) if n % 100 >= 10 else 'oh ' + ONES[n % 100]}"
    if "," not in digits and len(digits) == 4 and 1100 <= n <= 1999:
        return f"{_cardinal(n // 100)} hundred"
    return _cardinal(n)


def _ordinal(words: str) -> str:
    *head, last = words.split()
    if last in ORDINALS:
        last = ORDINALS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join([*head, last])


def spell_numbers(text: str) -> str:
    """text with each number in English words: "7th" seventh, "3.5" three point five,
    "50%" fifty percent, "$5" five dollars."""
    def spell(match: re.Match) -> str:
        words = number_words(match["int"])
        if match["dec"]:
            words += " point " + " ".join(ONES[int(d)] for d in match["dec"][1:])
        elif match["ord"]:
            words = _ordinal(words)
        if match["pct"]:
            words += " percent"
        if match["cur"]:
            words += " " + CURRENCIES[match["cur"]]
        return f" {words} "
    return NUMBER.sub(spell, text) if any(c.isdigit() for c in text) else text


@dataclass
class Word:
    text: str
    start: float | None = None  # seconds
    end: float | None = None
    heard: int = 0  # letters of the word matched in what Whisper heard


def align(script: str, whisper_words: list[tuple[str, float, float]]) -> list[Word]:
    """Times for each spoken word of script, from Whisper's word timestamps, matched
    letter by letter (Whisper may split, join or misspell words). Unmatched: None."""
    words = [Word(t) for t in tokens(script)]
    script_chars, owner = [], []
    for index, word in enumerate(words):
        for char in norm(word.text):
            script_chars.append(char)
            owner.append(index)
    heard_chars, times = [], []
    for text, start, end in whisper_words:
        for char in norm(text):
            heard_chars.append(char)
            times.append((start, end))
    matcher = difflib.SequenceMatcher(None, script_chars, heard_chars, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            word = words[owner[block.a + k]]
            start, end = times[block.b + k]
            word.start = start if word.start is None else min(word.start, start)
            word.end = end if word.end is None else max(word.end, end)
            word.heard += 1
    return words


@dataclass
class Window:
    start: int  # frames of the old audio kept before the window
    end: int  # frames from here on are kept after it
    old_text: str
    new_text: str  # the new words inside the window (what is generated)
    frames_hint: int  # [audio tokens: N]
    new_words: tuple[int, int]  # their indices in tokens(new script)


def window(old_script: str, new_script: str, words: list[Word], total_frames: int) -> Window | None:
    """The edit window, or None when the edit cannot be infilled (nothing spoken
    changed, the alignment misses a boundary word, or the edit is most of the chunk)."""
    old = [w.text for w in words]
    new = tokens(new_script)
    if not old or not new:
        return None
    matcher = difflib.SequenceMatcher(None, [norm(w) for w in old], [norm(w) for w in new], autojunk=False)
    changes = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not changes:
        return None
    i1, j1, i2, j2 = changes[0][1], changes[0][3], changes[-1][2], changes[-1][4]
    if i1 == i2 or j1 == j2:
        # Insertion or deletion: cover an unchanged neighbour, so the window holds audio
        # to replace and words to say (the word before; at the start, the one after).
        both = INSERT_CONTEXT == "both" and i1 == i2
        if i1 > 0:
            i1, j1 = i1 - 1, j1 - 1
            if both and i2 < len(old):
                i2, j2 = i2 + 1, j2 + 1
        elif i2 < len(old):
            i2, j2 = i2 + 1, j2 + 1
        else:
            return None
    if (i2 - i1) > MAX_WINDOW_SHARE * len(old):
        return None

    def boundary(index: int) -> float | None:
        """Seconds between word index-1 and word index (0 / the end at the edges)."""
        if index == 0:
            return 0.0
        if index == len(words):
            return total_frames / FPS
        before, after = words[index - 1].end, words[index].start
        if before is None or after is None:
            return None
        return (before + after) / 2

    start_s, end_s = boundary(i1), boundary(i2)
    if start_s is None or end_s is None or end_s <= start_s:
        return None
    start, end = int(round(start_s * FPS)), min(total_frames, int(round(end_s * FPS)))
    if end - start > MAX_WINDOW_SHARE * total_frames:
        return None
    old_text, new_text = " ".join(old[i1:i2]), " ".join(new[j1:j2])
    letters_old, letters_new = len(norm(old_text)), len(norm(new_text))
    hint = max(4, int(round((end - start) * (letters_new + 2) / (letters_old + 2))))
    return Window(start, end, old_text, new_text, hint, (j1, j2))


def segments(new_script: str, window_: Window, codes, with_speaker_tag) -> list:
    """The infilling prompt (report Figure 14): suffix audio before prefix audio, so the
    prefix sits next to the generated span."""
    suffix, prefix = codes[:, window_.end:], codes[:, :window_.start]
    return [
        "<|im_start|>system\n", INSTRUCTION, "<|im_end|>\n",
        "<|im_start|>user\n", f"{with_speaker_tag(new_script)} [audio tokens: {window_.frames_hint}]",
        "<|fim_suffix|>", *([suffix] if suffix.shape[1] else []),  # no audio after / before the window
        "<|fim_prefix|>", *([prefix] if prefix.shape[1] else []),
        "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>",
    ]


def heard_share(new_script: str, window_: Window, whisper_words: list[tuple[str, float, float]]) -> float:
    """Share of the window's new letters that Whisper heard in their place: the whole
    new script is aligned to the new audio, letter by letter, and the window's words
    are looked up."""
    words = align(new_script, whisper_words)[window_.new_words[0]:window_.new_words[1]]
    total = sum(len(norm(w.text)) for w in words)
    if not total:
        return 1.0
    return sum(min(w.heard, len(norm(w.text))) for w in words) / total


def match_ratio(script: str, whisper_words: list[tuple[str, float, float]]) -> float:
    """How well what Whisper heard matches the script, word for word (missing and extra
    words both count). Numbers count as their spelled-out words ("$5": five dollars)."""
    def words(text: str) -> list[str]:
        spelled = spell_numbers(unicodedata.normalize("NFKC", text).lower()).replace("-", " ")
        return [w for w in map(norm, spelled.split()) if w]

    heard = [w for text, _, _ in whisper_words for w in words(text)]
    said = [w for token in tokens(script) for w in words(token)]
    return difflib.SequenceMatcher(None, said, heard, autojunk=False).ratio()


def transcribe_words(transcriber, pcm: bytes) -> list[tuple[str, float, float]]:
    """Whisper word timestamps of 16-bit 44.1 kHz PCM (the server's CPU Whisper)."""
    import librosa

    audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    audio = librosa.resample(audio, orig_sr=44100, target_sr=16000)
    with transcriber.lock:
        segments_, _ = transcriber.model.transcribe(
            audio, beam_size=1, temperature=0.0, condition_on_previous_text=False, word_timestamps=True)
        return [(w.word, w.start, w.end) for s in segments_ for w in (s.words or [])]
