"""Long scripts: split into chunks, and map an edit onto the chunks it touches.

The splitting follows the design Fish Audio published for Drama 3 in fish-speech
(fish_omni/chunking.py): chunks of about 300 UTF-8 bytes, cut at sentence ends, then
clause marks, then spaces, never inside [tags] or <tag>...</tag> spans. This is an
independent implementation of that design, not a copy of the code.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

CHUNK_BYTES = 300  # a threshold, not a hard cap: a protected span longer than this stays whole
CHUNK_MAX_FRAMES = 600  # frames per chunk, about 28 s at 21.5 frames/s (as fish-speech)
WHITE = " \t\r\n 　"
SENTENCE_ENDS = ".!?;。！？；…\n"
CLAUSE_MARKS = ",:，：、"
PROTECTED = re.compile(r"<\|phoneme_start\|>.*?<\|phoneme_end\|>|<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>"
                       r"|\[[^\[\]]*\]", re.S)


def size(text: str) -> int:
    return len(text.encode("utf-8"))


def _pieces(text: str) -> list[tuple[str, bool]]:
    """(text, protected) runs: tags and tagged spans are protected from cuts."""
    out, last = [], 0
    for match in PROTECTED.finditer(text):
        if match.start() > last:
            out.append((text[last:match.start()], False))
        out.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        out.append((text[last:], False))
    return out


def _cut_after(pieces, marks: str, longer_than: int):
    """Cut free runs longer than longer_than bytes after each mark (kept with its run)."""
    out = []
    for text, protected in pieces:
        if protected or size(text) <= longer_than:
            out.append((text, protected))
            continue
        start = 0
        for index, char in enumerate(text):
            decimal = char == "." and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()
            if char in marks and not decimal:
                out.append((text[start:index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def _cut_by_length(pieces, limit: int):
    out = []
    for text, protected in pieces:
        if protected or size(text) <= limit:
            out.append((text, protected))
            continue
        current = ""
        for char in text:
            if current and size(current + char) > limit:
                out.append((current, False))
                current = ""
            current += char
        out.append((current, False))
    return out


def split(text: str, limit: int = CHUNK_BYTES) -> list[str]:
    """Chunks of about `limit` bytes, packed sentence by sentence."""
    if not text.strip(WHITE):
        return []
    if size(text) <= limit:
        return [text.strip(WHITE)]
    pieces = _cut_after(_pieces(text), SENTENCE_ENDS, 0)
    pieces = _cut_after(pieces, CLAUSE_MARKS, limit)
    pieces = _cut_after(pieces, " ", limit)
    pieces = _cut_by_length(pieces, limit)
    chunks, current = [], []
    for piece, _ in pieces:
        if current and size("".join(current) + piece) > limit:
            # Cues and tags with no words direct what follows them: they open the next
            # chunk. A chunk of cues alone would have the model improvise (or read ahead)
            cut = len(current)
            while cut and _wordless(current[cut - 1]):
                cut -= 1
            if cut:
                chunks.append("".join(current[:cut]))
                current = current[cut:]
        current.append(piece)
    chunks.append("".join(current))
    return [chunk.strip(WHITE) for chunk in chunks if chunk.strip(WHITE)]


def _wordless(text: str) -> bool:
    return not re.sub(r"\[[^\[\]]*\]|<[^<>]*>", "", text).strip(WHITE)


# ─── Edits of a chunked take ──────────────────────────────────────────────────


@dataclass
class PlannedChunk:
    text: str
    old: list[int] = field(default_factory=list)  # indices of the old chunks it replaces
    changed: bool = True


def _spans(line: str, chunks: list[str]) -> list[tuple[int, int]]:
    """Where each (stripped) chunk sits in the line it was cut from."""
    spans, position = [], 0
    for chunk in chunks:
        start = line.find(chunk, position)
        if start < 0:
            raise ValueError("chunk not found in its line")
        spans.append((start, start + len(chunk)))
        position = start + len(chunk)
    return spans


def plan_edit(old_line: str, old_chunks: list[str], new_line: str) -> list[PlannedChunk]:
    """New chunks for an edited line, regenerating as few old chunks as possible.

    Chunks still present verbatim at the start and the end of the new line are kept.
    In the stretch between, old chunk boundaries are kept wherever the text around
    them is unchanged (character diff), so an edit inside one chunk regenerates only
    that chunk; a boundary inside an edited stretch is dropped (its chunks merge)."""
    try:
        spans = _spans(old_line, old_chunks)
    except ValueError:
        return [PlannedChunk(text) for text in split(new_line)]
    head, position = 0, 0
    while head < len(old_chunks):
        at = len(new_line) - len(new_line[position:].lstrip(WHITE))
        if not new_line.startswith(old_chunks[head], at):
            break
        position = at + len(old_chunks[head])
        head += 1
    tail, limit = 0, len(new_line)
    while tail < len(old_chunks) - head:
        chunk = old_chunks[len(old_chunks) - 1 - tail]
        end = len(new_line[:limit].rstrip(WHITE))
        if end - len(chunk) < position or not new_line.startswith(chunk, end - len(chunk)):
            break
        limit = end - len(chunk)
        tail += 1
    kept_head = [PlannedChunk(old_chunks[i], [i], changed=False) for i in range(head)]
    kept_tail = [PlannedChunk(old_chunks[i], [i], changed=False) for i in range(len(old_chunks) - tail, len(old_chunks))]
    middle_new = new_line[position:limit]
    middle = list(range(head, len(old_chunks) - tail))
    if not middle_new.strip(WHITE):
        return kept_head + kept_tail  # pure deletion
    if not middle:  # pure insertion
        return kept_head + [PlannedChunk(text) for text in split(middle_new)] + kept_tail
    middle_old = old_line[spans[middle[0]][0]:spans[middle[-1]][1]]
    offset = spans[middle[0]][0]
    middle_spans = [(a - offset, b - offset) for a, b in spans[middle[0]:middle[-1] + 1]]
    middle_chunks = [old_chunks[i] for i in middle]
    opcodes = difflib.SequenceMatcher(None, middle_old, middle_new, autojunk=False).get_opcodes()

    def mapped(at: int) -> int | None:
        for tag, i1, i2, j1, _ in opcodes:
            if tag == "equal" and i1 <= at <= i2:
                return j1 + at - i1
        return None

    groups, starts = [[0]], [0]
    for index in range(1, len(middle_chunks)):
        # Any offset in the white space between two chunks is the same boundary.
        gap = range(middle_spans[index][0], middle_spans[index - 1][1] - 1, -1)
        boundary = next((b for b in map(mapped, gap) if b is not None), None)
        if boundary is None or boundary <= starts[-1]:
            groups[-1].append(index)
        else:
            groups.append([index])
            starts.append(boundary)
    starts.append(len(middle_new))

    planned: list[PlannedChunk] = []
    for group, start, end in zip(groups, starts, starts[1:]):
        text = middle_new[start:end].strip(WHITE)
        old = [middle[i] for i in group]
        if not text:
            continue  # the chunk was deleted
        if len(group) == 1 and text == middle_chunks[group[0]]:
            planned.append(PlannedChunk(text, old, changed=False))
            continue
        pieces = split(text) if size(text) > 2 * CHUNK_BYTES else [text]
        for number, piece in enumerate(pieces):
            planned.append(PlannedChunk(piece, old if number == 0 else [], changed=True))
    return kept_head + planned + kept_tail


def describe(planned: list[PlannedChunk], old_count: int, *, note: str | None) -> str:
    """One line for the GUI: which chunks the next Retake/Edit regenerates."""
    total = len(planned)
    if note:
        return f"**Retake** redoes all {total} chunks with the note, each guided by its old version."
    changed = [i for i, chunk in enumerate(planned) if chunk.changed]
    used = {i for chunk in planned for i in chunk.old}
    removed = old_count - len(used)
    dropped = f" {removed} old chunk{'s are' if removed > 1 else ' is'} cut out." if removed > 0 else ""
    if not changed:
        if removed > 0:
            return f"**Edit** regenerates nothing:{dropped} The remaining {total} chunks keep their audio."
        return "**Edit:** the script matches this take; change some words first (nothing would be regenerated)."
    if len(changed) == total:
        return f"**Edit** regenerates all {total} chunks (the change touches every chunk).{dropped}"
    first = planned[changed[0]].text
    preview = first if len(first) <= 90 else first[:87] + "…"
    kept = total - len(changed)
    if len(changed) == 1:
        return (f"**Edit** regenerates only chunk {changed[0] + 1} of {total}: “{preview}”. "
                f"The other {kept} chunk{'s keep their' if kept > 1 else ' keeps its'} audio.{dropped}")
    numbers = ", ".join(str(i + 1) for i in changed)
    return f"**Edit** regenerates chunks {numbers} of {total}; the other {kept} keep their audio.{dropped}"
