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

Short single-chunk lines take the Space's own code path unchanged.
"""

from __future__ import annotations

import collections
import os
import sys
import re
import time
from dataclasses import dataclass

import torch

import chunking
import infill
import subtitles

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
}
# Drama 3's streaming-input format (technical report, Figure 10): each chunk's turn also
# shows the chunk that follows, after <|endofturn|>, so the model voices the current
# chunk knowing what comes next (an empty next chunk marks the last one).
END_OF_TURN = "<|endofturn|>"
STREAM_INSTRUCTION = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"
KEEPALIVE_S = 2.0  # empty packets while a chunk is held back (the server's idle limit is 15 s)


def log(message: str) -> None:
    print(f"[long] {message}", flush=True)


def spoken(text: str) -> str:
    """Words only: no [cues] or <tags>, white space collapsed (for comparisons)."""
    return " ".join(re.sub(r"\[[^\[\]]*\]|<[^<>]*>", " ", text).split())


RETAKE = re.compile(r"Direction: (?P<note>[^\n]*)\nLine: (?P<line>.*)", re.S)


def parse_turn(text: str) -> tuple[str | None, str | None, str]:
    """(note, direction, words) of a turn.

    note: a Retake the Space's retake_text wrote (every sentence opens with [note]);
    its words lose the acting cues, which the note replaces. direction: a
    "Direction: …\nLine: …" the user wrote; its Line keeps its own cues, and every
    chunk of it is sent under the same Direction."""
    match = RETAKE.fullmatch(text)
    if not match:
        return None, None, text
    note, line = match.group("note"), match.group("line")
    if line.startswith(f"[{note}] "):
        return note, None, line.replace(f"[{note}] ", "")
    return None, note, line


def parse_retake(text: str) -> tuple[str | None, str]:
    """(note, words) of a turn (see parse_turn); a user's Direction is not a note."""
    note, _, words = parse_turn(text)
    return note, words


def directed_turn(direction: str | None, text: str) -> str:
    """One chunk of a user-directed Line, under the take's Direction."""
    return f"Direction: {direction}\nLine: {text}" if direction else text


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


def join_pause(text: str, pcm: bytes, rate: int = 44100, tone: list[bytes] | None = None) -> bytes:
    """The pause to put after a chunk that ends on text, whose audio is pcm: room tone
    of the chunks in `tone` (see room_tone), or silence."""
    end = text.rstrip().rstrip("\"'”’)]»」』").rstrip()
    kind = "sentence" if end[-1:] in SENTENCE_END else "clause" if end[-1:] in CLAUSE_END else "word"
    missing = JOIN_PAUSE_S[kind] - edge_silence(pcm, rate)[1]
    return room_tone(tone or [pcm], int(max(0.0, missing) * rate))


# Room tone: a cloned voice keeps its recording's background (hiss, hum, room), and a
# pause of digital silence between two chunks makes that background drop out and come
# back. The pause is filled instead with noise of the background's spectrum and level,
# measured on the quietest stretches of the take's audio. Takes whose background is
# below ROOM_TONE_MIN_RMS keep silent pauses.
ROOM_TONE_FRAME = 4096  # samples per analysis frame (93 ms: 11 Hz bins, so a mains hum stays near its pitch)
ROOM_TONE_CEILING = 0.02  # RMS above which a frame is never background (-34 dBFS)
ROOM_TONE_MIN_RMS = 0.0005  # quieter backgrounds are left silent (-66 dBFS)
ROOM_TONE_MIN_FRAMES = 4  # background needed to measure it (0.37 s)


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
    measured = background(pcms)
    if measured is None:
        return bytes(2 * samples)
    spectrum, level = measured
    rng = np.random.default_rng(zlib.crc32(pcms[-1][-8192:]) + samples)
    white = np.fft.rfft(rng.standard_normal(samples))
    bins = np.linspace(0, len(spectrum) - 1, len(white))
    shaped = np.fft.irfft(white * np.interp(bins, np.arange(len(spectrum)), spectrum), n=samples)
    shaped *= level / max(float(np.sqrt((shaped ** 2).mean())), 1e-9)
    return (np.clip(shaped, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


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

        def speech(self, body, *, deadline=None):
            try:
                request = de.parse_request(body)
                plan = self.plan(request)
            except de.RequestError as error:
                yield {"type": "error", "message": f"Drama 3 request: {error}"}
                return
            if plan is None:  # one chunk: the Space's own path, unchanged
                pcm = bytearray()
                for event in super().speech(body, deadline=deadline):
                    if event["type"] == "speech.audio.delta":
                        pcm += event["pcm"]
                    elif event["type"] == "speech.audio.done" and event.get("frames"):
                        # kept (codes + audio) so an edit of this line can be infilled
                        _, direction, words = parse_turn(request.text)
                        remember(request.text, LongTake(line=words, direction=direction, chunks=[
                            ChunkAudio(words, request.text, self.last_codes(event["frames"]), bytes(pcm))]))
                    yield event
                return
            try:
                yield from self.long_speech(request, *plan)
            except de.RequestError as error:
                yield {"type": "error", "message": f"Drama 3 request: {error}"}

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

        def long_speech(self, request, note, words, planned, previous):
            started = time.perf_counter()
            voices = self.voices(request, previous)
            # Earlier short takes the server sends (a fresh long line retaken from a short take)
            extra = []
            if previous is None:
                for turn, reference in zip(request.history, request.references[request.voice_count:]):
                    extra.append((turn, self.encode(reference)))
            regenerate = sum(chunk.changed for chunk in planned)
            log(f"{len(planned)} chunks, {regenerate} to generate"
                + (f", {len(planned) - regenerate} kept" if previous is not None else "")
                + (f", note “{note}”" if note else ""))
            direction = parse_turn(request.text)[1]
            records: list[ChunkAudio] = []
            frames_total, finish, first_audio = 0, "stop", None
            timed = subtitles.is_timed(words)
            position = 0  # bytes of audio sent so far (the subtitle timeline)

            def plain(text: str) -> str:  # a chunk's words without its subtitle timing
                return subtitles.cue(text).text if timed else text

            for index, chunk in enumerate(planned):
                cue = subtitles.cue(chunk.text) if timed else None
                if cue is not None and cue.start is not None:
                    # a subtitle cue starts at its time (or at once, if the one before ran over)
                    pause = room_tone([r.pcm for r in records[-3:]], max(0, int(cue.start * 44100) - position // 2))
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
                    for offset in range(0, len(record.pcm), PACKET_BYTES):
                        yield {"type": "speech.audio.delta", "pcm": record.pcm[offset:offset + PACKET_BYTES]}
                elif (record := (yield from self.try_infill(previous, chunk, note, request))) is not None:
                    for offset in range(0, len(record.pcm), PACKET_BYTES):
                        yield {"type": "speech.audio.delta", "pcm": record.pcm[offset:offset + PACKET_BYTES]}
                else:
                    text = subtitles.spoken_turn(cue) if cue is not None else chunk.text
                    turn = retake_turn(note, spoken(text) or text) if note else directed_turn(direction, text)
                    if note and cue is not None and cue.frames:
                        turn += f" [audio tokens: {cue.frames}]"
                    streaming = SETTINGS["lookahead"]
                    if streaming:
                        following = plain(planned[index + 1].text) if index + 1 < len(planned) else ""
                        prompt_turn = f"{turn}{END_OF_TURN}{following}"
                    else:
                        prompt_turn = turn
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
                    live = not records
                    for attempt in range(1 if live else 2):
                        frames, reason, pcm = yield from self.generate(rows, request, turn, live=live)
                        if reason != "stall" or live:
                            break
                        log(f"chunk {index + 1}: stalled after {len(pcm) / 88200:.1f}s, making it again")
                    if not live:
                        for offset in range(0, len(pcm), PACKET_BYTES):
                            yield {"type": "speech.audio.delta", "pcm": pcm[offset:offset + PACKET_BYTES]}
                    if reason in ("length", "stall"):
                        finish = reason
                    if not frames:
                        log(f"chunk {index + 1}: no audio (end of speech at once)")
                        continue
                    record = ChunkAudio(chunk.text, prompt_turn, self.last_codes(frames), pcm)
                    lead, tail = edge_silence(pcm)
                    log(f"chunk {index + 1}: {len(pcm) / 88200:.2f}s, silence {lead:.2f}s at the start, "
                        f"{tail:.2f}s at the end, finish {reason}, ends with {chunk.text[-12:]!r}")
                position += len(record.pcm)
                frames_total += record.codes.shape[1]
                first_audio = first_audio or round((time.perf_counter() - started) * 1000)
                records.append(record)
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

    server.app.add_api_route("/api/takes/{take_id}.srt", take_subtitles, methods=["GET"],
                             dependencies=[server.Depends(server.identify)])
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
