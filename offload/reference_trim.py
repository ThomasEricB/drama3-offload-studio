"""Reference clips: keep as much of the 20 s limit as possible.

The Space trims a clip longer than REF_MAX_SECONDS at the single quietest 150 ms of
its last 40 % (12-20 s), so the cut lands between words. But a clean pause at 13 s
beats an ordinary breath at 19 s, and the voice reference ends up ~13-15 s long.

This keeps the Space's rule (cut inside a pause in that window, short fade, then peak
normalisation) and changes only which pause: the *latest* one that is still a real
gap, i.e. within 6 dB of the quietest window or 15 dB below the clip's typical speech
level.
"""

from __future__ import annotations

import numpy as np

WINDOW_S = 0.15  # as the Space: 150 ms analysis windows, 10 ms apart
HOP_S = 0.01
SEARCH_FROM = 0.6  # the last 40 % of the limit
NEAR_QUIETEST = 2.0  # RMS within 6 dB of the quietest window
BELOW_SPEECH = 0.178  # or 15 dB below the clip's median window RMS


def tidy_reference(wav: np.ndarray, sample_rate: int, *, rate: int, max_seconds: float) -> tuple[np.ndarray, bool]:
    """The Space's tidy_reference, cutting at the latest pause before the limit."""
    import librosa

    if sample_rate != rate:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=rate)
    wav = np.nan_to_num(wav.astype(np.float32))
    trimmed, _ = librosa.effects.trim(wav, top_db=40, frame_length=1024, hop_length=256)
    if trimmed.size:
        wav = trimmed
    cut = False
    limit = int(max_seconds * rate)
    if wav.size > limit:
        start = int(SEARCH_FROM * limit)
        window, hop = int(WINDOW_S * rate), int(HOP_S * rate)
        frames = np.lib.stride_tricks.sliding_window_view(wav[start:limit], window)[::hop]
        rms = np.sqrt((frames**2).mean(axis=1))
        speech = np.sqrt((np.lib.stride_tricks.sliding_window_view(wav[:limit], window)[::hop] ** 2).mean(axis=1))
        threshold = max(NEAR_QUIETEST * float(rms.min()), BELOW_SPEECH * float(np.median(speech)))
        latest = int(np.flatnonzero(rms <= threshold)[-1])
        wav = wav[: start + latest * hop + window // 2].copy()
        fade = min(int(0.03 * rate), wav.size)
        wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        cut = True
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 0:
        wav = wav * min(10.0, 0.9 / peak)
    return wav.astype(np.float32), cut


def install(server) -> None:
    """Replace server.tidy_reference (looked up when a clip is registered). The limit is
    read per clip, so changing server.REF_MAX_SECONDS (20 s / 60 s) applies at once."""
    server.tidy_reference = lambda wav, sample_rate: tidy_reference(
        wav, sample_rate, rate=server.SAMPLE_RATE, max_seconds=server.REF_MAX_SECONDS)
