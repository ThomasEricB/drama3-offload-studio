"""The register guard: a take with a phrase that opens far above the others is made again.

Drama 3 sometimes voices one phrase of a take in a much higher register than the rest, as
if stitched in from another take. The word check cannot hear it (the words are right).
Adapted from drama-tensorfold's engine patch 0011 (plotarmordev/drama-tensorfold,
docs/speech-redraws.md#register): pitch every 10 ms by normalized autocorrelation with
Praat's To Pitch (ac) defaults (75-600 Hz, voicing threshold 0.45, silence threshold 0.03
of the take's peak, octave cost 0.01); unvoiced runs of 0.25 s part phrases, a phrase
voiced for less than 0.25 s is skipped, and a phrase's opening is the median pitch of its
first 0.25 s voiced. A phrase after the first trips when its opening is JUMP_ST or more
above the median opening of the other phrases and HIGH_SHARE of its voiced frames lie
above the others' median by HIGH_ST or more (drama-tensorfold takes an absolute Hz for
that, calibrated per voice; a step above the take's own phrases works for any voice). The
take's jump is the largest such excess.

Experimental (the Studio's "Enable Experimental Changes"), and in Quality Mode only: a
take or chunk that passes the word check and trips is made once more, and the remake is
kept if it passes as well with a smaller jump. A long script's chunks are measured with
the phrase openings of the chunks before them. DRAMA3_REGISTER_ST sets JUMP_ST; 0 turns
the guard off.
"""

from __future__ import annotations

import os

import numpy as np

JUMP_ST = float(os.environ.get("DRAMA3_REGISTER_ST", 6.0))
HIGH_ST = float(os.environ.get("DRAMA3_REGISTER_HIGH_ST", 3.0))
HIGH_SHARE = float(os.environ.get("DRAMA3_REGISTER_HIGH_SHARE", 0.5))
RATE = 16000
HOP = RATE // 100  # 10 ms
WINDOW = 3 * RATE // 75  # three periods of the lowest pitch, as Praat's ac method
MIN_HZ, MAX_HZ = 75.0, 600.0
VOICING, SILENCE, OCTAVE_COST = 0.45, 0.03, 0.01
GAP_S = OPENING_S = MIN_VOICED_S = 0.25


def pitch(samples: np.ndarray, rate: int) -> np.ndarray:
    """Pitch in Hz every 10 ms (0 = unvoiced) of mono float audio."""
    if rate != RATE:
        import scipy.signal

        samples = scipy.signal.resample_poly(samples, RATE, rate)
    samples = np.asarray(samples, dtype=np.float64)
    peak = np.abs(samples).max() if samples.size else 0.0
    if peak <= 0 or samples.size < WINDOW:
        return np.zeros(0)
    count = 1 + (samples.size - WINDOW) // HOP
    frames = np.lib.stride_tricks.as_strided(
        samples, (count, WINDOW), (samples.strides[0] * HOP, samples.strides[0])).copy()
    local_peak = np.abs(frames).max(1)
    frames -= frames.mean(1, keepdims=True)
    window = np.hanning(WINDOW)
    size = 1 << int(np.ceil(np.log2(2 * WINDOW)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames * window, size)) ** 2, size)[:, :WINDOW]
    window_ac = np.fft.irfft(np.abs(np.fft.rfft(window, size)) ** 2, size)[:WINDOW]
    ac = ac / np.maximum(ac[:, :1], 1e-12) / np.maximum(window_ac / window_ac[0], 1e-6)
    low, high = int(RATE / MAX_HZ), min(int(RATE / MIN_HZ), WINDOW - 2)
    lags = np.arange(low, high + 1)
    strength = ac[:, low:high + 1]
    # Praat's octave cost: a candidate's strength less OCTAVE_COST * log2(MIN_HZ / f), so
    # a lower octave must be stronger to win
    score = strength - OCTAVE_COST * np.log2(MIN_HZ / (RATE / lags))[None, :]
    # only true peaks are candidates (as Praat's): low-frequency audio correlates highly
    # at the shortest lags without a period there, which read as the 600 Hz limit
    peak_lag = np.zeros_like(strength, dtype=bool)
    peak_lag[:, 1:-1] = (strength[:, 1:-1] > strength[:, :-2]) & (strength[:, 1:-1] >= strength[:, 2:])
    score = np.where(peak_lag, score, -np.inf)
    best = score.argmax(1)
    r = strength[np.arange(count), best]
    lag = lags[best].astype(np.float64)
    inner = (best > 0) & (best < len(lags) - 1)  # parabolic refinement of the peak
    left = strength[np.arange(count), np.clip(best - 1, 0, None)]
    right = strength[np.arange(count), np.clip(best + 1, None, len(lags) - 1)]
    bend = left - 2 * r + right
    shift = np.where(inner & (bend < 0), 0.5 * (left - right) / np.where(bend < 0, bend, -1), 0.0)
    hz = RATE / (lag + shift)
    voiced = peak_lag.any(1) & (r >= VOICING) & (local_peak >= SILENCE * peak)
    hz = np.where(voiced, hz, 0.0)
    # Praat's path cost keeps single frames from jumping an octave; a 5-frame median over
    # each voiced run does the same here
    smooth = hz.copy()
    for start, end in runs(hz > 0):
        if end - start >= 3:
            run = np.pad(hz[start:end], 2, mode="edge")
            smooth[start:end] = np.median(np.lib.stride_tricks.sliding_window_view(run, 5), axis=1)
    return smooth


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def openings(hz: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(opening pitch, voiced pitches) of each phrase."""
    phrases, current, gap = [], [], 0
    for value in hz:
        if value > 0:
            if gap * HOP >= GAP_S * RATE and current:
                phrases.append(current)
                current = []
            current.append(value)
            gap = 0
        else:
            gap += 1
    if current:
        phrases.append(current)
    frames = int(OPENING_S * RATE / HOP)
    return [(float(np.median(p[:frames])), np.array(p)) for p in phrases if len(p) * HOP >= MIN_VOICED_S * RATE]


def jump(samples: np.ndarray, rate: int, before: list[float] | None = None) -> float | None:
    """The take's register jump in semitones (0: none qualifies; None: not measurable).
    before: openings of earlier chunks of the same take, compared with as well."""
    found = openings(pitch(samples, rate))
    earlier = list(before or [])
    if len(found) + len(earlier) < 2:
        return None
    worst = 0.0
    for index, (opening, voiced) in enumerate(found):
        if index == 0 and not earlier:
            continue  # a take's first phrase sets the register
        others = earlier + [o for i, (o, _) in enumerate(found) if i != index]
        if not others:
            continue
        median = float(np.median(others))
        excess = 12 * np.log2(opening / median)
        high = np.mean(voiced >= median * 2 ** (HIGH_ST / 12))
        if excess >= JUMP_ST and high >= HIGH_SHARE:
            worst = max(worst, excess)
    return worst


def take_openings(samples: np.ndarray, rate: int) -> list[float]:
    return [o for o, _ in openings(pitch(samples, rate))]


def active() -> bool:
    """On with the Studio's "Enable Experimental Changes" (upstream_sampling), and only for
    takes the word check holds (Quality Mode): a streamed take has been heard already."""
    import sys

    sampling = sys.modules.get("upstream_sampling")
    return JUMP_ST > 0 and bool(getattr(sampling, "ENABLED", False))


def pcm_jump(pcm: bytes, before: list[float] | None = None) -> float | None:
    """jump() of 44.1 kHz PCM16 audio."""
    return jump(np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768, 44100, before)


def pcm_openings(pcm: bytes) -> list[float]:
    return take_openings(np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768, 44100)
