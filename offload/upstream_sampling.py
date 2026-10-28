"""Changes to the Space's sampling from others' findings, behind one switch.

All are experimental and off by default: the Studio's "Enable Experimental Changes" box
(set_enabled; DRAMA3_EXPERIMENTAL_SAMPLING=1 starts with them on) turns them on and off
together, from the next take. Off, every draw is the Space's own, bit for bit, except a
codebook code the codec lacks (below).

From torenyx/drama3-24gb:

- the main token (end of speech or a semantic code) is drawn with no top-p cut; the
  codebook heads keep the request's top_p, and top_k still applies. In a drone the
  repeating code holds ~97% and end of speech is second at ~3%, which top_p 0.9 cuts on
  every step, so the drone cannot end (docs/failures.md: 96 of 96 steps). There, whole
  scripts at top_p 1 instead of 0.9 went from 3 of 8 long silences to 0, and from 4 of 8
  with substantial omissions to 1 (8 unpaired runs each);
- a remake of a take or chunk that stalled (long_takes sets set_relaxed for it) also
  draws its RAS redraw with no top-k or top-p cut, as its stalled-chunk retry does
  (runtime/stalled_chunk.py: semantic k=0, p=1 on the retry; the codebook heads unchanged).

Here, on lines full of pauses and whispers (two cloned voices, 5 lines x 8 seeds, Quality
Mode) the two caught 70 drones to 136, passed the word check first time in 61 takes to 51,
and left 17 takes missing words after every retry to 26; the relaxed remakes scored like
ordinary ones, so the gain is the top_p. A blind check picked them 7 of 11, none worst.

From SGLang Omni's fixes of 2026-10-20 (sglang-omni #2810, #2890):

- the main token and its RAS redraw keep the token whose probability crosses top_p, as
  upstream's semantic sampling does (#2810); the codebook heads still drop it. With the
  main token uncut this only matters to the RAS redraw (top_p 0.95);
- the codebook heads draw with top_k at most 30, whatever the request asks (#2890: wider
  caps there cost speaker similarity). The Space sends top_k 4097 (no cap, see server.py
  ALL_CANDIDATES), which kept the main token from going silent and also uncapped the heads.

In a blind test of these two (before the others) no setup was picked best more than
chance (22 / 21 / 18 / 17 of 78); takes with the head cap were picked worst 7 times to 17
without, and a cloned voice's pitch varied less between seeds (Joker: std 9.3 -> 7.3
semitones). Keeping the crossing token caught more drones (14, and 17 with both, to 8).

With the box on or off, the codebook heads never draw a code past the codec's 1024-entry
residual codebooks (sglang-omni #3278): the codec clamps such a code to 1023, another
sound. Here 0.6% of head draws were past it, all but a few in the frame decoded after the
end of speech, which is never heard; about one take in eight had one in its audio, each
in a take that had already gone wrong (it ended at once and was made again).

The switches are flag tensors the decode reads at every step, so the captured CUDA graphs
follow them without being captured again. Applies to the CUDA-graph decode
(drama3_engine.decode_candidates, which pause_fix and drama3_portable call too) and to the
tower's own decode_step. The MLX backend samples in mlx_audio, unchanged.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("DRAMA3_EXPERIMENTAL_SAMPLING", "0").strip().lower() in {"1", "true", "yes", "on"}
CODEBOOK_TOP_K = 30  # sglang-omni audio_decoder.py
# Codes in each of the codec's 9 residual codebooks (dac/rvq.py codebook_size); the
# codebook heads score 4096, and the codec clamps a code past 1023 to 1023
RESIDUAL_CODEBOOK_SIZE = 1024
_flags: dict[str, dict[str, torch.Tensor]] = {}  # per device, read inside the decode
RELAXED = False


def _key(device) -> str:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def flags(device) -> dict[str, torch.Tensor]:
    """The switches on device (made outside graph capture: install makes the CUDA ones)."""
    key = _key(device)
    if key not in _flags:
        _flags[key] = {"experimental": torch.tensor(ENABLED, device=key),
                       "relaxed": torch.tensor(RELAXED, device=key)}
    return _flags[key]


def set_enabled(on: bool) -> None:
    """The experimental changes, from the next take."""
    global ENABLED
    ENABLED = bool(on)
    for value in _flags.values():
        value["experimental"].fill_(ENABLED)


def set_relaxed(on: bool) -> None:
    """The next generation remakes one that stalled: no cut on its RAS redraw."""
    global RELAXED
    on = bool(on) and ENABLED
    if on == RELAXED:
        return
    RELAXED = on
    for value in _flags.values():
        value["relaxed"].fill_(RELAXED)


def sample(logits, temperature, top_k, top_p, *, keep_crossing=False, no_top_p=False, uncut=False,
           legal=None):
    """tower.sample_top_k_top_p. keep_crossing (a bool or a flag tensor) keeps the token
    that crosses top_p; no_top_p (a flag tensor) makes no top-p cut; uncut (a flag tensor)
    drops no token at all; legal (a count) drops ids from legal up after the other rules,
    keeping the most probable legal id. The other ids keep their odds and the noise is the
    same, so a draw changes only where an id past legal would have won it."""
    from fish_speech.models.text2semantic.fish_omni.tower import MIN_TEMPERATURE

    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=MIN_TEMPERATURE)
    else:
        temperature = max(temperature, MIN_TEMPERATURE)
    values, ids = torch.sort(logits.float() / temperature, descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = values.softmax(-1).cumsum(-1) > top_p
    if keep_crossing is not False:  # a token is dropped when the mass before it already exceeds top_p
        shifted = torch.cat((torch.zeros_like(dropped[..., :1]), dropped[..., :-1]), -1)
        dropped = shifted if keep_crossing is True else torch.where(keep_crossing, shifted, dropped)
    if no_top_p is not False:
        dropped = dropped & ~no_top_p
    dropped = dropped | (ranks >= top_k)
    if uncut is not False:
        dropped = dropped & ~uncut
    dropped[..., 0] = False
    if legal is not None:
        illegal = ids >= legal
        dropped = dropped | illegal
        dropped.scatter_(-1, (~illegal).int().argmax(-1, keepdim=True), False)
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_()
    choice = (probabilities / noise).argmax(-1, keepdim=True)
    return ids.gather(-1, choice).squeeze(-1)


def decode_candidates(tower, candidates, candidate_logits, hidden, positions, frames, history,
                      *, temperature, top_k, top_p) -> torch.Tensor:
    """drama3_engine.decode_candidates with the sampling above."""
    from fish_speech.models.text2semantic.fish_omni.tower import (
        RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW_FRAMES,
    )

    switch = flags(candidate_logits.device)
    on, relaxed = switch["experimental"], switch["relaxed"]
    first = candidates[sample(candidate_logits, temperature, top_k, top_p, keep_crossing=on, no_top_p=on)]
    retry = candidates[sample(candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, keep_crossing=on,
                              uncut=relaxed)]
    repeated = (history[-RAS_WINDOW_FRAMES:] == first[:, None]).any(-1)
    repeated &= tower.is_semantic(first)
    main = torch.where(repeated, retry, first)
    c0 = (main - tower.semantic_begin).clamp(0, tower.config.vocab_size - 1)
    semantic = tower.is_semantic(main)
    hidden = tower.run_shared(hidden, c0, semantic, positions, frames)
    codes = [c0]
    k = top_k if isinstance(top_k, torch.Tensor) else torch.tensor(top_k, device=on.device)
    heads_k = torch.where(on, k.clamp(max=CODEBOOK_TOP_K), k)
    for layer, head in zip(tower.residual_layers, tower.codebook_heads):
        hidden = layer(hidden, codes[-1], semantic, positions)
        codes.append(sample(head(hidden), temperature, heads_k, top_p, legal=RESIDUAL_CODEBOOK_SIZE))
    return torch.stack((main, *codes), dim=-1)


def install() -> None:
    import drama3_engine
    from fish_speech.models.text2semantic.fish_omni import tower

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            flags(torch.device("cuda", index))
    drama3_engine.decode_candidates = decode_candidates

    def decode_step(self, logits, hidden, positions, frames, history, *, temperature, top_k, top_p,
                    check_finite=True):
        if check_finite and not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
        if self.main_candidates is None or self.main_candidates.device != self.device:
            eos = torch.tensor([self.eos_id], device=self.device)
            band = torch.arange(self.semantic_begin, self.semantic_end + 1, device=self.device)
            self.main_candidates = torch.cat((eos, band)).sort().values
        candidates = self.main_candidates
        return decode_candidates(self, candidates, logits[:, candidates], hidden, positions, frames, history,
                                 temperature=temperature, top_k=top_k, top_p=top_p)

    tower.FishOmniAudioDecoder.decode_step = decode_step
    print(f"[sampling] experimental changes "
          f"{'on' if ENABLED else 'off'} (the Studio's box switches them)", flush=True)
