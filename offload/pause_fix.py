"""The drama3-small "pause fix", in the Space's CUDA-graph decode (no audio.cpp).

Drama 3 (the full model too, less often than its compressed versions) can go silent at
a tag or sentence boundary and never resume or stop. drama3-small's authors fixed it in the sampler
(drama3-small-audiocpp.patch, generator.cpp); this is the same rule:

- A frame is silent when its first audio code is one of silence_codes.json (143 codes).
- After MAX_PAUSE_FRAMES silent frames in a row (10 = 0.47 s), those codes are masked
  from the next main-token draw: the model must resume speaking or end.
- Length guard, from the text's expected length (2 frames per Latin letter or digit,
  3.88 per CJK character, 23.9 per [tag] / <tag>, + 8.5): a pause after 2x that length
  ends the take; 3x ends it regardless.

The masking and the silence count run inside the decode graph (tensor ops); the two
stop rules run on the host, which reads the count with the step's status.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

import stall_guard

MAX_PAUSE_FRAMES = int(os.environ.get("DRAMA3_MAX_PAUSE_FRAMES", 10))
# A pause the model dithers (silence codes broken by single quiet frames that are not
# in the list) never makes MAX_PAUSE_FRAMES in a row: it is also cut once
# PAUSE_WINDOW_MAX of the last PAUSE_WINDOW frames are silent or QUIET_CODES
PAUSE_WINDOW = int(os.environ.get("DRAMA3_PAUSE_WINDOW", 20))
PAUSE_WINDOW_MAX = int(os.environ.get("DRAMA3_PAUSE_WINDOW_MAX", 10))
# The quiet first codes such pauses dither with (each 3-6x more common there than in
# speech, where they are about 1% of frames: breaths, pause edges). Not silence codes:
# they never start a pause count, but they count in the window and are masked with
# the silence codes while a pause is cut, so the model has to speak
QUIET_CODES = (112, 436, 1242, 1573, 2546, 2586, 3303)
END_RATIO = float(os.environ.get("DRAMA3_PAUSE_END_RATIO", 2.0))
CAP_RATIO = float(os.environ.get("DRAMA3_PAUSE_CAP_RATIO", 3.0))
_CJK_PUNCT = {0x3002, 0x3001, 0xFF01, 0xFF1F, 0x300C, 0x300D, 0x2026, 0x30FB, 0xFF08, 0xFF09}
SILENCE_CODES: list[int] = []
SILENCE_SET: set[int] = set()


def log(message: str) -> None:
    print(f"[pause] {message}", flush=True)


def load_codes(silence_codes_path: str) -> None:
    SILENCE_CODES[:] = sorted({int(c) for c in json.load(open(silence_codes_path, encoding="utf-8"))})
    SILENCE_SET.clear()
    SILENCE_SET.update(SILENCE_CODES)


def expected_frames(text: str) -> float:
    """The patch's fitted estimate of a line's length in frames (21.5 per second)."""
    if text.startswith("Direction:") and "\nLine:" in text:
        text = text.split("\nLine:", 1)[1]
    cjk = latin = spans = 0
    square = angle = 0
    for char in text:
        code = ord(char)
        if char == "[":
            square += 1
            spans += 1
        elif char == "]":
            square = max(0, square - 1)
        elif char == "<":
            angle += 1
            spans += 1
        elif char == ">":
            angle = max(0, angle - 1)
        elif square or angle:
            continue
        elif code >= 0x3000 and code not in _CJK_PUNCT:
            cjk += 1
        elif code < 0x80 and char.isalnum():
            latin += 1
    return 3.88 * cjk + 2.00 * latin + 23.9 * spans + 8.5


def install(silence_codes_path: str) -> None:
    import drama3_engine as de

    load_codes(silence_codes_path)
    base_graphs = de.DecodeGraphs

    class PauseClampGraphs(base_graphs):
        """The Space's decode graph plus the pause clamp (masking + silence count)."""

        def __init__(self, model, candidate_ids, candidate_head) -> None:
            super().__init__(model, candidate_ids, candidate_head)
            device = self.hidden.device
            codes = torch.tensor(SILENCE_CODES, dtype=torch.long)
            candidates = torch.zeros(len(candidate_ids), dtype=torch.bool)
            candidates[1 + codes] = True  # candidates are [EOS, semantic 0 .. 4095]
            lookup = torch.zeros(model.audio_decoder.config.vocab_size, dtype=torch.bool)
            lookup[codes] = True
            self.silence_candidates = candidates[None].to(device)
            self.silence_lookup = lookup.to(device)
            quiet = torch.tensor(QUIET_CODES, dtype=torch.long)
            candidates[1 + quiet] = True
            lookup[quiet] = True
            self.pause_candidates = candidates[None].to(device)
            self.pause_lookup = lookup.to(device)
            self.silent_run = torch.zeros(1, dtype=torch.long, device=device)
            self.silent_window = torch.zeros(PAUSE_WINDOW, dtype=torch.long, device=device)
            self.silence_window = torch.zeros(PAUSE_WINDOW, dtype=torch.long, device=device)
            # Read when a graph is captured: "quiet" counts the silence and quiet codes in the
            # window; "silence" only the silence codes (tag_guidance's graphs: a whisper is
            # largely quiet codes, while the pauses that stall are mostly silence codes)
            self.window_rule = "quiet"
            self.status = torch.zeros(3, dtype=torch.long, device=device)  # main, finite, silent run

        def run(self, kv_len: int) -> None:
            backbone, tower = self.model.backbone, self.model.audio_decoder
            logits = self.main_logits()
            finite = torch.isfinite(logits).all()
            if MAX_PAUSE_FRAMES > 0:
                clamp = self.silence_candidates & (self.silent_run >= MAX_PAUSE_FRAMES)
                if self.window_rule == "quiet":
                    clamp = clamp | (self.pause_candidates & (self.silent_window.sum() >= PAUSE_WINDOW_MAX))
                elif self.window_rule == "silence":
                    clamp = clamp | (self.silence_candidates & (self.silence_window.sum() >= PAUSE_WINDOW_MAX))
                logits = logits.masked_fill(clamp, float("-inf"))
            sampled = de.decode_candidates(
                tower, self.candidate_ids, logits, self.hidden, self.position, self.frame, self.history,
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            )
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            self.advance(embeddings, kv_len)
            self.out_frames.index_copy_(0, self.out_index.clamp(max=de.MAX_NEW_TOKENS_CAP), sampled)
            silent = self.silence_lookup.index_select(0, sampled[0, 1:2])  # [1], no host sync
            self.silent_run.copy_(torch.where(silent, self.silent_run + 1, torch.zeros_like(self.silent_run)))
            paused = self.pause_lookup.index_select(0, sampled[0, 1:2])
            self.silent_window.copy_(torch.cat((self.silent_window[1:], paused.long())))
            self.silence_window.copy_(torch.cat((self.silence_window[1:], silent.long())))
            self.status.copy_(torch.stack((sampled[0, 0], finite.long(), self.silent_run[0])))
            self.frame.copy_(sampled)
            self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
            self.position.add_(1)
            self.out_index.add_(1)

        def main_logits(self) -> torch.Tensor:
            """The main token's logits over the candidates (tag_guidance replaces this)."""
            return F.linear(self.hidden, self.candidate_head)

        def advance(self, embeddings: torch.Tensor, kv_len: int) -> None:
            """Feed the sampled frame to the backbone: the next step's hidden state."""
            hidden = self.model.backbone(inputs_embeds=embeddings[None], input_pos=self.position + 1,
                                         static_kv_len=kv_len)
            self.hidden.copy_(hidden[0])

        def start(self, hidden, position, prompt_frames, temperature, top_k, top_p) -> None:
            super().start(hidden, position, prompt_frames, temperature, top_k, top_p)
            self.silent_run.zero_()
            self.silent_window.zero_()
            self.silence_window.zero_()

    de.DecodeGraphs = PauseClampGraphs
    base_engine = de.Engine

    class PauseClampEngine(base_engine):
        @torch.inference_mode()
        def frames(self, prompt, request):
            """The Space's frame loop, plus the length guard and the end-after-pause rule."""
            graphs = self.prefill(prompt, request)
            first_position = len(prompt) - 1
            eos = self.config.eos_token_id
            expected = expected_frames(request.text)
            end_frames, cap_frames = int(END_RATIO * expected), int(CAP_RATIO * expected)
            status = torch.empty(3, dtype=torch.long, pin_memory=True)
            ready = torch.cuda.Event()
            guard = stall_guard.StallGuard()
            graphs.replay(first_position)
            for step in range(request.max_new_tokens):
                status.copy_(graphs.status, non_blocking=True)
                ready.record()
                if step + 1 < request.max_new_tokens:
                    graphs.replay(first_position + step + 1)
                ready.synchronize()
                main, finite, silent = status.tolist()
                if not finite:
                    raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
                if main == eos:
                    yield from guard.release(step)
                    return "stop"
                self.frame_ready = ready
                stalled, counts = guard.add(main, step + 1)
                if stalled:
                    log(f"stall: a drone from frame {guard.released}; the take ends there")
                    return "stall"
                yield from counts
                if MAX_PAUSE_FRAMES > 0 and step + 1 >= cap_frames:
                    yield from guard.release(step + 1)
                    return "stop"  # length guard: babble backstop
                if MAX_PAUSE_FRAMES > 0 and silent >= MAX_PAUSE_FRAMES and step + 1 >= end_frames:
                    yield from guard.release(step + 1)
                    return "stop"  # a pause after the text should be done
            yield from guard.release(request.max_new_tokens)
            return "length"

    PauseClampEngine.__name__ = base_engine.__name__
    de.Engine = PauseClampEngine
