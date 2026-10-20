"""Make the model follow <tags>: classifier-free guidance on a line's <…> markup.

A cloned voice copies the delivery of its reference clip, and that can outweigh the
script's own tags: with a loud, energetic clone, <whisper>…</whisper> came out voiced in
nearly every take (0.93 of the audio voiced; random voices 0.26). The model does read
the tag, it just weighs it less than the reference.

Guidance amplifies what the tags change. Each decode step runs the backbone on two rows
in one batch: the prompt as written, and the same prompt with the current line's <…>
markup blanked (each of its tokens replaced by a space, so both rows keep the same
positions and share one KV cache layout). The main token is drawn from

    log p = log p_tagged + WEIGHT * (log p_tagged - log p_blanked)

over the audio tokens the tagged row itself finds plausible (at least PLAUSIBLE times
its most likely one), while end of speech keeps its tagged probability. Without those
two limits, guided whispers lost their last words (6 of 12 takes; with them, 1 of 16).
The other codebooks come from the tagged row, as before.

Measured with a cloned voice that resisted tags (8-12 takes per line): <whisper> voiced
share 0.65 -> 0.25 (English) and 0.21 -> 0.06 (Portuguese), every word kept; a whispered
span inside a voiced line came out whispered in 9 of 10 takes (2 of 10 without); <crying>
pitch 234 -> 298 Hz, <shouting> 261 Hz against 217 Hz for the same line untagged.

In these graphs the pause fix's windowed rule counts only its silence codes, not the
7 quiet codes: a whisper is largely quiet codes, and the full rule turned guided
whispers back into voice (0.24 voiced instead of 0.06), while without any window rule
dithered pauses stalled into drones again.

Only cloned voices are guided (the prompt has a reference clip): random and designed
voices follow tags on their own, and there guidance amplified the tags' side effects as
well. "Direction: …\nLine: <whispers> Okay... </whispers> Are you…" with a random voice
skips its whispered span in 4 of 10 takes unguided, 10 of 10 guided; with the clone,
guidance took it from 6 to 9 of 10 complete, the span whispered and the rest voiced.

Only lines with <…> markup pay for it: two backbone rows per frame (and up to twice the
experts read), in their own CUDA graphs; about 0.75 s per second of audio on an RTX 5090
instead of 0.5 s, and 0.3 GiB more VRAM. Lines without <…> run exactly as before.
[cues] and <|speaker|> tokens are not touched. DRAMA3_TAG_GUIDANCE sets WEIGHT (0 turns
it off), DRAMA3_TAG_PLAUSIBLE the plausibility floor.
"""

from __future__ import annotations

import math
import os
import re
import time

import torch
import torch.nn.functional as F

WEIGHT = float(os.environ.get("DRAMA3_TAG_GUIDANCE", "4"))
PLAUSIBLE = float(os.environ.get("DRAMA3_TAG_PLAUSIBLE", "0.01"))
MARKUP = re.compile(r"</?[A-Za-z][^<>]*>")  # <tag …> and </tag>; not <|special|> tokens
USER_TURN = "<|im_start|>user\n"
LOOKAHEAD = "<|endofturn|>"  # long_takes.END_OF_TURN


def log(message: str) -> None:
    print(f"[tags] {message}", flush=True)


def blanked_prompt(tokenizer, prompt: torch.Tensor, eos_id: int, semantic_start: int) -> torch.Tensor | None:
    """prompt with the <…> markup of its last user turn (the line being voiced) blanked,
    or None when that turn has none. Earlier turns are history and stay as they were."""
    column = prompt[:, 0].tolist()
    opener = tokenizer.encode(USER_TURN).ids
    starts = [i for i in range(len(column) - len(opener), -1, -1) if column[i:i + len(opener)] == opener]
    if not starts:
        return None
    # Only a cloned voice: its reference clip is what outweighs the tags. Random and
    # designed voices follow tags on their own, and there guidance amplified the tags'
    # side effects too (a whispered span after a Direction: skipped in 10 of 10 takes,
    # 4 of 10 unguided)
    system = [i for i in column[:starts[-1]] if i < semantic_start]
    if "[Reference 1]" not in tokenizer.decode(system, skip_special_tokens=False):
        return None
    first = starts[0] + len(opener)
    try:
        last = column.index(eos_id, first)  # <|im_end|> closes the turn
    except ValueError:
        return None
    ids = column[first:last]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoding = tokenizer.encode(text)
    if encoding.ids != ids:  # the turn was one tokenized segment; anything else is not ours to edit
        return None
    # A long take's chunk carries the next chunk's text after <|endofturn|> (lookahead):
    # only the chunk being voiced is guided
    text = text.split(LOOKAHEAD, 1)[0]
    spans = [match.span() for match in MARKUP.finditer(text)]
    if not spans:
        return None
    space = tokenizer.encode(" ").ids[0]
    rows = prompt.clone()
    blanked = 0
    for index, (start, end) in enumerate(encoding.offsets):
        if end > start and any(start >= a and end <= b for a, b in spans):
            rows[first + index, 0] = space
            blanked += 1
    return rows if blanked else None


def guided_logits(tagged: torch.Tensor, blanked: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Log-probabilities over the candidates [end of speech, audio tokens…]."""
    tagged, blanked = torch.log_softmax(tagged.float(), -1), torch.log_softmax(blanked.float(), -1)
    eos = tagged[..., :1]
    audio = tagged[..., 1:] + weight * (tagged[..., 1:] - blanked[..., 1:])
    if PLAUSIBLE > 0:  # only among the tokens the tagged prompt itself finds plausible
        floor = tagged[..., 1:].max(-1, keepdim=True).values + math.log(PLAUSIBLE)
        audio = audio.masked_fill(tagged[..., 1:] < floor, -1e4)  # finite: the step checks for non-finite logits
    audio = torch.log_softmax(audio, -1) + torch.log1p(-eos.exp().clamp(max=1 - 1e-6))
    return torch.cat((eos, audio), -1)


def install() -> bool:
    """Wrap the engine's decode graphs and Engine (after pause_fix, which has the hooks)."""
    import drama3_engine as de

    if WEIGHT <= 0:
        log("tag guidance off (DRAMA3_TAG_GUIDANCE=0)")
        return False
    base_graphs = de.DecodeGraphs
    if not hasattr(base_graphs, "main_logits"):
        log("tag guidance off: it needs the pause fix's decode step (run Install)")
        return False

    class GuidedGraphs(base_graphs):
        """The decode graphs, plus a second set that steps two backbone rows."""

        def __init__(self, model, candidate_ids, candidate_head) -> None:
            super().__init__(model, candidate_ids, candidate_head)
            device = self.hidden.device
            self.pair_hidden = torch.zeros((2, self.hidden.shape[-1]), dtype=self.hidden.dtype, device=device)
            self.weight = torch.tensor(WEIGHT, dtype=torch.float32, device=device)
            self.guided_graphs: dict[int, torch.cuda.CUDAGraph] = {}
            self.guided = False  # the current take uses the guided graphs
            self._guiding = False  # capturing them

        def main_logits(self) -> torch.Tensor:
            if not self._guiding:
                return super().main_logits()
            logits = F.linear(self.pair_hidden, self.candidate_head)
            return guided_logits(logits[:1], logits[1:], self.weight)

        def advance(self, embeddings: torch.Tensor, kv_len: int) -> None:
            if not self._guiding:
                return super().advance(embeddings, kv_len)
            hidden = self.model.backbone(inputs_embeds=embeddings.expand(2, -1)[:, None],
                                         input_pos=self.position + 1, static_kv_len=kv_len)
            self.pair_hidden.copy_(hidden[:, 0])
            self.hidden.copy_(hidden[0])  # the tower samples the other codebooks from the tagged row

        def capture_guided(self, kv_len: int) -> None:
            plain, self.graphs = self.graphs, self.guided_graphs
            self._guiding, self.window_rule = True, "silence"
            try:
                self.capture(kv_len)
            finally:
                self._guiding, self.window_rule = False, "quiet"
                self.graphs = plain

        def replay(self, position: int) -> None:
            graphs = self.guided_graphs if self.guided else self.graphs
            graphs[next(b for b in self.buckets if b >= position + 2)].replay()

    class GuidedEngine(de.Engine):
        def decode_graphs(self):
            backbone = self.model.backbone
            if self.graphs is None and getattr(backbone, "max_batch_size", 0) < 2:
                # Two rows per cache from the start: the plain graphs use the first
                # (about 0.25 GB more for the second)
                capacity = de.DecodeGraphs.graph_capacity(self.model)
                backbone.setup_caches(max_batch_size=2, max_seq_len=capacity)
                self.model.audio_decoder.setup_caches(capacity)
                self.model.cache_capacity = capacity
            return super().decode_graphs()

        def graph_for(self, kv_len: int) -> None:
            super().graph_for(kv_len)
            if kv_len == de.WARM_UP_BUCKETS[0]:
                self.guided_graph_for(kv_len)  # the common bucket, ready before the first tagged line

        def guided_graph_for(self, kv_len: int) -> None:
            graphs = self.decode_graphs()
            if kv_len not in graphs.guided_graphs:
                started = time.perf_counter()
                with torch.cuda.device(self.device):
                    graphs.capture_guided(kv_len)
                log(f"guided decode graph for {kv_len} cache rows ready in {time.perf_counter() - started:.1f}s")

        def prefill(self, prompt, request):
            graphs = self.decode_graphs()
            blanked = blanked_prompt(self.tokenizer, prompt, self.config.eos_token_id, self.config.semantic_start_token_id)
            if blanked is None:
                graphs.guided = False
                return super().prefill(prompt, request)
            # The Space's prefill, on both rows
            model = self.model
            backbone, tower = model.backbone, model.audio_decoder
            prompt_rows = len(prompt)
            capacity = prompt_rows + request.max_new_tokens
            if capacity > graphs.graph_capacity(model):
                raise de.RequestError("the prompt and audio limit exceed the context")
            needed = next(b for b in graphs.buckets if b >= capacity)
            for bucket in graphs.buckets:
                if prompt_rows + 1 <= bucket <= needed:
                    self.guided_graph_for(bucket)
            device = tower.device
            frames, plain = prompt.to(device), blanked.to(device)
            positions = torch.arange(prompt_rows, device=device)
            for start in range(0, prompt_rows, de.PREFILL_CHUNK_ROWS):
                end = min(start + de.PREFILL_CHUNK_ROWS, prompt_rows)
                rows = torch.stack([tower.embed_backbone(f[start:end], backbone.embed(f[start:end, 0]))
                                    for f in (frames, plain)])
                hidden = backbone(inputs_embeds=rows, input_pos=positions[start:end])
                paired_end = min(end, prompt_rows - 1)
                if paired_end > start:
                    tower.prefill(hidden[0, : paired_end - start], positions[start:paired_end],
                                  frames[start:paired_end], frames[start + 1: paired_end + 1])
            graphs.start(hidden[0, -1:], prompt_rows - 1, frames, request.temperature, request.top_k, request.top_p)
            graphs.pair_hidden.copy_(hidden[:, -1])
            graphs.guided = True
            log(f"guidance {WEIGHT:g} on the line's <tags>")
            return graphs

    GuidedGraphs.__name__ = base_graphs.__name__
    GuidedEngine.__name__ = de.Engine.__name__
    de.DecodeGraphs = GuidedGraphs
    de.Engine = GuidedEngine
    log(f"tag guidance on: weight {WEIGHT:g} for lines with <…> markup")
    return True
