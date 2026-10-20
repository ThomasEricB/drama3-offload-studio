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

Voice guidance (the Studio's "Check every word before playing" takes only): the same
guidance on the reference clip, so a clone sounds more like its clip. The comparison row
is the prompt without its [Reference] blocks, the prompt Drama 3 uses for a voice of its
own (the "input_text" form of Selective CFG, arXiv 2509.19668: keep the text, drop the
voice prompt), at VOICE_WEIGHT:

    log p = log p + VOICE_WEIGHT * (log p - log p_without_voice)   (+ the tag term above)

Measured on two clean Trump clips (16 takes per setting and voice, 4 lines in English and
Portuguese): speaker similarity (ECAPA) 0.443 -> 0.608 and 0.578 -> 0.695, also +0.10 on
a Trump recording the model never saw, x-vector +0.016 / +0.003, median pitch 5.4 -> 2.8
and 1.8 -> 1.1 semitones from the clip's; every word kept. Weight 2 beat 1 and 3, as in
SemBridge (arXiv 2608.07462). A clip with a music bed has its bed copied louder too.

The row without the voice is shorter: it sits right-aligned in its cache row, its first
token at position P - U, behind padding (copies of the prompt rows) that full attention
skips (ROW_START) and whose recurrent state is reset once it is done. RoPE is relative,
so the shifted positions change nothing (its last hidden state matched the same prompt
run alone: cosine 0.988-0.999 over single lines, long-script chunks, retakes and
tagged lines, against 0.73-0.99 for the prompt's own row). A tagged line gets both:
three rows per frame. The cost: about 1.5x the time per take. The caches keep two
rows (a third, even unused, changes how the compiled step reads them, and so its
roundings: plain and tagged takes stay sample-identical to before only with two); a
tagged line with voice guidance gets a three-row set of its own (0.77 GiB, allocated on
first use and kept).
"""

from __future__ import annotations

import math
import os
import re
import time

import torch
import torch.nn.functional as F

WEIGHT = float(os.environ.get("DRAMA3_TAG_GUIDANCE", "4"))
VOICE_WEIGHT = float(os.environ.get("DRAMA3_VOICE_GUIDANCE", "2"))  # the Studio's slider starts here
ROWS = 3  # the prompt, tags blanked, voice removed
CACHE_ROWS = 2  # every take's caches; a third row has caches of its own (see GuidedEngine.use_caches)
PLAUSIBLE = float(os.environ.get("DRAMA3_TAG_PLAUSIBLE", "0.01"))
MARKUP = re.compile(r"</?[A-Za-z][^<>]*>")  # <tag …> and </tag>; not <|special|> tokens
USER_TURN = "<|im_start|>user\n"
LOOKAHEAD = "<|endofturn|>"  # long_takes.END_OF_TURN
STREAM_INSTRUCTION = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"  # long_takes


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


def voiceless_prompt(tokenizer, config, prompt: torch.Tensor) -> torch.Tensor | None:
    """prompt without its [Reference] blocks (the system turn holds only the plain or the
    streaming instruction, as drama3_engine and long_takes write it for a voice of the
    model's own), or None when it has none. Earlier turns (history) are kept."""
    import drama3_engine as de

    column = prompt[:, 0].tolist()
    try:
        end = column.index(config.eos_token_id)  # the system turn's <|im_end|>
    except ValueError:
        return None
    system = tokenizer.decode([i for i in column[:end] if i < config.semantic_start_token_id],
                              skip_special_tokens=False)
    if "[Reference 1]" not in system:
        return None
    instruction = STREAM_INSTRUCTION if STREAM_INSTRUCTION in system else de.TTS_INSTRUCTION
    head = de.segment_rows(["<|im_start|>system\n", instruction], tokenizer, config)
    return torch.cat((head, prompt[end:]))


def guided_logits(tagged: torch.Tensor, others: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Log-probabilities over the candidates [end of speech, audio tokens…]: tagged is the
    prompt's row [1, C], others the comparison rows [K, C], each pushed away from at its
    weight [K]."""
    tagged, others = torch.log_softmax(tagged.float(), -1), torch.log_softmax(others.float(), -1)
    eos = tagged[..., :1]
    audio = tagged[..., 1:] + (weights[:, None] * (tagged[..., 1:] - others[..., 1:])).sum(0, keepdim=True)
    if PLAUSIBLE > 0:  # only among the tokens the tagged prompt itself finds plausible
        floor = tagged[..., 1:].max(-1, keepdim=True).values + math.log(PLAUSIBLE)
        audio = audio.masked_fill(tagged[..., 1:] < floor, -1e4)  # finite: the step checks for non-finite logits
    audio = torch.log_softmax(audio, -1) + torch.log1p(-eos.exp().clamp(max=1 - 1e-6))
    return torch.cat((eos, audio), -1)


ROW_START: torch.Tensor | None = None  # per cache row: its first position (later ones skip what is before)
SHIFTED = False  # a right-aligned row is in use (the voice's prefill and decode graphs); else the Space's mask


def attention_forward(self, x: torch.Tensor, context) -> torch.Tensor:
    """drama3_fast.attention_forward, with each cache row's first position (ROW_START)
    in the mask: a row's queries from its first position on do not see the rows before
    it. ROW_START is 0 for the prompt's own row, so its attention is unchanged."""
    import drama3_fast
    from fish_speech.models.text2semantic.fish_omni.layers import apply_neox_rope

    batch, length, _ = x.shape
    q_width = self.num_heads * self.head_dim * 2
    kv_width = self.num_kv_heads * self.head_dim
    query_gate, key, value = F.linear(x, self.qkv_fused).split([q_width, kv_width, kv_width], dim=-1)
    query, gate = query_gate.view(batch, length, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
    gate = gate.reshape(batch, length, self.num_heads * self.head_dim)
    query = self.q_norm(query)
    key = self.k_norm(key.reshape(batch, length, self.num_kv_heads, self.head_dim))
    value = value.reshape(batch, length, self.num_kv_heads, self.head_dim)
    query = apply_neox_rope(query, context.cos, context.sin).transpose(1, 2)
    key = apply_neox_rope(key, context.cos, context.sin).transpose(1, 2)
    value = value.transpose(1, 2)
    if context.use_cache:
        self.k_cache[:batch].index_copy_(2, context.input_pos, key)
        self.v_cache[:batch].index_copy_(2, context.input_pos, value)
        key = self.k_cache[:batch, :, : context.kv_len]
        value = self.v_cache[:batch, :, : context.kv_len]
        key_positions = torch.arange(context.kv_len, device=x.device)
        query_positions = context.input_pos
        mask = key_positions[None, :] <= query_positions[:, None]
        if SHIFTED:
            # Only then: compiled into the decode graphs, the [B, 1, T, S] mask changes how
            # the step's kernels are fused (other roundings, 1.4% slower), so the plain and
            # tag-guided graphs keep the Space's [T, S] one
            first = ROW_START[:batch].view(batch, 1, 1, 1)
            mask = mask & ((key_positions >= first) | (query_positions[:, None] < first))
        is_causal = False
    else:
        mask = None
        is_causal = length > 1
    attended = drama3_fast.fish_layers.grouped_attention(query, key, value, mask, is_causal=is_causal,
                                                         scale=self.head_dim**-0.5)
    attended = attended.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
    return self.o_proj(attended * torch.sigmoid(gate))


def install() -> bool:
    """Wrap the engine's decode graphs and Engine (after pause_fix, which has the hooks);
    before the weights load, whose attention layers bind drama3_fast.attention_forward."""
    global ROW_START
    import drama3_engine as de
    import drama3_fast

    if WEIGHT <= 0 and VOICE_WEIGHT <= 0:
        log("guidance off (DRAMA3_TAG_GUIDANCE=0 and DRAMA3_VOICE_GUIDANCE=0)")
        return False
    base_graphs = de.DecodeGraphs
    if not hasattr(base_graphs, "main_logits"):
        log("guidance off: it needs the pause fix's decode step (run Install)")
        return False
    ROW_START = torch.zeros(ROWS, dtype=torch.long, device=torch.device("cuda", torch.cuda.current_device()))
    drama3_fast.attention_forward = attention_forward

    class GuidedGraphs(base_graphs):
        """The decode graphs, plus sets that step 2 or 3 backbone rows."""

        def __init__(self, model, candidate_ids, candidate_head) -> None:
            super().__init__(model, candidate_ids, candidate_head)
            device = self.hidden.device
            self.pair_hidden = torch.zeros((ROWS, self.hidden.shape[-1]), dtype=self.hidden.dtype, device=device)
            self.weights = torch.zeros(ROWS - 1, dtype=torch.float32, device=device)
            # by (rows, a right-aligned row): (2, False) tags; (2, True) the voice; (3, True) both
            self.guided_graphs: dict[tuple[int, bool], dict[int, torch.cuda.CUDAGraph]] = {
                (2, False): {}, (2, True): {}, (3, True): {}}
            self.rows, self.shifted = 1, False  # the current take's graphs (1 row: the plain ones)
            self._guiding = 0  # rows of the graphs being captured

        def main_logits(self) -> torch.Tensor:
            if not self._guiding:
                return super().main_logits()
            logits = F.linear(self.pair_hidden[:self._guiding], self.candidate_head)
            return guided_logits(logits[:1], logits[1:], self.weights[:self._guiding - 1])

        def advance(self, embeddings: torch.Tensor, kv_len: int) -> None:
            if not self._guiding:
                return super().advance(embeddings, kv_len)
            hidden = self.model.backbone(inputs_embeds=embeddings.expand(self._guiding, -1)[:, None],
                                         input_pos=self.position + 1, static_kv_len=kv_len)
            self.pair_hidden[:self._guiding].copy_(hidden[:, 0])
            self.hidden.copy_(hidden[0])  # the tower samples the other codebooks from the prompt's row

        def capture_guided(self, kv_len: int, rows: int, shifted: bool) -> None:
            global SHIFTED
            plain, self.graphs = self.graphs, self.guided_graphs[rows, shifted]
            self._guiding, self.window_rule, SHIFTED = rows, "silence", shifted
            try:
                self.capture(kv_len)
            finally:
                self._guiding, self.window_rule, SHIFTED = 0, "quiet", False
                self.graphs = plain

        def replay(self, position: int) -> None:
            graphs = self.guided_graphs[self.rows, self.shifted] if self.rows > 1 else self.graphs
            graphs[next(b for b in self.buckets if b >= position + 2)].replay()

    class GuidedEngine(de.Engine):
        voice_weight = 0.0  # set by long_takes around the takes the word check holds

        caches: dict[int, list] | None = None  # cache tensors by rows (2: every take's; 3: on first use)

        def decode_graphs(self):
            backbone = self.model.backbone
            if self.graphs is None and getattr(backbone, "max_batch_size", 0) < CACHE_ROWS:
                # Two rows per cache from the start: the plain graphs use the first
                # (about 0.25 GB more for the second)
                capacity = de.DecodeGraphs.graph_capacity(self.model)
                backbone.setup_caches(max_batch_size=CACHE_ROWS, max_seq_len=capacity)
                self.model.audio_decoder.setup_caches(capacity)
                self.model.cache_capacity = capacity
            return super().decode_graphs()

        CACHE_NAMES = {"full_attention": ("self_attn", ("k_cache", "v_cache"))}
        STATE_NAMES = ("linear_attn", ("conv_state", "recurrent_state"))

        def cache_tensors(self):
            """[(module, name)] of every backbone cache tensor."""
            out = []
            for layer in self.model.backbone.model.layers:
                module, names = self.CACHE_NAMES.get(layer.layer_type, self.STATE_NAMES)
                out += [(getattr(layer, module), name) for name in names]
            return out

        def use_caches(self, rows: int) -> None:
            """Point the backbone at the caches for `rows` rows (prefill and capture read
            the attributes; captured graphs keep the tensors they were captured with)."""
            self.decode_graphs()  # sets the caches up on first use
            slots = self.cache_tensors()
            if self.caches is None:
                self.caches = {CACHE_ROWS: [getattr(m, n) for m, n in slots]}
            wanted = CACHE_ROWS if rows <= CACHE_ROWS else rows
            if wanted not in self.caches:
                started = time.perf_counter()
                self.caches[wanted] = [t.new_zeros((wanted, *t.shape[1:])) for t in self.caches[CACHE_ROWS]]
                log(f"caches for {wanted} rows: {sum(t.nbytes for t in self.caches[wanted]) / 2**30:.2f} GiB "
                    f"in {time.perf_counter() - started:.1f}s")
            for (module, name), tensor in zip(slots, self.caches[wanted]):
                setattr(module, name, tensor)
            self.model.backbone.max_batch_size = wanted  # the prefill's bound check

        def graph_for(self, kv_len: int) -> None:
            # the plain graphs are captured on every take's caches, whichever the last take
            # used (drama3_offload.rebalance recaptures them between takes)
            self.use_caches(1)
            super().graph_for(kv_len)
            if kv_len == de.WARM_UP_BUCKETS[0] and WEIGHT > 0:
                self.guided_graph_for(kv_len, 2, False)  # the common bucket, ready before the first tagged line

        def guided_graph_for(self, kv_len: int, rows: int, shifted: bool) -> None:
            """Captured on first use (the voice's: on the first take the word check holds,
            ~5 s)."""
            graphs = self.decode_graphs()
            if kv_len not in graphs.guided_graphs[rows, shifted]:
                self.use_caches(rows)
                started = time.perf_counter()
                with torch.cuda.device(self.device):
                    graphs.capture_guided(kv_len, rows, shifted)
                log(f"guided decode graph ({rows} rows{', voice' if shifted else ''}) for {kv_len} cache rows "
                    f"ready in {time.perf_counter() - started:.1f}s")

        def comparisons(self, prompt):
            """(rows, first position, weight) of each comparison row of this prompt."""
            out = []
            if WEIGHT > 0:
                blanked = blanked_prompt(self.tokenizer, prompt, self.config.eos_token_id,
                                         self.config.semantic_start_token_id)
                if blanked is not None:
                    out.append((blanked, 0, WEIGHT))
            if self.voice_weight > 0:
                voiceless = voiceless_prompt(self.tokenizer, self.config, prompt)
                if voiceless is not None:
                    first = len(prompt) - len(voiceless)
                    out.append((torch.cat((prompt[:first], voiceless)), first, self.voice_weight))
            return out

        def prefill(self, prompt, request):
            graphs = self.decode_graphs()
            ROW_START.zero_()
            comparisons = self.comparisons(prompt)
            self.use_caches(1 + len(comparisons))
            if not comparisons:
                graphs.rows, graphs.shifted = 1, False
                return super().prefill(prompt, request)
            # The Space's prefill, on every row
            rows = 1 + len(comparisons)
            firsts = [first for _, first, _ in comparisons]
            shifted = any(firsts)
            model = self.model
            backbone, tower = model.backbone, model.audio_decoder
            prompt_rows = len(prompt)
            capacity = prompt_rows + request.max_new_tokens
            if capacity > graphs.graph_capacity(model):
                raise de.RequestError("the prompt and audio limit exceed the context")
            needed = next(b for b in graphs.buckets if b >= capacity)
            for bucket in graphs.buckets:
                if prompt_rows + 1 <= bucket <= needed:
                    self.guided_graph_for(bucket, rows, shifted)
            device = tower.device
            frames = [prompt.to(device)] + [rows_.to(device) for rows_, _, _ in comparisons]
            ROW_START[1:rows].copy_(torch.tensor(firsts))
            graphs.weights[:rows - 1].copy_(torch.tensor([weight for _, _, weight in comparisons]))
            # chunks end where a right-aligned row starts: its padding's recurrent state is reset there
            cuts = sorted({0, prompt_rows, *firsts})
            bounds = sorted({*cuts, *(b for a, z in zip(cuts, cuts[1:]) for b in range(a, z, de.PREFILL_CHUNK_ROWS))})
            positions = torch.arange(prompt_rows, device=device)
            global SHIFTED
            SHIFTED = shifted
            try:
                last = self.prefill_rows(frames, firsts, bounds, positions)
            finally:
                SHIFTED = False
            graphs.start(last[:1], prompt_rows - 1, frames[0], request.temperature, request.top_k, request.top_p)
            graphs.pair_hidden[:rows].copy_(last)
            graphs.rows, graphs.shifted = rows, shifted
            log("guidance on " + " and ".join(
                f"the line's <tags> ({weight:g})" if first == 0 else f"the voice ({weight:g})"
                for _, first, weight in comparisons))
            return graphs

        def prefill_rows(self, frames, firsts, bounds, positions) -> torch.Tensor:
            """Prefill every row; returns each row's last hidden state [rows, H]."""
            backbone, tower = self.model.backbone, self.model.audio_decoder
            prompt_rows = len(frames[0])
            for start, end in zip(bounds, bounds[1:]):
                embeddings = torch.stack([tower.embed_backbone(f[start:end], backbone.embed(f[start:end, 0]))
                                          for f in frames])
                hidden = backbone(inputs_embeds=embeddings, input_pos=positions[start:end])
                paired_end = min(end, prompt_rows - 1)
                if paired_end > start:
                    tower.prefill(hidden[0, : paired_end - start], positions[start:paired_end],
                                  frames[0][start:paired_end], frames[0][start + 1: paired_end + 1])
                for row, first in enumerate(firsts, 1):
                    if first == end:  # the padding is done: this row's own prompt starts here
                        for layer in backbone.model.layers:
                            if layer.layer_type != "full_attention":
                                layer.linear_attn.conv_state[row].zero_()
                                layer.linear_attn.recurrent_state[row].zero_()
            return hidden[:, -1]

    GuidedGraphs.__name__ = base_graphs.__name__
    GuidedEngine.__name__ = de.Engine.__name__
    de.DecodeGraphs = GuidedGraphs
    de.Engine = GuidedEngine
    log(f"guidance on: <tags> {WEIGHT:g}; the voice {VOICE_WEIGHT:g} when every word is checked")
    return True
