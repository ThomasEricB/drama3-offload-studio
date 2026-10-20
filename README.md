# Drama 3 Offload Studio

A Pinokio launcher for [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3), built from the official [fishaudio/Drama-3](https://huggingface.co/spaces/fishaudio/Drama-3) Space.

To install it, open Pinokio, choose **Download from URL** and paste `https://github.com/ThomasEricB/drama3-offload-studio`.

It has two models (the full one also in two MLX versions for Apple Silicon), installed and started from the same menu:

| Menu | Model | GPU | Download |
| --- | --- | --- | --- |
| **Install / Start (24+ GB VRAM)** | the full Drama 3 (FP8), part of it in system RAM on 24-32 GB cards | NVIDIA 24 GB+ (on a Mac: the eager fallback engine, see below) | 41 GB |
| **Install / Start MLX 4-bit (Apple Silicon, 64 GB)** | the full Drama 3 as [drama-3-MLX-4bit-HQ](https://huggingface.co/fcmeyer/drama-3-MLX-4bit-HQ): experts 4-bit, the rest bfloat16; peaks at ~39 GB of unified memory | Apple Silicon, 64 GB (48 GB with a raised limit) | 26 GB |
| **Install / Start MLX 3-bit (Apple Silicon, 48 GB)** | the full Drama 3 as [drama-3-MLX-3bit-HQ](https://huggingface.co/fcmeyer/drama-3-MLX-3bit-HQ): experts 3-bit (the most sensitive layers 4-bit), attention and tower 8-bit; peaks at ~34 GB | Apple Silicon, 48 GB | 21 GB |
| **Install Small GGUF / Start (Small GGUF, 16 GB VRAM)** | [drama3-small](https://huggingface.co/pentacoxian-dev/drama3-small-GGUF): pruned to ~45% of the experts, ~3-bit, distilled on **Japanese and English** | NVIDIA 16 GB+ | 10 GB |

Both share the app and its Python environment, so the second model can be added at any time; Update re-checks only the models that are installed.

### Small GGUF (16 GB VRAM)

drama3-small is published for audio.cpp; here it runs in this Studio's own PyTorch engine, without audio.cpp (`offload/gguf_drama3.py`):

- The GGUF is read directly (memory-mapped). Its Q8_0 matrices stay int8 with their scales and its IQ3_XXS experts stay as raw 3-bit blocks on the GPU; they are expanded to bfloat16 just before use (a decode step expands only the 8 routed experts). Dequantization is bit-identical to gguf-py's reference.
- Each layer keeps its own number of experts (256 in 4 layers, 100-101 in the rest); the 4,097-row speech-only head plugs into the engine's speech head; the codec comes from the GGUF too. Reference clips are encoded on the CPU (once per voice, ~3 s for 20 s on a 16-thread CPU), so the GPU only holds the vocoding half of the codec.
- The pause fix is on (see below): without it, drama3-small went silent after the first sentence and never ended; with it, test lines came out complete.
- Sampling defaults to the model card's 0.7 / 0.7 / 30 in the Studio.

Measured on the RTX 5090, with PyTorch capped to simulate a card: weights 9.1 GiB; decode step 12.8 ms; 1.0-2.6x real time. Under a **12.5 GiB cap (a 16 GB card with a desktop)**, the worst-case set (retakes with long histories, Surprise retakes, a 5-chunk long script, also with 60 s references) ran with no out-of-memory error. Under a 10.5 GiB cap (12 GB card) it ran out of memory, so 12 GB cards are not supported. Speed on 16 GB cards was not measured; expect it to be slower (less memory bandwidth). Quality: complete, intelligible takes in Clone, Surprise and Design (Whisper word match 0.95-1.0 in English); other languages than Japanese and English are not what it was tuned for.

Drama 3 is a 35B-parameter MoE text-to-speech model (3B active per token) for expressive, multi-character speech in 80+ languages. The Space's Studio UI runs locally as **Drama 3 Offload Studio**, with the model loaded in the same process. Its header and status badge show the local device and how many expert layers are offloaded to RAM (the Space showed "ZeroGPU"):

- **Surprise**: the model picks a voice.
- **Clone**: use one of the three preset voices, upload a clip or record from the microphone. Whisper transcribes the clip on the CPU, and you can edit the transcript.
- **Design**: cast a voice from a written description.
- **Give a note**: re-take the same line with new acting direction. Earlier takes stay in the prompt.
- **Workflow**: a `gr.Workflow` canvas (at `/workflow/`) that uses the same engine.

Inline tags such as `[sighs]` and `[laughs]` place events at that point in the line.

## Requirements

| | Linux | Windows | macOS |
| --- | --- | --- | --- |
| Hardware | NVIDIA RTX 40-series or newer | NVIDIA RTX 40-series or newer | Apple Silicon, **experimental** (see below) |
| Driver / OS | NVIDIA driver 580+ | NVIDIA driver 580+ | macOS 14+ |
| Memory | RAM for offloaded experts (below) | RAM for offloaded experts (below) | MLX 4-bit: 64 GB; MLX 3-bit: 48 GB; eager fallback: 64 GB minimum, 96 GB+ recommended |
| Status | Tested (RTX 5090, 32 GB) | Untested: dependencies verified to resolve | Untested on a Mac. MLX: dependencies resolve and the models load, no take completed yet (see below); eager fallback: verified on Linux without CUDA |

- NVIDIA: RTX 40-series (Ada) or newer has the FP8 tensor cores the experts run on; RTX 30-series and older stop with a clear message. torch 2.13 is the CUDA 13 build, so the driver must be 580 or newer. With all experts in RAM the VRAM peak was about 14 GiB, so 16-24 GB cards should work, but only 32 GB was tested.
- System RAM (NVIDIA): 30 GiB of pinned RAM holds every expert (the VRAM holds copies of the most-used ones). Loading peaks at about 40 GiB: 64 GB recommended for 24 GB cards, 48 GB for 32 GB cards.
- Disk: about 41 GB for the checkpoint (`models/drama-3` in the launcher folder) plus about 10 GB for the Python environment.
- Not supported: AMD GPUs, Intel Macs (no PyTorch 2.13 build), CPU-only machines. Install stops on these with a message.

### Windows

Install takes torch 2.13.0+cu130 from the PyTorch index (PyPI's Windows torch is CPU-only) and `triton-windows` 3.7.1, the build of the Triton version torch 2.13 uses on Linux. `torch.compile` is on, as on Linux (triton-windows bundles its own C compiler). Small GGUF depends on it: its Q8_0 matrices and IQ3_XXS experts are expanded on every step, which compile fuses into a few kernels (measured on the 5090: one 8192x2048 Q8_0 matrix 0.3-0.55 ms uncompiled vs 0.03 ms compiled, the 8 routed experts of a layer 3.6 ms vs 0.08 ms), so uncompiled it runs several times slower than real time. The full model loses about 25% without it (24.3 instead of 18.3 ms per frame). If a layer still fails to compile on some machine, it runs uncompiled instead of failing the take, with the same output. Compiling on Windows is untested; Start takes longer while it compiles. The expert offload pins RAM the same way as on Linux (30 GiB); if Windows refuses to pin that much, Start stops with a message saying so.

### macOS: MLX

On Apple Silicon the menu offers the full model through [MLX](https://github.com/ml-explore/mlx), in two conversions by fcmeyer, run by [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s Drama 3 port (pinned to the commit they were published for, with `mlx` 0.32.3). Pick by memory:

| Menu | Model | Unified memory | Speed (M5 Max, their numbers) |
| --- | --- | --- | --- |
| **MLX 4-bit (64 GB)** | experts 4-bit, everything else bfloat16; 24.5 GB + 1.6 GB codec | peak ~39 GB: a 64 GB Mac, or 48 GB with `sudo sysctl iogpu.wired_limit_mb=42000` (until reboot; quit large apps first) | 2.3x real time |
| **MLX 3-bit (48 GB)** | experts 3-bit, 15 sensitive layers 4-bit, attention / tower 8-bit; 19.7 GB + 1.6 GB codec | peak ~34 GB: a 48 GB Mac | 2.7x real time |

In their 500-clip test both had the same failure rate and word error rate as unquantized MLX within the noise (4.6% and 3.6% of clips, vs 4.0%); the 4-bit is closer to bfloat16 token by token.

**Status: untested on a Mac, and not yet verified to produce a complete take.** On Linux, MLX's CPU and CUDA builds both load the models through this engine (the 4-bit as 24.5 GiB, the 3-bit as 20.1 GiB) and generate frames, but too slowly to finish a take there (the CPU build runs on one core at 7-11 s per frame; the CUDA build was still on its startup warm-up take after 4 minutes), so neither speed nor output on Apple Silicon is known. Please report how it behaves.

`offload/drama3_mlx.py` swaps only the model and codec: the Space's prompts, server and UI, and this Studio's pause fix, stall guard, long takes, lookahead, infill edits and subtitles are meant to work as on NVIDIA. The pause fix is applied inside each MLX step, so step n + 1 is still queued before step n is read; all MLX work runs on one thread of its own (MLX streams belong to the thread that made them).

The Start entries pass the model to `start.js` (`DRAMA3_BACKEND=mlx`, `DRAMA3_MLX_DIR`); **Start (24+ GB VRAM)** on a Mac still runs the eager PyTorch engine below on the official checkpoint, as the fallback.

### macOS: eager PyTorch fallback (experimental)

The Space's engine needs CUDA (Triton FP8 kernels, CUDA graphs). On a Mac without the MLX models, `offload/drama3_portable.py` replaces it with an eager engine on the Apple GPU (MPS) that keeps the Space's prompts, sampling, model code and codec:

- Decode runs step by step without CUDA graphs or `torch.compile`; vocoding runs on the same device.
- MPS has no FP8 type, so the experts stay as their raw bytes (30 GiB) and are converted per token through a 256-entry table and the checkpoint's block scales, the same values fish-speech's own dequantization produces.
- Weights take about 40 GB of unified memory, plus a few GB while generating.
- Timeouts are raised (first audio may take long), and the startup warm-up is skipped.

What was verified, on Linux with CUDA hidden so the code ran exactly as it would without CUDA, but on the CPU:

- The byte-stored experts dequantize **bit-identically** to fish-speech's own FP8 dequantization, on real checkpoint tensors.
- A full take through the engine came back as the right words ("Hello there! I did not expect you so soon.", transcribed by Whisper) and ended naturally.
- Through the real web server, with the same engine running eagerly on a GPU (no CUDA graphs, Triton or compile, byte-stored experts): Clone, Design, Surprise and a retake with a note all came back as the right words. On the CPU the server's between-chunk timeout (raised to 10 minutes) still runs out, so CPU mode is for debugging only.

**It has not run on a Mac.** Apple's MPS backend may lack an operation it uses: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so such an operation runs on the CPU instead of failing, which is slower. Speed on the Apple GPU is unknown and probably below real-time; on the CPU it is unusable (about 0.1 frames/s, real time is 21.5). Please report how it behaves.

### Expert offload (fits on a 32 GB GPU)

The Space loads all ~37 GiB of weights onto one GPU. This launcher adds `offload/drama3_offload.py`, which keeps the routed MoE experts in **pinned system RAM** and the **most-used ones in VRAM**. The experts are 30 GiB of FP8 weights: 40 layers of 256 experts, 768 MiB per layer. The Space's code in `app/` is not modified.

- Each offloaded weight is a CUDA tensor that aliases pinned host memory. `torch.compile`, the Triton FP8 kernels and CUDA-graph capture work unchanged, and the bytes are read over PCIe.
- A decode step gathers just the 8 experts each layer routes to (24 MiB per layer) into VRAM with a wide copy kernel (~50 GiB/s on PCIe 5 x16), then runs the original FP8 GEMM. The math is the same, on the same bytes.
- **Expert cache.** Routing is uneven: over 8 takes (clone, Surprise, whispers, anger), a layer's 64 most-used experts took 66% of its routings. So VRAM holds the most-used experts of every layer rather than all the experts of some layers, ranked by `offload/expert_profile.json` (how often each expert was chosen while decoding those takes). The gather reads a cached expert from VRAM and the others from RAM. With the same VRAM, 44% of the expert reads of other takes came from VRAM, instead of 17-40% with whole layers. Every layer keeps all its experts in pinned RAM as well (30 GiB, whatever the card), so the cache can be re-sized without reloading. Prefill (the prompt, which uses nearly every expert) still reads RAM in place: running the cached experts from VRAM there measured no faster.
- Measured on an RTX 5090 (PCIe 5 x16), same session, the same 6 takes; a 24 GB card was simulated by raising the reserve by 7 GiB (22.5 GiB free, as reported from an RTX 4090):

| VRAM for experts | Placement | Decode step | Speed (real takes) |
| --- | --- | --- | --- |
| 12 GiB (5090, 20 s references) | 16 whole layers (before) | 20.3-21.2 ms | 1.9-2.1x real time |
| 12 GiB (5090, 20 s references) | **4,157 most-used experts** (41% of all) | **15.2 ms** | **2.3-2.75x** |
| 9 GiB (5090, 60 s references) | 3,144 most-used experts | 17.6 ms | |
| 5.2 GiB (24 GB card) | 7 whole layers (before) | 22.5 ms | 1.7-1.9x |
| 5.2 GiB (24 GB card) | **1,772 most-used experts** | **20.0 ms** | **1.8-2.1x** |
| none | all 40 layers in RAM | 25.8 ms | 1.2-1.5x |

Whole layers measured 18.3 ms on an earlier day; the same-session numbers above are the comparison. The Space's own all-on-GPU placement (~37.5 GiB of weights) does not fit a 32 GB card. The decode step is measured once per start and varies by about 1 ms.

Real time is 46 ms per frame (21.5 frames/s). A slower PCIe link (Gen 4, or x8) makes the offloaded layers proportionally slower.

Settings live in the `env` block of `start.js`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DRAMA3_OFFLOAD_LAYERS` | `auto` | `auto` fills the VRAM left after the reserve with the most-used experts. A number keeps that many whole layers in RAM and the rest on the GPU (the placement before the cache). `0` gives the Space's original all-on-GPU behaviour (needs ~45 GB). |
| `DRAMA3_VRAM_RESERVE_GB` | `9` | VRAM that `auto` leaves free with 20 s references. The engine's working memory peaked at ~6.5 GiB (20 s reference, 2-take histories of long takes, Surprise retakes, long scripts); the worst case left 3.0 GiB free on the 5090. |
| `DRAMA3_VRAM_RESERVE_60S_GB` | `12` | The same with 60 s references unlocked: working memory peaked at ~8.7 GiB; the worst case left 3.7 GiB free. |

The start log reports the split it chose, for example `[offload] ... 0 of 40 layers fit, plus the 4157 most-used experts of the other 40 (with 60 s references: 0 layers + 3133 experts)`.

**torch.compile settings per take.** The Space applies its compile settings (`emulate_precision_casts`, a recompile limit of 32) once, in the thread that builds the engine; in this torch they are per thread and every take runs in a new one. `offload/serve.py` applies them before every take, so graphs compiled later (the 4096-row decode graph long prompts need, now also captured at startup) are built the same way.

### Pause fix

Drama 3 sometimes goes silent at a tag or sentence boundary and never resumes or stops. The fix published with [drama3-small](https://huggingface.co/pentacoxian-dev/drama3-small-GGUF) (there for audio.cpp) is ported to this Studio's own engine (`offload/pause_fix.py`), for both models, on NVIDIA inside the CUDA-graph decode step and in the experimental Mac engine: after 10 silent frames in a row (0.47 s; a frame is silent when its first audio code is one of the 143 codes in `silence_codes.json`), silence is masked from the next draw, so the model must resume speaking or end; and a length guard estimated from the text ends a take that pauses after twice its expected length, or reaches three times it. The model also pauses in silence codes broken every few frames by a quiet code that is not on the list, which never makes 10 in a row: a pause is therefore also cut when 10 of the last 20 frames are silent or one of 7 such quiet codes (`[clears throat] Senhoras e senhores... [hesitates]` stalled in 7 of 8 takes before, 0 of 8 after; normal speech kept the same words, length and pauses). Install downloads the 813-byte codes file. Tested on the full model: normal lines, slow "dramatic pauses" retakes and a 72 s long script all kept every word (53/53, 53/53, 196/196). `DRAMA3_MAX_PAUSE_FRAMES` in `start.js` (default `10`; `0` turns it off).

### Stall guard

Now and then a take derails after its first words into a low, steady drone that never ends. The model cycles through a few dozen quiet codes that are not among the pause fix's silence codes, so the take ran on to its length cap (16-19 s of drone after "I told you."), and the drone also happens with the pause fix off. `offload/stall_guard.py` watches the main codes, an idea from T8star's [ComfyUI-Drama3-T8](https://github.com/T8mars/Comfyui-Drama-3):

- **Holding back:** while the last 24 frames have 18 or fewer distinct codes, new frames are held back (not yet turned into audio).
- **Ending the take:** when the last 64 frames have 30 or fewer, the take ends where the hold began, so the drone is never heard.
- **Retrying:** a drone at the start falls inside the server's 2.5 s opening check, which retries once. A long script's later chunks are held until complete (they are made at about 2x real time while the chunk before plays) and a stalled one is made again.

The limits were fitted on 96 takes of the full model (4 lines x 12 seeds x 2 sampling settings):
- **Normal speech:** never fewer than 40 distinct codes in any 64 frames, so 0 false stalls. Holds happen on 1% of frames, for at most 0.8 s.
- **The 3 derailed takes:** fell to 22-23 and are cut after about 1.5 s of audio instead of 18 s.

The limits were not re-fitted for the Small GGUF. `DRAMA3_STALL_GUARD` in `start.js` (`0` = off).

### Tag guidance (`<whisper>…</whisper>` and every other `<tag>`)

A cloned voice copies the delivery of its reference clip, and a loud, energetic clip can outweigh the script's own tags: `<whisper>…</whisper>` came out voiced in nearly every take with such a clone, while random voices whispered. On NVIDIA (full model), `offload/tag_guidance.py` makes the model follow `<…>` markup anyway:
- **How:** for a line with `<…>` markup, each decode step runs the line twice in one batch, as written and with its tags blanked, and pushes the next frame further in the direction the tags move it (classifier-free guidance, weight 4). Only audio tokens the tagged line itself finds plausible are boosted, and end of speech is left as the tagged line predicts it, so words are not dropped.
- **Measured** with a clone that resisted tags (8-12 takes per line): `<whisper>` 0.65 to 0.25 of the audio voiced (English) and 0.21 to 0.06 (Portuguese), every word kept; a whispered span inside a voiced line whispered in 9 of 10 takes instead of 2; `<crying>` pitch 234 to 298 Hz; `<shouting>` 261 Hz against 217 Hz untagged.
- **Only cloned voices:** random and designed voices already follow tags, and guidance there amplified the tags' side effects too (a whispered span after a `Direction:` line was skipped in 10 of 10 takes instead of 4 of 10). With the clone, that same directed line came out complete in 9 of 10 takes instead of 6, the span whispered and the rest voiced.
- **Cost:** only for lines with `<…>`: about 0.75 s per second of audio on an RTX 5090 instead of 0.5 s, and 0.3 GiB more VRAM. Lines without `<…>` run exactly as before; `[cues]` are not affected.

`DRAMA3_TAG_GUIDANCE` (default `4`; `0` = off). Not yet measured with the Small GGUF, so it is off there.

### Every word read, while streaming

The model sometimes loses its place in a line: it skips a span (more often a tagged one), stops before the end, or reads a `Direction:` aloud. `offload/word_check.py` and `offload/long_takes.py` catch these without giving up streaming:
- **Directed first takes:** `Direction: …` / `Line: …` is Drama 3's directing format, learned as a conversation about an earlier take (the Space uses it so for retakes). On a first take the model often lost its place in it, so a first take is sent as the line under a `[style prefix]` instead, which Drama 3 never speaks; retakes keep the `Direction:` form.
- **Opening check:** the server holds a take's first 2.5 s before any audio goes out anyway; within that, the first 2 s are transcribed and must match the start of the line, or the take starts again (at most twice). A spoken direction or a skipped opening span scored 0.53 or less, good openings mostly 0.88-1.0.
- **Ending check:** when a take ends, it is transcribed, and if its last words are missing (two or more, or any after a stall) they are made as a continuation (the take so far as context, as a long script's next chunk) and stream after a short pause.
- **Whisper on the GPU** for these checks (a 2 s opening in ~0.08 s instead of ~2 s on the CPU, ~1.5 GB of the VRAM reserve), loaded in the background at Start; without CUDA Whisper they use the CPU one.

`Direction: Donald Trump is having a chat with Elizabeth Holmes from Theranos` / `Line: <whispers> Okay... Alright... Good chat. </whispers> Are you being real, right now?`, streamed: complete in 11 of 12 takes with a random voice (1 of 12 before) and 10 of 10 with a clone (3 of 10 before), the span whispered and the rest voiced; first audio after ~1.3-2.9 s with a random voice when no restart is needed.

**Check every word before playing** (Studio checkbox in its own box between Voice and Sampling, off by default; `DRAMA3_WORD_CHECK=1` for the API): each take, and each chunk of a long script, is held until made and its whole transcript compared with the line (below 90 % it is made again, at most twice, and the best is sent). Every word is checked, but a take plays once made instead of streaming.

### Voice guidance (with “Check every word before playing”)

Since those takes are not streamed, they can afford a slower, closer clone. On NVIDIA (full model), `offload/tag_guidance.py` guides a cloned voice towards its clip the way tag guidance does for `<tags>`: each step also runs the prompt **without** its reference clip (the prompt Drama 3 uses for a voice of its own) and pushes the next frame away from it, at strength 2 (the **Voice guidance** slider folded under the checkbox, like Sampling, 0-3; `DRAMA3_VOICE_GUIDANCE` for the API; 0 = off). This is the "keep the text, drop the voice prompt" form of classifier-free guidance (Selective CFG, arXiv 2509.19668); 2 measured best (1 and 3 were tried), as in SemBridge (arXiv 2608.07462).
- **Measured** on two clean Trump clips through the Studio path (16 takes per voice and setting, 4 lines in English and Portuguese): speaker similarity (ECAPA) 0.445 to 0.583 and 0.616 to 0.680; on a Trump recording the model never saw, 0.476 to 0.590; a second speaker model (x-vector) +0.020 and +0.002; median pitch 5.0 to 3.1 and 1.6 to 1.2 semitones from the clip's. Every word kept.
- **Tags too:** a line with `<tags>` gets both guidances (three rows per step); the Holmes line's `<whispers>` span stayed whispered in 6 of 6 takes, the rest voiced.
- **Long scripts and clone retakes** are guided chunk by chunk / take by take. **Surprise and Design retakes are not:** their voice reference is the earlier take, which their history holds too, and guidance on that small difference ended the line after its first sentence (6 of 8 takes; 1 of 4 unguided).
- **The clip's background comes along:** a clip with music under the voice gets that music louder in its takes, and twice in 48 takes it ran on for 17 s after the line. A held take with more than 4 s after its last word (as Whisper places it) is cut 0.4 s after it; in 358 test takes that cut only such runaways (5-17 s), never a word. A clean clip works best.
- **Cost:** about 1.5x the time per take (a short line ~5.5 s instead of ~3.8 s; a 40 s script 46 s instead of 21 s). A tagged line with voice guidance allocates a three-row cache set on first use (0.77 GiB, kept). Streamed takes and takes with the slider at 0 run exactly as before (seeded takes are sample-identical).

### 20 s and 60 s references

60 s references need about 2.2 GiB more working memory, so the expert split follows the **voice each take uses**: on the 5090, the 4,157 most-used experts stay on the GPU for takes with a voice of 20 s or less, and 3,144 for takes with a longer clone voice.

- **Allow reference clips up to 60 s** (Clone panel) only decides how long *new* clips may be. Clips added while it is on are marked **“· 60 s”** in the voice list and keep working after you untick it.
- A take (or retake, or edit) with a “· 60 s” voice switches to the 60 s layout; the next take with a shorter voice switches back to the faster one. The switch happens between takes, in its own lock: it drops the decode CUDA graphs (they hold the experts' addresses), re-sizes the expert cache (3 GiB) from pinned RAM and captures the graphs again, ~1 s (5 s the first time). Nothing moves while consecutive takes need the same layout. Tested with the cache: 5 takes alternating a 38 s and a 10 s voice re-sized it 4 times, exactly at the changes, all takes complete.
- The checkboxes and the hint show the server's current settings on every page load (also after switching in another tab).
- A fixed `DRAMA3_OFFLOAD_LAYERS` turns the switching off. Small GGUF needs no switching (60 s references fit 16 GB).

### Other GPUs (estimates)

These estimates are for whole layers in VRAM, from before the expert cache, which measured 10% (24 GB) to 27% (32 GB) faster on the 5090 (above); with the cache every card pins all 30 GiB of experts in RAM. Only the RTX 5090 was measured. The table applies the measured numbers (7.5 GiB of dense weights and codec, 0.75 GiB per expert layer, 6.5 / 8.7 GiB working memory, ~2.4 GiB taken by the desktop) and scales speed: the decode step is 6.9 ms of compute on the 5090 (scaled by memory bandwidth for other GPUs, as batch-1 decoding is bandwidth-bound) plus 0.47 ms per expert layer in RAM on PCIe 5.0 x16 (about twice that on PCIe 4.0 x16 or 5.0 x8). The model reproduces the 5090's measured 18.3 / 20.2 ms. Real time is 46.5 ms per frame; "x" is how much faster than real time a take is generated.

| GPU | 20 s references | 60 s references | RAM for experts |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell (96 GB) | all 40 layers on the GPU, ~5x | same | none |
| RTX PRO 5000 Blackwell (48 GB) | 37 layers, ~3.5x | 33 layers, ~2.9x | 2 / 5 GiB |
| RTX 6000 Ada / L40S (48 GB) | 37 layers, ~2.1-2.3x | 33 layers, ~1.8-1.9x | 2 / 5 GiB |
| **RTX 5090 (32 GB, measured)** | **16 layers, ~2x** | **12 layers, ~1.8x** | 18 / 21 GiB |
| RTX PRO 4500 Blackwell (32 GB) | 16 layers, ~1.5x | 12 layers, ~1.4x | 18 / 21 GiB |
| RTX 5000 Ada (32 GB) | 16 layers, ~0.8x | 12 layers, ~0.8x | 18 / 21 GiB |
| RTX 4090 (24 GB) | 6 layers, ~0.8x | 2 layers, ~0.8x | 26 / 28 GiB |
| 16 GB cards (5080, 5070 Ti, 4080, 4070 Ti Super, 5060 Ti, 4060 Ti) | the full model does not fit with a desktop: use **Small GGUF** | does not fit | |
| 12 GB and less | does not fit | does not fit | |

Below 1x, a take is generated more slowly than it plays: the stream catches up by buffering, and long takes take longer than their length. Every card also needs ~40 GB of free RAM while loading (64 GB recommended) and an RTX 40-series or newer GPU (FP8). Start logs the split it chose and warns when even all-experts-in-RAM leaves less than the reserve.

## How to use

1. **Install** clones the Space into `app/`, installs `requirements.txt` into `app/env` (torch 2.13.0: the CUDA 13 build on Linux and Windows, the MPS build on macOS) and downloads the checkpoint at the revision the engine pins into `models/drama-3`. Start loads it from there and never downloads: if files are missing it stops and asks you to run Install.
2. **Start** loads Drama 3 on the CPU, moves the planned experts to pinned RAM and everything else to the GPU (about a minute), then runs one warm-up take to compile kernels and capture CUDA graphs. **Open Web UI** appears once the server is listening; the warm-up finishes in the background about 30 s later. On a Mac the weights move to the Apple GPU instead and there is no warm-up.
3. Write a line, choose a voice mode and press generate. Audio streams while it is generated.

Other optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DRAMA3_WARM_UP` | `1` | `0` skips the startup warm-up take |
| `DRAMA3_COMPILE` | `1` | `0` turns off regional `torch.compile` (NVIDIA only) |
| `DRAMA3_BACKEND` | `auto` | `cuda` (NVIDIA engine + offload), `mlx` (Apple Silicon; set by the MLX Start entries) or `portable` (eager engine); `auto` picks `cuda` when available, else `portable` |
| `DRAMA3_MLX_DIR` | set per MLX entry | The MLX model folder: `models/drama-3-mlx-4bit` or `models/drama-3-mlx-3bit` |
| `DRAMA3_DEVICE` | `mps` | Device of the portable engine: `mps` or `cpu` (CPU works but is far too slow for real use) |
| `WHISPER_MODEL` | `turbo` | faster-whisper model for transcribing reference clips |
| `TAKE_TIMEOUT_S` | `180` | Per-take timeout |

Sessions (uploaded voices, takes and directing history) are held in memory and are lost on restart.

### Long scripts, edits and 60 s references

The Space's limits (600 characters, 20 s references) were there to spare Hugging Face's shared GPUs. Locally:

- **No script length limit.** A script longer than about 300 bytes is split into chunks: at sentence ends, then clause marks, then spaces, never inside `[tags]` or `<tag>…</tag>`. The chunks are generated one after another and play as one take. Each chunk sees the first chunk and the latest one as history (their text and audio codes), the layout Fish Audio uses in fish-speech, so the voice stays the same from chunk to chunk. Each chunk is limited to about 28 s of audio. Each chunk is a separate generation that the model ends right on its last syllable, so a pause is added at every join: 0.45 s after a sentence end, 0.25 s after a clause mark, 0.12 s otherwise, counting the silence the chunk already ends with (measured joins: 0.4-0.5 s, like the model's own pauses between sentences; without it they were 0-0.1 s, and the next line followed abruptly). Tested: a 1,034-character story played as 5 chunks, 51-80 s long, at 1.9-2.1x real-time, with a 0.98-0.995 word match against the script in Clone, Surprise and Design. The counter under the script shows the chunk count. (`offload/chunking.py`, `offload/long_takes.py`)
- **Read ahead across chunks.** Each chunk is voiced knowing the next one: Drama 3's streaming-input format, from the [technical report](https://github.com/fishaudio/fish-speech/blob/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf)'s Figure 10, where the chunk's turn is followed by `<|endofturn|>` and the next chunk. Results on a 5-chunk story, 6 seeds per voice:
  - Clone voice: the pitch difference across a join fell from 3.35 to 2.29 semitones on average, and the spread of pitch between chunks from 1.28 to 0.92.
  - Surprise voices: no change.
  - Words: unchanged (0.99 match either way); takes were about 6% brisker.

  The **Read ahead across chunks** checkbox under the stage (`DRAMA3_LOOKAHEAD`), on by default.
- **Edit by infilling.** Leave the note empty, change some words of the script and press **Retake**. Only the changed words are regenerated, and the audio before and after them is kept. This is Drama 3's own editing method (report section 5.5 and Figure 14):
  - Whisper word timestamps locate the edit window.
  - The model fills it from a prompt holding the new text, a duration hint, and the audio after and before the window.
  - The whole clip is decoded together, so the seams are smooth.
  - Whisper then checks that the new words are heard (at least half of their letters) and that the clip matches its script as well as before. Otherwise it retries once, then falls back to regenerating the whole chunk.
  - Numbers are compared spelled out, in English, on both sides: "7" matches "seven", "$5" "five dollars", "50%" "fifty percent", "21st" "twenty-first", "1999" "nineteen ninety-nine".

  It works for single lines and inside long takes; in a 5-chunk take, a one-word insertion took 11.5 s.
  - Tested on 16 edits x 2 voices (substitutions, insertions, deletions, at the start too): 14/16 infilled on the first attempt, with a word match of 0.994. The other 2 fell back, because Whisper wrote "seven" as "7"; numbers are now compared spelled out, which fixes that.
  - Each infill edit runs Whisper twice on the CPU: a few seconds on a short line.
  - The **Edit by infilling** checkbox (`DRAMA3_INFILL`). With it off, an empty-note Retake re-reads the edited script with the earlier takes as context, as before.
- **Retake / Edit long takes (experimental).** Tick the box under the stage to retake or edit a take made of several chunks. With an empty note, only the chunks whose words changed are regenerated; the others keep their exact audio, and the line under the box says which chunk(s) will be redone before you press Retake. A one-word edit in a 5-chunk take regenerated 1 chunk in 7 s instead of 27. With a note, every chunk is redone, each guided by its old version. Off by default.
- **Subtitles.** Two buttons under the stage:
  - **Import subtitles (.srt)** turns an .srt file into a timed script, one cue per line, written `{00:01.200 --> 00:03.500} Text`. Each cue is voiced as its own chunk with Drama 3's duration hint (`[audio tokens: N]`, about 21.5 per second) and placed at its start time. Silences between cues are kept, and a cue that runs over pushes the next one back. Tested: three cues at 0.5, 5.0 and 10.0 s started at 0.55, 5.05 and 10.05 s. Timed lines can be edited like any script, cue by cue.
  - **Subtitles of this take (.srt)** writes the subtitles of the selected take: its script, timed by Whisper on its audio, one cue per sentence (at most 84 characters or 6 s).
- **Language and duration hints.** `[language: fr]` names a line's main language, and `[audio tokens: 65]` asks for about 3 s (report Table 10). Neither is spoken.
- **Reference clips up to 60 s.** Clips are trimmed to 20 s by default, as on the Space. Tick **Allow reference clips up to 60 s** (Clone panel) to keep up to the engine's 60 s limit (see *20 s and 60 s references* below). With the default 12 GiB reserve, a 59 s reference, two retakes and a long script left at least 3.7 GiB of VRAM free.

`DRAMA3_REF_UNLOCKED` and `DRAMA3_LONG_RETAKE` in `start.js` set the initial state of those two checkboxes (`1` = on).

**Longer reference clips.** A clone reference over 20 s is cut in a pause between words. The Space cut at the *quietest* moment between 12 and 20 s, which often kept only 12-15 s. `offload/reference_trim.py` cuts at the *latest* real pause instead: on test clips 17-20 s are kept (14.5 s before on average), still ending on a whole word or sentence.

**One take at a time.** On Hugging Face every take ran in its own GPU worker. Here all takes share one engine, and two at once would corrupt each other's audio. So `offload/serve.py` limits the server to one take (`DEMO_MAX_ACTIVE=1`); further takes wait in the queue, which the UI shows. It also holds a lock so a take started during the startup warm-up waits for it to finish.

**Versions are pinned.** `offload/` hooks into the Space's engine internals, so Install checks out the Space commit it was tested with (`c368729`, which in turn pins checkpoint revision `e0a9218`); Update pulls the launcher and reruns Install. To move to a newer Space, change the commit in `install.js` and test again.

**Reset** deletes `app/` (code and Python environment). The checkpoint in `models/drama-3` (about 39 GB) is kept so a reinstall doesn't download it again; delete that folder to reclaim the space.

## API

The app is a FastAPI server with Gradio mounted at `/` and `/workflow/`. The simplest programmatic path is its own JSON API:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | Engine phase, the preset voices, recent latency |
| `POST` | `/api/voices` | Multipart `audio` upload → registers a clone voice and returns `{id, transcript, …}` |
| `POST` | `/api/generate` | Generate a take. The response is a `text/event-stream` |
| `GET` | `/api/takes/{take_id}.wav` | The finished take as WAV |
| `GET` | `/api/takes/{take_id}.srt` | Subtitles of the take (its script timed by Whisper) |

`POST /api/generate` body:

```json
{
  "text": "[sighs] Fine. If we miss it, it's on you.",
  "voice": { "mode": "clone", "id": "clone1" },
  "temperature": 1.0, "top_p": 0.9, "top_k": 0
}
```

- `voice.mode` is `"clone"` (with `id` set to a preset `clone1`/`clone2`/`clone3` or an id returned by `/api/voices`, and an optional `transcript`), `"design"` (with `name` and `description`, plus an optional `scene`) or `"random"`.
- To re-take a line with a director's note, send `"retake_of": "<take_id>", "note": "angrier, faster"`. With `"note": ""` it is an edit: send the changed `text`. The changed words are infilled, or, if that is not possible, the line is read with the earlier takes as context.
- `text` has no length limit; long scripts are chunked and stream as one take. A text whose lines start with `{start --> end}` (as **Import subtitles** writes them) is a timed script.
- Each SSE event is a `data: {json}` line with a `type`: `queued`, `start`, `audio` (base64 16-bit PCM in `pcm`), `retrying`, `error` or `done` (which carries `take_id`).

Replace `http://127.0.0.1:PORT` below with the URL shown in Pinokio.

### curl

```bash
curl -N -X POST http://127.0.0.1:PORT/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there. [laughs] I did not expect you so soon.","voice":{"mode":"clone","id":"clone1"}}'
# read take_id from the final "done" event, then:
curl -o take.wav http://127.0.0.1:PORT/api/takes/TAKE_ID.wav

# clone your own voice
curl -F audio=@me.wav http://127.0.0.1:PORT/api/voices
```

### Python

```python
import json, requests

BASE = "http://127.0.0.1:PORT"

def speak(text, voice, out="take.wav"):
    with requests.post(f"{BASE}/api/generate", json={"text": text, "voice": voice}, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "error":
                raise RuntimeError(event["message"])
            if event["type"] == "done":
                wav = requests.get(f"{BASE}/api/takes/{event['take_id']}.wav").content
                open(out, "wb").write(wav)
                return event

speak("Good evening. [whispers] They're listening.", {"mode": "clone", "id": "clone2"})
speak("Ahoy! Who goes there?", {"mode": "design", "name": "Captain",
      "description": "a gravelly old sea captain, booming and theatrical"})
```

### JavaScript

```javascript
const BASE = "http://127.0.0.1:PORT";

async function speak(text, voice) {
  const res = await fetch(`${BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice }),
  });
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "done") return `${BASE}/api/takes/${event.take_id}.wav`;
    }
  }
}

speak("I told you this would happen.", { mode: "random" }).then(console.log);
```

The Gradio UI's own endpoints are also available through `gradio_client` (`Client("http://127.0.0.1:PORT/")`, then `client.view_api()`), but the JSON API above is what the Studio itself uses.

## License

The Space code is Apache-2.0. The vendored fish-speech code keeps its own license (`app/fish_speech/LICENSE`). The model weights are under the [Fish Audio Community / Research License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE). Only clone voices you have permission to use.
