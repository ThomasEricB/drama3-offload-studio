"""Fish Audio Drama 3 from a drama3-small GGUF, in PyTorch (no audio.cpp).

drama3-small (pentacoxian-dev/drama3-small-GGUF) is Drama 3 with its routed experts
pruned to 100-256 per layer and quantized to IQ3_XXS, the other matrices in Q8_0, and a
4,097-row speech-only output head. Its names are the original checkpoint's, so the
Space's own model code runs it; this module only has to read the GGUF and keep the
weights compact on the GPU:

- Q8_0 matrices stay int8 with one fp16 scale per 32 values (Q8Linear, Q8Embedding)
  and are expanded to bfloat16 just before each matmul / for the looked-up rows.
- IQ3_XXS experts stay as their raw 98-byte blocks (IQ3Experts). A decode step expands
  only the 8 routed experts; prefill expands each expert it hits, one at a time.
- Dequantization follows ggml (and gguf-py's reference, which tests check against).

All of it fits a 16 GB card: ~9.7 GiB of weights instead of ~12.5 GiB in bfloat16.
"""

from __future__ import annotations

import copy
import json
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ggml type id -> (name, bytes per block, values per block)
GGML_TYPES = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 8: ("Q8_0", 34, 32), 18: ("IQ3_XXS", 98, 256), 30: ("BF16", 2, 1)}
_SCALARS = {0: "u1", 1: "i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 7: "?", 10: "<u8", 11: "<i8", 12: "<f8"}


# ─── GGUF file ────────────────────────────────────────────────────────────────


@dataclass
class TensorInfo:
    name: str
    type: str
    shape: tuple[int, ...]  # logical, outermost first (PyTorch order)
    offset: int
    nbytes: int


class GGUF:
    """A memory-mapped GGUF: metadata, embedded files and tensors by checkpoint name."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.pos = 0
        if bytes(self.mm[:4]) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        self.pos = 4
        version, n_tensors, n_kv = self._read("<I"), self._read("<Q"), self._read("<Q")
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        self.meta = {}
        for _ in range(n_kv):
            key = self._string()
            self.meta[key] = self._value(self._read("<I"))
        raw = []
        for _ in range(n_tensors):
            name = self._string()
            dims = [self._read("<Q") for _ in range(self._read("<I"))]
            raw.append((name, dims, self._read("<I"), self._read("<Q")))
        alignment = int(self.meta.get("general.alignment", 32))
        data_start = (self.pos + alignment - 1) // alignment * alignment
        # audio.cpp stores checkpoint names and logical shapes in metadata
        names = self.meta.get("audiocpp.tensor_names") or [name for name, *_ in raw]
        ranks = self.meta.get("audiocpp.tensor_ranks")
        flat = self.meta.get("audiocpp.tensor_shapes")
        self.tensors: dict[str, TensorInfo] = {}
        cursor = 0
        for index, (stored, dims, type_id, offset) in enumerate(raw):
            if type_id not in GGML_TYPES:
                raise ValueError(f"{stored}: unsupported ggml type {type_id}")
            type_name, block_bytes, block_values = GGML_TYPES[type_id]
            count = int(np.prod(dims))
            if ranks is not None:
                shape = tuple(int(v) for v in flat[cursor:cursor + int(ranks[index])])
                cursor += int(ranks[index])
            else:
                shape = tuple(reversed(dims))
            name = str(names[index])
            name = name.split("/", 1)[1] if "/" in name else name
            group = "codec" if str(names[index]).startswith("codec_weights/") else "model"
            self.tensors[f"{group}:{name}"] = TensorInfo(
                name, type_name, shape, data_start + offset, count // block_values * block_bytes)

    # Header reading ---------------------------------------------------------------

    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        value = struct.unpack(fmt, bytes(self.mm[self.pos:self.pos + size]))[0]
        self.pos += size
        return value

    def _string(self) -> str:
        length = self._read("<Q")
        value = bytes(self.mm[self.pos:self.pos + length]).decode("utf-8")
        self.pos += length
        return value

    def _value(self, kind: int):
        if kind == 8:
            return self._string()
        if kind == 9:
            inner, count = self._read("<I"), self._read("<Q")
            if inner in _SCALARS:  # numeric arrays (incl. 28 MB of embedded files): one slice
                dtype = np.dtype(_SCALARS[inner])
                values = np.frombuffer(bytes(self.mm[self.pos:self.pos + count * dtype.itemsize]), dtype=dtype)
                self.pos += count * dtype.itemsize
                return values
            return [self._value(inner) for _ in range(count)]
        dtype = np.dtype(_SCALARS[kind])
        value = np.frombuffer(bytes(self.mm[self.pos:self.pos + dtype.itemsize]), dtype=dtype)[0]
        self.pos += dtype.itemsize
        return value.item()

    # Contents ---------------------------------------------------------------------

    def embedded_file(self, name: str) -> bytes:
        names = list(self.meta["audiocpp.embedded_files.names"])
        offsets = self.meta["audiocpp.embedded_files.offsets"]
        index = names.index(name)
        return self.meta["audiocpp.embedded_files.data"][int(offsets[index]):int(offsets[index + 1])].tobytes()

    def info(self, name: str, group: str = "model") -> TensorInfo:
        return self.tensors[f"{group}:{name}"]

    def raw(self, info: TensorInfo) -> np.ndarray:
        return self.mm[info.offset:info.offset + info.nbytes]

    def names(self, group: str = "model") -> list[str]:
        return [key.split(":", 1)[1] for key in self.tensors if key.startswith(group + ":")]


# ─── Dequantization ───────────────────────────────────────────────────────────


def _iq3_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """IQ3_XXS lookup tables from gguf-py (the 256x4 magnitude grid, 128 sign patterns)."""
    from gguf.quants import IQ2_XXS, IQ3_XXS

    IQ3_XXS.init_grid()
    grid = torch.from_numpy(np.ascontiguousarray(IQ3_XXS.grid.reshape(256, 4))).to(device)
    signs = torch.frombuffer(bytearray(IQ2_XXS.ksigns), dtype=torch.uint8).to(device)
    return grid, signs


_TABLES: dict = {}


def iq3_tables(device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(grid, sign patterns, sign shifts, bit positions) on device; built once per device."""
    key = str(device)
    if key not in _TABLES:
        grid, signs = _iq3_tables(device)
        _TABLES[key] = (grid, signs, torch.tensor(_SIGN_SHIFTS, device=device, dtype=torch.int32),
                        torch.arange(8, device=device, dtype=torch.int32))
    return _TABLES[key]


_SIGN_SHIFTS = (0, 7, 14, 21)


def dequantize_iq3_xxs(blocks: torch.Tensor, grid: torch.Tensor, ksigns: torch.Tensor,
                       shifts: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """IQ3_XXS blocks [..., 98] (uint8) -> float32 [..., 256], as ggml / gguf-py."""
    lead = blocks.shape[:-1]
    b = blocks.reshape(-1, 98)
    n = b.shape[0]
    d = b[:, :2].contiguous().view(torch.float16).float()  # [n, 1]
    qs = b[:, 2:66].long()  # 64 grid indices
    aux = b[:, 66:98].contiguous().view(torch.int32)  # 8 words: 4 x 7 sign bits + 4 scale bits
    db = d * (0.5 + ((aux >> 28) & 0xF).float()) * 0.5  # [n, 8]
    pattern = ksigns[((aux.unsqueeze(-1) >> shifts) & 0x7F).long()].int()  # [n, 8, 4]
    bits = (pattern.unsqueeze(-1) >> positions) & 1  # [n, 8, 4, 8]
    sign = 1.0 - 2.0 * bits.float()
    magnitude = grid[qs].reshape(n, 8, 4, 8)
    return (db.reshape(n, 8, 1, 1) * magnitude * sign).reshape(*lead, 256)


def split_q8_0(raw: np.ndarray, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Q8_0 bytes -> (int8 values [*shape], fp16 scales [*shape[:-1], cols/32])."""
    blocks = np.asarray(raw).reshape(-1, 34)
    scales = torch.from_numpy(np.ascontiguousarray(blocks[:, :2]).view(np.float16).reshape(*shape[:-1], shape[-1] // 32))
    values = torch.from_numpy(np.ascontiguousarray(blocks[:, 2:]).view(np.int8).reshape(shape))
    return values, scales


def expand_q8_0(values: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cols = values.shape[-1]
    grouped = values.reshape(*values.shape[:-1], cols // 32, 32).float() * scales.float().unsqueeze(-1)
    return grouped.reshape(values.shape).to(dtype)


def dense(gguf: GGUF, info: TensorInfo, dtype: torch.dtype) -> torch.Tensor:
    """Any tensor as a dense CPU tensor of dtype (for small tensors and tests)."""
    raw = gguf.raw(info)
    if info.type == "F32":
        return torch.from_numpy(np.asarray(raw).view(np.float32).reshape(info.shape).copy()).to(dtype)
    if info.type == "F16":
        return torch.from_numpy(np.asarray(raw).view(np.float16).reshape(info.shape).copy()).to(dtype)
    if info.type == "Q8_0":
        return expand_q8_0(*split_q8_0(raw, info.shape), dtype)
    if info.type == "IQ3_XXS":
        blocks = torch.from_numpy(np.asarray(raw).reshape(*info.shape[:-1], info.shape[-1] // 256, 98).copy())
        return dequantize_iq3_xxs(blocks, *iq3_tables("cpu")).reshape(info.shape).to(dtype)
    raise ValueError(info.type)


# ─── Compact modules ──────────────────────────────────────────────────────────


class Q8Linear(nn.Module):
    """nn.Linear (no bias) over Q8_0 weights, expanded to the compute dtype per call."""

    def __init__(self, values: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> None:
        super().__init__()
        self.out_features, self.in_features = values.shape
        self.register_buffer("qweight", values)
        self.register_buffer("qscale", scales)
        # The model reads .weight for its dtype / device (cache allocation) only
        self.register_buffer("weight", torch.empty(0, dtype=dtype, device=values.device), persistent=False)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, expand_q8_0(self.qweight, self.qscale, x.dtype))


class Q8Embedding(nn.Module):
    """nn.Embedding over Q8_0 rows: only the looked-up rows are expanded."""

    def __init__(self, values: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = values.shape
        self.dtype = dtype
        self.register_buffer("qweight", values)
        self.register_buffer("qscale", scales)
        self.register_buffer("weight", torch.empty(0, dtype=dtype, device=values.device), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return expand_q8_0(self.qweight[ids], self.qscale[ids], self.dtype)


class IQ3Experts(nn.Module):
    """Routed experts as raw IQ3_XXS blocks: gate / up [E, 512, 2048], down [E, 2048, 512]."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.num_experts, self.hidden, self.intermediate, self.fp8 = gate.shape[0], hidden, intermediate, False
        self.register_buffer("gate_blocks", gate)  # [E, I, H/256, 98]
        self.register_buffer("up_blocks", up)
        self.register_buffer("down_blocks", down)  # [E, H, I/256, 98]
        for name, table in zip(("grid", "ksigns", "shifts", "positions"), iq3_tables(gate.device)):
            self.register_buffer(name, table, persistent=False)

    def expert_weights(self, index, dtype: torch.dtype):
        tables = (self.grid, self.ksigns, self.shifts, self.positions)
        gate = dequantize_iq3_xxs(self.gate_blocks[index], *tables).flatten(-2)
        up = dequantize_iq3_xxs(self.up_blocks[index], *tables).flatten(-2)
        down = dequantize_iq3_xxs(self.down_blocks[index], *tables).flatten(-2)
        return torch.cat((gate, up), dim=-2).to(dtype), down.to(dtype)

    def forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        from fish_speech.models.text2semantic.fish_omni.moe import GATHERED_EXPERT_MAX_ROWS, MoEExperts

        if topk_ids.numel() <= GATHERED_EXPERT_MAX_ROWS:
            return MoEExperts.forward_gathered(self, x, topk_ids, topk_weights)
        return MoEExperts.forward_grouped(self, x, topk_ids, topk_weights)


# ─── Model ────────────────────────────────────────────────────────────────────


def _set_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    owner, _, leaf = name.rpartition(".")
    target = module.get_submodule(owner) if owner else module
    old = getattr(target, leaf)
    if tuple(old.shape) != tuple(tensor.shape):
        raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {tuple(old.shape)}")
    target._parameters[leaf] = nn.Parameter(tensor.to(old.dtype), requires_grad=False)


def _replace(module: nn.Module, name: str, new: nn.Module) -> None:
    owner, _, leaf = name.rpartition(".")
    setattr(module.get_submodule(owner) if owner else module, leaf, new)


def build_model(gguf: GGUF, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """(config, FishOmniModel with compact weights on device, its candidate ids)."""
    from tokenizers import Tokenizer
    from fish_speech.models.text2semantic.fish_omni.backbone import CHECKPOINT_BACKBONE_PREFIX, FishOmniBackbone
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
    from fish_speech.models.text2semantic.fish_omni.tower import TOWER_CHECKPOINT_PREFIXES, FishOmniAudioDecoder

    raw_config = json.loads(gguf.embedded_file("config.json"))
    per_layer = raw_config["text_config"]["num_experts_per_layer"]
    config = FishOmniConfig.from_dict(raw_config)
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=False, dtype=dtype)
        tower = FishOmniAudioDecoder(config, dtype=dtype)
    del backbone.lm_head  # replaced by the GGUF's 4,097-row speech head below

    def place(info: TensorInfo) -> nn.Module | torch.Tensor:
        raw = gguf.raw(info)
        if info.type == "Q8_0" and len(info.shape) == 2:
            values, scales = split_q8_0(raw, info.shape)
            return values.to(device), scales.to(device)
        return dense(gguf, info, dtype).to(device)

    loaded = set()
    for name in gguf.names("model"):
        info = gguf.info(name)
        if name.startswith(CHECKPOINT_BACKBONE_PREFIX):
            module, local = backbone, "model." + name[len(CHECKPOINT_BACKBONE_PREFIX):]
        elif name == "lm_head.weight":
            continue
        else:
            prefix = next((p for p in TOWER_CHECKPOINT_PREFIXES if name.startswith(p)), None)
            if prefix is None:
                raise ValueError(f"unknown tensor {name}")
            module, local = tower, TOWER_CHECKPOINT_PREFIXES[prefix] + name[len(prefix):]
        if ".mlp.experts." in local:
            continue  # stacked below, per layer
        if local.endswith(".mlp.gate.weight"):  # router: one row per kept expert
            layer = int(local.split(".")[2])
            _replace(module, local[:-len(".weight")], nn.Linear(config.text.hidden_size, per_layer[layer], bias=False,
                                                                  dtype=dtype, device="meta"))
        target_name = local[:-len(".weight")] if local.endswith(".weight") else None
        target = module.get_submodule(target_name) if target_name else None
        if info.type == "Q8_0" and isinstance(target, nn.Linear):
            values, scales = place(info)
            _replace(module, target_name, Q8Linear(values, scales, dtype))
        elif info.type == "Q8_0" and isinstance(target, nn.Embedding):
            values, scales = place(info)
            _replace(module, target_name, Q8Embedding(values, scales, dtype))
        else:
            _set_parameter(module, local, place(info))
        loaded.add(name)

    for index, layer in enumerate(backbone.model.layers):
        base = f"{CHECKPOINT_BACKBONE_PREFIX}layers.{index}.mlp.experts."

        def blocks(part: str) -> torch.Tensor:
            info = gguf.info(base + part + ".weight")
            *outer, cols = info.shape
            if info.type != "IQ3_XXS" or outer[0] != per_layer[index]:
                raise ValueError(f"layer {index} {part}: {info.type} {info.shape}")
            array = np.asarray(gguf.raw(info)).reshape(*outer, cols // 256, 98)
            return torch.from_numpy(array.copy()).to(device)

        layer.mlp.experts = IQ3Experts(blocks("gate_proj"), blocks("up_proj"), blocks("down_proj"),
                                       config.text.hidden_size, config.text.moe_intermediate_size)
        layer.mlp.top_k = config.text.num_experts_per_tok

    leftovers = [n for n, p in (*backbone.named_parameters(), *tower.named_parameters()) if p.is_meta]
    if leftovers:
        raise ValueError(f"not in the GGUF: {leftovers[:5]} ({len(leftovers)} tensors)")
    tokenizer = Tokenizer.from_str(gguf.embedded_file("tokenizer.json").decode("utf-8"))
    model = FishOmniModel(config, backbone, tower, tokenizer).eval().requires_grad_(False)
    band = torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1)
    candidate_ids = torch.cat((torch.tensor([config.eos_token_id]), band)).sort().values
    head_info = gguf.info("lm_head.weight")
    if head_info.shape[0] != len(candidate_ids):
        raise ValueError(f"speech head has {head_info.shape[0]} rows, expected {len(candidate_ids)}")
    backbone.register_buffer("candidate_ids", candidate_ids.to(device), persistent=False)
    backbone.register_buffer("candidate_head", dense(gguf, head_info, dtype).to(device), persistent=False)
    return config, model


# ─── Codec and engine hook ────────────────────────────────────────────────────


def load_codec(gguf: GGUF, dtype: torch.dtype = torch.float32):
    """The Fish DAC codec from the GGUF's codec_weights (weight norm already folded)."""
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
    from fish_speech.codec import build_codec

    codec = build_codec()
    for module in list(codec.modules()):  # fold first: the GGUF stores the folded weights
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    expected = codec.state_dict()
    state = {name: dense(gguf, gguf.info(name, "codec"), torch.float32) for name in expected}
    codec.load_state_dict(state, strict=True)
    return codec.eval().to(dtype=dtype)


def install(path: str) -> None:
    """Make the Space's engine load Drama 3 from the GGUF at `path`."""
    import time
    import drama3_engine
    import drama3_fast

    def load_weights(checkpoint_dir=None, device: str = "cuda"):
        started = time.perf_counter()
        target = torch.device(device if device != "cuda" else f"cuda:{torch.cuda.current_device()}")
        gguf = GGUF(path)
        config, model = build_model(gguf, target)
        cpu_codec = load_codec(gguf).requires_grad_(False)  # encodes reference clips (fp32, cached per clip)
        codec = copy.deepcopy(cpu_codec)
        for part in (codec, codec.quantizer):  # the GPU copy only vocodes: drop the encode-only parts (~0.85 GiB)
            for name in ("encoder", "downsample", "pre_module"):
                if hasattr(part, name):
                    setattr(part, name, None)
        codec.decoder.half()  # as the Space: fp16 decoder
        codec.to(target)
        CPU_CODEC[0] = cpu_codec
        drama3_fast.install_decode_attention()
        print(f"[small] drama3-small loaded from GGUF in {time.perf_counter() - started:.0f}s: "
              f"{torch.cuda.memory_allocated(target) / 2**30:.2f} GiB on {target}", flush=True)
        return drama3_engine.Weights(config, model, codec)

    drama3_engine.load_weights = load_weights
    base = drama3_engine.Engine

    class CpuEncodeEngine(base):
        @torch.inference_mode()
        def encode(self, reference):
            """Reference codes from the CPU codec (the GPU copy cannot encode)."""
            import hashlib
            key = hashlib.sha1(str(reference.get("data") or reference.get("base64")).encode()).hexdigest()
            if (codes := self.reference_codes.get(key)) is not None:
                self.reference_codes.move_to_end(key)
                return codes
            samples = torch.from_numpy(drama3_engine.reference_audio(reference))
            audio = samples.clamp(-1.0, 1.0)[None, None]
            indices, _ = CPU_CODEC[0].encode(audio, torch.tensor([audio.shape[-1]]))
            codes = indices[0].long().cpu()
            self.reference_codes[key] = codes
            while len(self.reference_codes) > drama3_engine.REFERENCE_CACHE_SIZE:
                self.reference_codes.popitem(last=False)
            return codes

    CpuEncodeEngine.__name__ = base.__name__
    drama3_engine.Engine = CpuEncodeEngine


CPU_CODEC: list = [None]
