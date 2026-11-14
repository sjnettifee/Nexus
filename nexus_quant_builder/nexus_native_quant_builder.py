#!/usr/bin/env python3
"""
Nexus Native Quant Builder

All-in-one GUI + builder for creating a native Nexus model package directly
from HuggingFace safetensors.

This does NOT require GGUF.
This does NOT require a donor model.
This does NOT rename/fuse/flatten tensors.
It preserves original HF tensor names and shapes, keeps the full multimodal
feature set by default, and only changes storage for the tensors you choose
to quantize.

Native package layout:

  MyModel.nxmodel/
    nexus_model.json          # manifest with config, tensor table, offsets
    metadata/                 # copied HF config/tokenizer/chat files
      config.json
      tokenizer.json
      tokenizer_config.json
      ...
    <modelname>-000.nxw       # raw tensor payloads (named after the model)
    <modelname>-001.nxw       # optional multipart

Remote split mode creates:

  MyModel.local.nxmodel/
  MyModel.remote.nxmodel/

The engine-side update should add a Nexus-native parser that reads
nexus_model.json and the .nxw files instead of GGUF.

Requires:
  python3
  tkinter
  numpy

Run:
  python3 nexus_quant_policy_app.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import shutil
import struct
import threading
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# ----------------------------
# Constants
# ----------------------------

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "U64": 8,
    "I32": 4,
    "U32": 4,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

DECISIONS = [
    "AUTO",
    "KEEP_SOURCE",
    "KEEP_F32",
    "Q4G64",
    "Q5G64",
    "Q6G64",
    "Q8G64",
    "SKIP",
]

QUANT_BITS = {
    "Q4G64": 4,
    "Q5G64": 5,
    "Q6G64": 6,
    "Q8G64": 8,
}

FULL_DECISIONS = {"KEEP_SOURCE", "KEEP_F32", "SKIP"}

COMPONENT_LANGUAGE = "language"
COMPONENT_VISUAL = "visual"
COMPONENT_AUDIO = "audio"
COMPONENT_MTP = "mtp"
COMPONENT_OTHER = "other"

NXM_MAGIC = b"NEXUS_NXM_V1\0\0\0\0"  # 16 bytes
NXM_HEADER_STRUCT = "<16sQQQ"  # magic, manifest_json_len, file_table_json_len, reserved

# FAT32's real limit is 2^32 - 1 bytes (4 GiB - 1 byte); warn a bit under that so the
# estimate (which is approximate pre-build) doesn't miss a file that lands right on the line.
FAT32_FILE_SIZE_WARN_BYTES = 3_800_000_000


# ----------------------------
# Data model
# ----------------------------

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: str
    source: str
    shard: str = ""
    nbytes: int = 0
    data_start: int = 0
    offset_start: int = 0
    offset_end: int = 0
    layer: Optional[int] = None
    component: str = COMPONENT_OTHER
    tensor_class: str = "unknown"
    auto_decision: str = "KEEP_SOURCE"
    decision: str = "KEEP_SOURCE"
    reason: str = ""
    manual: bool = False
    notes: str = ""

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def elements(self) -> int:
        total = 1
        for d in self.shape:
            total *= int(d)
        return total

    @property
    def mib(self) -> float:
        return self.nbytes / (1024 * 1024) if self.nbytes else 0.0


@dataclass
class QuantSettings:
    default_quant: str = "Q4G64"
    embedding_quant: str = "Q5G64"
    output_quant: str = "Q5G64"
    ffn_quant: str = "Q4G64"
    attention_quant: str = "Q4G64"
    ssm_matrix_quant: str = "Q4G64"
    min_matrix_mib: float = 1.0
    keep_1d: bool = True
    keep_norms: bool = True
    keep_biases: bool = True
    keep_ssm_control: bool = True
    keep_conv1d: bool = True
    keep_small_tensors: bool = True
    full_precision_decision: str = "KEEP_SOURCE"

    include_language: bool = True
    include_visual: bool = True
    include_audio: bool = True
    include_mtp: bool = True
    include_other: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputSettings:
    mode: str = "Single native file"
    output_basename: str = "model-nexus-native"
    max_part_gib: float = 8.0
    remote_tail_layers: int = 0
    duplicate_embeddings_to_remote: bool = False
    split_count: int = 2                  # "Split across several computers": how many pieces
    split_layers: str = ""                # layers per piece, e.g. "20,22,22"; blank = as even as possible
    copy_metadata_files: bool = True
    chunk_mib: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScanResult:
    path: str
    source_type: str
    tensors: List[TensorInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorAssignment:
    package_role: str
    tensor: TensorInfo


@dataclass
class PackagePlan:
    package_dir: str
    role: str
    layer_start: Optional[int]
    layer_end: Optional[int]
    tensors: List[TensorInfo] = field(default_factory=list)
    estimated_tensor_bytes: int = 0
    notes: List[str] = field(default_factory=list)


# ----------------------------
# Helpers
# ----------------------------

def product(vals: Iterable[int]) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def fmt_bytes(n: int) -> str:
    n = int(n or 0)
    if n >= 1024**4:
        return f"{n / 1024**4:.2f} TiB"
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def format_shape(shape: List[int]) -> str:
    return "[" + ", ".join(str(x) for x in shape) + "]"


def estimate_nbytes(dtype: str, shape: List[int]) -> int:
    return int(product(shape) * DTYPE_BYTES.get(dtype, 0))


def infer_component(name: str) -> str:
    # Classify removable feature modules first. Some model wrappers put
    # vision/MTP under a generic model.* namespace, so these checks must run
    # before the broader language-model checks.
    lo = name.lower()

    # Multi-token prediction / speculative decoding modules. Qwen3.6 uses
    # names like mtp.fc.weight and mtp.layers.0.*; other converters sometimes
    # wrap this as model.mtp.* or model.language_model.mtp.*.
    if (
        lo.startswith("mtp.")
        or lo.startswith("model.mtp.")
        or lo.startswith("model.language_model.mtp.")
        or ".mtp." in lo
        or "multi_token" in lo
        or "speculative" in lo
        or "draft_model" in lo
    ):
        return COMPONENT_MTP

    # Common visual/VL module names, including Qwen3.6's model.visual.*
    # blocks/merger/patch_embed tensors and common projector aliases.
    if (
        lo.startswith("model.visual.")
        or lo.startswith("visual.")
        or lo.startswith("vision_model.")
        or lo.startswith("model.vision_model.")
        or lo.startswith("model.vision_tower.")
        or lo.startswith("vision_tower.")
        or lo.startswith("image_encoder.")
        or lo.startswith("model.image_encoder.")
        or lo.startswith("multi_modal_projector.")
        or lo.startswith("model.multi_modal_projector.")
        or lo.startswith("mm_projector.")
        or lo.startswith("model.mm_projector.")
        or lo.startswith("vision_proj.")
        or lo.startswith("model.vision_proj.")
    ):
        return COMPONENT_VISUAL

    # Common audio/speech module names. This is intentionally broad so future
    # models keep audio tensors instead of falling into "other" by accident.
    if (
        lo.startswith("model.audio.")
        or lo.startswith("audio.")
        or lo.startswith("audio_model.")
        or lo.startswith("model.audio_model.")
        or lo.startswith("model.audio_tower.")
        or lo.startswith("audio_tower.")
        or lo.startswith("model.speech.")
        or lo.startswith("speech.")
        or lo.startswith("model.speech_model.")
        or lo.startswith("speech_model.")
        or lo.startswith("model.whisper.")
        or lo.startswith("whisper.")
        or (lo.startswith("model.encoder.") and ("audio" in lo or "speech" in lo))
    ):
        return COMPONENT_AUDIO

    # Language core. Covers Qwen3.6 wrapper names and common text-only HF names.
    if (
        lo.startswith("model.language_model.")
        or lo.startswith("language_model.")
        or lo.startswith("model.layers.")
        or lo.startswith("model.embed_tokens.")
        or lo.startswith("model.norm.")
        or lo.startswith("decoder.layers.")
        or lo.startswith("model.decoder.layers.")
        or lo.startswith("model.decoder.embed_tokens.")
        or lo.startswith("transformer.")
        or lo == "lm_head.weight"
        or lo.endswith(".lm_head.weight")
    ):
        return COMPONENT_LANGUAGE

    return COMPONENT_OTHER


def parse_language_layer_index(name: str) -> Optional[int]:
    # Strict: visual.blocks.N must not count as decode layer N.
    for pat in (
        r"^model\.language_model\.layers\.(\d+)\.",
        r"^language_model\.layers\.(\d+)\.",
        r"^model\.layers\.(\d+)\.",
        r"^decoder\.layers\.(\d+)\.",
        r"^model\.decoder\.layers\.(\d+)\.",
        r"^blk\.(\d+)\.",
    ):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def classify_tensor(name: str, shape: List[int]) -> str:
    lo = name.lower()

    if any(x in lo for x in ["embed_tokens.weight", "token_embd.weight", "embed"]):
        if lo.endswith(".weight"):
            return "embedding"

    if "norm" in lo or "layernorm" in lo or "ln_" in lo:
        return "norm"

    if lo.endswith(".bias") or ".bias" in lo:
        return "bias"

    if lo == "lm_head.weight" or lo.endswith(".lm_head.weight") or lo.endswith("output.weight"):
        return "output"

    if any(x in lo for x in ["a_log", "ssm_a", "dt_bias", "ssm_dt"]):
        return "ssm_control"

    if "conv1d" in lo or "conv_1d" in lo or "ssm_conv" in lo:
        return "conv1d"

    if "rope" in lo or "rotary" in lo or "freq" in lo:
        return "rope_freq"

    if any(x in lo for x in ["linear_attn.out_proj", "linear_attn.in_proj_z", "ssm_out", "ssm_beta", "ssm_alpha"]):
        return "ssm_matrix"

    if any(x in lo for x in [
        "linear_attn.in_proj_qkv",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "attn.qkv",
        "attn.proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]):
        return "attention"

    if any(x in lo for x in [
        ".mlp.",
        "gate_proj",
        "up_proj",
        "down_proj",
        "linear_fc1",
        "linear_fc2",
        "ffn",
        "feed_forward",
    ]):
        return "ffn"

    if len(shape) >= 2:
        return "matrix"

    return "other"


def is_embedding(t: TensorInfo) -> bool:
    return t.tensor_class == "embedding"


def is_final_output(t: TensorInfo) -> bool:
    lo = t.name.lower()
    return (
        t.name == "lm_head.weight"
        or lo.endswith("lm_head.weight")
        or lo == "model.language_model.norm.weight"
        or lo.endswith(".language_model.norm.weight")
    )


def decision_to_storage_type(t: TensorInfo) -> str:
    if t.decision == "KEEP_SOURCE":
        return t.dtype
    if t.decision == "KEEP_F32":
        return "F32"
    if t.decision in QUANT_BITS:
        return f"NQ{QUANT_BITS[t.decision]}_G64"
    if t.decision == "SKIP":
        return "SKIP"
    return t.dtype


def estimated_output_nbytes(t: TensorInfo) -> int:
    if t.decision == "SKIP":
        return 0
    if t.decision == "KEEP_SOURCE":
        return t.nbytes
    if t.decision == "KEEP_F32":
        return t.elements * 4
    if t.decision in QUANT_BITS:
        bits = QUANT_BITS[t.decision]
        group = 64
        cols = int(t.shape[-1]) if t.shape else t.elements
        rows = max(1, t.elements // max(1, cols))
        groups_per_row = (cols + group - 1) // group
        packed_bytes = (group * bits + 7) // 8
        return rows * groups_per_row * (2 + packed_bytes)  # f16 scale + packed q
    return t.nbytes


def tensor_included_by_component(t: TensorInfo, settings: QuantSettings) -> bool:
    if t.component == COMPONENT_LANGUAGE:
        return settings.include_language
    if t.component == COMPONENT_VISUAL:
        return settings.include_visual
    if t.component == COMPONENT_AUDIO:
        return settings.include_audio
    if t.component == COMPONENT_MTP:
        return settings.include_mtp
    return settings.include_other


def auto_decide(t: TensorInfo, settings: QuantSettings) -> Tuple[str, str]:
    if not tensor_included_by_component(t, settings):
        return "SKIP", f"component excluded: {t.component}"

    cls = t.tensor_class
    matrix_like = t.ndim >= 2
    size_ok = t.mib >= settings.min_matrix_mib

    if settings.keep_1d and t.ndim <= 1:
        return settings.full_precision_decision, "1D/control vector kept source precision"

    if settings.keep_small_tensors and not size_ok:
        return settings.full_precision_decision, f"below threshold {settings.min_matrix_mib:.2f} MiB"

    if settings.keep_norms and cls == "norm":
        return settings.full_precision_decision, "norm kept source precision"

    if settings.keep_biases and cls == "bias":
        return settings.full_precision_decision, "bias kept source precision"

    if settings.keep_ssm_control and cls == "ssm_control":
        return settings.full_precision_decision, "SSM/DeltaNet control kept source precision"

    if settings.keep_conv1d and cls == "conv1d":
        return settings.full_precision_decision, "conv1d kernel kept source precision"

    if cls in {"rope_freq", "other"} and not matrix_like:
        return settings.full_precision_decision, "special non-matrix kept source precision"

    if cls == "embedding":
        return settings.embedding_quant, "embedding matrix quant policy"

    if cls == "output":
        return settings.output_quant, "output/lm_head quant policy"

    if cls == "ffn":
        return settings.ffn_quant, "FFN matrix quant policy"

    if cls == "attention":
        return settings.attention_quant, "attention matrix quant policy"

    if cls == "ssm_matrix":
        return settings.ssm_matrix_quant, "SSM big matrix quant policy"

    if matrix_like and size_ok:
        return settings.default_quant, "generic big matrix quant policy"

    return settings.full_precision_decision, "default keep source precision"


# ----------------------------
# Safetensors scanning
# ----------------------------

@dataclass
class SafeTensorHeader:
    path: Path
    header: Dict[str, Any]
    header_len: int
    data_start: int


def read_safetensors_header(path: Path) -> SafeTensorHeader:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too small")
        header_len = struct.unpack("<Q", raw)[0]
        hb = f.read(header_len)
        if len(hb) != header_len:
            raise ValueError(f"{path} has truncated header")
        header = json.loads(hb.decode("utf-8"))
        return SafeTensorHeader(path=path, header=header, header_len=header_len, data_start=8 + header_len)


def scan_safetensors_file(path: Path) -> List[TensorInfo]:
    st = read_safetensors_header(path)
    tensors: List[TensorInfo] = []
    for name, meta in st.header.items():
        if name == "__metadata__":
            continue
        dtype = str(meta.get("dtype", "UNKNOWN"))
        shape = [int(x) for x in meta.get("shape", [])]
        offsets = meta.get("data_offsets", [0, estimate_nbytes(dtype, shape)])
        start = int(offsets[0])
        end = int(offsets[1])
        component = infer_component(name)
        tensors.append(TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            source="safetensors",
            shard=str(path),
            nbytes=end - start,
            data_start=st.data_start,
            offset_start=start,
            offset_end=end,
            layer=parse_language_layer_index(name),
            component=component,
            tensor_class=classify_tensor(name, shape),
        ))
    return tensors


def scan_hf_folder(path: Path) -> ScanResult:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No .safetensors files found in {path}")

    tensors: List[TensorInfo] = []
    total_file_bytes = 0
    for f in files:
        total_file_bytes += f.stat().st_size
        tensors.extend(scan_safetensors_file(f))

    meta: Dict[str, Any] = {
        "source_shard_count": len(files),
        "source_file_bytes": total_file_bytes,
        "source_tensor_bytes": sum(t.nbytes for t in tensors),
        "source_files": [str(f) for f in files],
    }

    for name in ["config.json", "generation_config.json", "tokenizer_config.json"]:
        p = path / name
        if p.exists():
            try:
                meta[name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                meta[f"{name}_parse_error"] = True

    return ScanResult(str(path), "hf_safetensors_folder", tensors, meta)


def scan_path(path: Path) -> ScanResult:
    if path.is_dir():
        return scan_hf_folder(path)
    if path.suffix.lower() == ".safetensors":
        tensors = scan_safetensors_file(path)
        return ScanResult(str(path), "safetensors_file", tensors, {
            "source_shard_count": 1,
            "source_file_bytes": path.stat().st_size,
            "source_tensor_bytes": sum(t.nbytes for t in tensors),
            "source_files": [str(path)],
        })
    raise ValueError("Choose a HuggingFace folder or a .safetensors file")


def infer_n_layers(scan: ScanResult) -> int:
    cfg = scan.metadata.get("config.json")
    if isinstance(cfg, dict):
        for key in ["num_hidden_layers", "n_layer", "num_layers", "block_count"]:
            if key in cfg:
                try:
                    return int(cfg[key])
                except Exception:
                    pass
        # Some multimodal configs nest the language config.
        for nested in ["text_config", "language_config", "llm_config"]:
            if isinstance(cfg.get(nested), dict):
                for key in ["num_hidden_layers", "n_layer", "num_layers"]:
                    if key in cfg[nested]:
                        try:
                            return int(cfg[nested][key])
                        except Exception:
                            pass

    return max((t.layer for t in scan.tensors if t.layer is not None), default=-1) + 1


def summarize(scan: Optional[ScanResult]) -> Dict[str, Any]:
    if not scan:
        return {}
    tensors = scan.tensors
    return {
        "tensor_count": len(tensors),
        "source_file_bytes": int(scan.metadata.get("source_file_bytes", sum(t.nbytes for t in tensors))),
        "source_tensor_bytes": sum(t.nbytes for t in tensors),
        "estimated_output_bytes": sum(estimated_output_nbytes(t) for t in tensors),
        "by_decision": count_by(tensors, lambda t: t.decision),
        "by_component": count_by(tensors, lambda t: t.component),
        "by_class": count_by(tensors, lambda t: t.tensor_class),
        "n_layers": infer_n_layers(scan),
    }


def find_nested_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config dict that most likely describes the language core."""
    if not isinstance(config, dict):
        return {}
    for key in ("text_config", "language_config", "llm_config", "language_model_config"):
        if isinstance(config.get(key), dict):
            return config[key]
    return config


def infer_layer_kinds(tensors: List[TensorInfo]) -> Dict[str, str]:
    by_layer: Dict[int, set] = {}
    for t in tensors:
        if t.component != COMPONENT_LANGUAGE or t.layer is None:
            continue
        s = by_layer.setdefault(int(t.layer), set())
        n = t.name
        if ".linear_attn." in n:
            s.add("linear_attn")
        if ".self_attn." in n:
            s.add("self_attn")
        if ".mlp." in n:
            s.add("mlp")
    out: Dict[str, str] = {}
    for layer, kinds in sorted(by_layer.items()):
        if "linear_attn" in kinds and "self_attn" in kinds:
            out[str(layer)] = "hybrid"
        elif "linear_attn" in kinds:
            out[str(layer)] = "linear_attn"
        elif "self_attn" in kinds:
            out[str(layer)] = "self_attn"
        else:
            out[str(layer)] = "+".join(sorted(kinds)) or "unknown"
    return out



# ---------------------------------------------------------------------------
# Conventions the runtime needs but cannot read from the tensors.  The .nxm keeps the ORIGINAL
# HuggingFace weights (never permuted, never norm-baked), so both answers follow from the HF
# model_type.  They are written into the manifest so the engine reads them instead of guessing.
#   rope_layout : "half_split" = HF rotate_half (NeoX).  "interleaved" = adjacent pairs (GPT-J).
#   norm        : "direct"  -> y = rms(x) * w          (Llama, Mistral, Qwen2, Qwen3, ...)
#                 "zero_centered" -> y = rms(x)*(1+w)  (Gemma family, Qwen3-Next / Qwen3.5+)
# Anything not listed gets "unknown" so the engine falls back to inference AND logs it.
# ---------------------------------------------------------------------------
_ROPE_HALF_SPLIT = {
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "qwen3_next",
    "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text", "gemma", "gemma2", "gemma3",
    "gemma3_text", "phi", "phi3", "phimoe", "olmo", "olmo2", "olmoe", "gpt_neox", "internlm2",
    "starcoder2", "stablelm", "granite", "falcon",
}
_ROPE_INTERLEAVED = {"gptj", "cohere", "cohere2", "glm", "glm4"}
_NORM_ZERO_CENTERED = {
    "gemma", "gemma2", "gemma3", "gemma3_text", "qwen3_next", "qwen3_5", "qwen3_5_moe",
    "qwen3_5_text", "qwen3_5_moe_text",
}
_NORM_DIRECT = {
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "phi3", "phimoe",
    "olmo2", "olmoe", "internlm2", "granite", "starcoder2",
}


def infer_conventions(model_type, language_model_type=None) -> Dict[str, Any]:
    names = [str(t).lower() for t in (language_model_type, model_type) if t]
    def pick(*tables):
        for n in names:
            for label, tab in tables:
                if n in tab:
                    return label, n
        return "unknown", (names[0] if names else None)
    rope, rope_src = pick(("half_split", _ROPE_HALF_SPLIT), ("interleaved", _ROPE_INTERLEAVED))
    norm, norm_src = pick(("zero_centered", _NORM_ZERO_CENTERED), ("direct", _NORM_DIRECT))
    return {"rope_layout": rope, "norm": norm,
            "derived_from_model_type": rope_src or norm_src, "labeled_by": "nexus_quant_builder"}


def build_model_info(scan: ScanResult, plan: Optional[PackagePlan] = None) -> Dict[str, Any]:
    """Compact metadata Nexus can use before touching weight payloads."""
    config = scan.metadata.get("config.json", {})
    language_config = find_nested_config(config if isinstance(config, dict) else {})
    tensors = plan.tensors if plan is not None else scan.tensors

    def cfg_get(*keys, default=None):
        for key in keys:
            if isinstance(language_config, dict) and key in language_config:
                return language_config.get(key)
            if isinstance(config, dict) and key in config:
                return config.get(key)
        return default

    arch = cfg_get("architectures", default=[])
    if isinstance(arch, list):
        arch_list = arch
    elif arch:
        arch_list = [arch]
    else:
        arch_list = []

    source_path = Path(scan.path)
    component_counts = count_by(tensors, lambda t: t.component)
    class_counts = count_by(tensors, lambda t: t.tensor_class)
    decision_counts = count_by(tensors, lambda t: t.decision)
    storage_counts: Dict[str, int] = {}
    storage_bytes: Dict[str, int] = {}
    for t in tensors:
        st = decision_to_storage_type(t)
        storage_counts[st] = storage_counts.get(st, 0) + 1
        storage_bytes[st] = storage_bytes.get(st, 0) + estimated_output_nbytes(t)

    return {
        "name": source_path.name if source_path.name else "unknown",
        "model_type": cfg_get("model_type", default=None),
        "architectures": arch_list,
        "torch_dtype": cfg_get("torch_dtype", default=None),
        "vocab_size": cfg_get("vocab_size", default=None),
        "hidden_size": cfg_get("hidden_size", "n_embd", default=None),
        "intermediate_size": cfg_get("intermediate_size", default=None),
        "num_hidden_layers": infer_n_layers(scan),
        "num_attention_heads": cfg_get("num_attention_heads", "n_head", default=None),
        "num_key_value_heads": cfg_get("num_key_value_heads", "n_head_kv", default=None),
        "max_position_embeddings": cfg_get("max_position_embeddings", "n_positions", default=None),
        "rope_theta": cfg_get("rope_theta", default=None),
        "conventions": infer_conventions(
            config.get("model_type") if isinstance(config, dict) else None,
            language_config.get("model_type") if isinstance(language_config, dict) else None),
        "bos_token_id": cfg_get("bos_token_id", default=None),
        "eos_token_id": cfg_get("eos_token_id", default=None),
        "source": {
            "path": scan.path,
            "type": scan.source_type,
            "shard_count": scan.metadata.get("source_shard_count"),
            "source_file_bytes": scan.metadata.get("source_file_bytes"),
            "source_tensor_bytes": scan.metadata.get("source_tensor_bytes"),
            "source_files": scan.metadata.get("source_files", []),
        },
        "counts": {
            "tensor_count": len(tensors),
            "parameter_count": int(sum(t.elements for t in tensors if t.decision != "SKIP")),
            "component_counts": component_counts,
            "class_counts": class_counts,
            "decision_counts": decision_counts,
            "storage_counts": storage_counts,
            "storage_bytes": storage_bytes,
        },
        "language_layer_kinds": infer_layer_kinds(scan.tensors),
    }


_SHARD_NAME_RE = re.compile(r"^model-\d+-of-\d+$", re.IGNORECASE)


def derive_basename(path: Path) -> str:
    """Default output name from what was actually opened, instead of the
    generic 'model-nexus-native' every build used to start with.
    Folder  -> the folder's own name (that's normally the model name:
               'Qwen3.5-7B-Instruct', etc).
    A single .safetensors file -> its filename without the extension, unless
               it's an anonymous shard name like 'model-00001-of-00002', in
               which case the containing folder's name is used instead."""
    try:
        if path.is_dir():
            name = path.name
        else:
            stem = path.stem
            name = path.parent.name if _SHARD_NAME_RE.match(stem) else stem
    except Exception:
        name = ""
    name = (name or "").strip()
    # filesystem-safe: no path separators or characters Windows/macOS/Linux all reject
    name = re.sub(r'[\\/:*?"<>|]+', "-", name)
    name = re.sub(r"\s+", "-", name).strip("-._")
    name = name[:80]
    return name or "model-nexus-native"


def count_by(items: List[TensorInfo], fn: Callable[[TensorInfo], str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        key = fn(it)
        out[key] = out.get(key, 0) + 1
    return out


# ----------------------------
# Output planning
# ----------------------------

SPLIT_MODE = "Split across several computers"


def split_layer_sizes(n_layers: int, count: int, text: str, layer_bytes: Optional[List[int]] = None, global_bytes: int = 0) -> List[int]:
    """Layers per piece.
    'text' = comma list (e.g. 20,22,22): the list decides the number of pieces; if it adds up to fewer layers than the model has the last piece
    takes the rest, if more it is trimmed.
    Blank = 'count' pieces BALANCED BY SIZE: piece 1 also carries the global tensors (embeddings, output layer, vision...), so it gets fewer layers."""
    n_layers = max(1, int(n_layers))
    count = max(1, min(int(count or 1), n_layers))
    given = []
    for part in re.split(r"[,\s;]+", (text or "").strip()):
        if part.isdigit() and int(part) > 0:
            given.append(int(part))
    if given:
        out: List[int] = []
        left = n_layers
        for g in given:
            if left <= 0:
                break
            out.append(min(g, left)); left -= out[-1]
        if left > 0:
            out[-1] += left
        return out
    if layer_bytes and len(layer_bytes) == n_layers and sum(layer_bytes) > 0 and count > 1:
        total = float(sum(layer_bytes) + global_bytes)
        cum = [float(global_bytes)]                       # cum[i] = bytes in pieces up to and including layer i-1 (globals count for piece 1)
        for x in layer_bytes:
            cum.append(cum[-1] + x)
        cuts = [0]                                         # piece k ends at the layer boundary whose running total is closest to (k+1)/count of everything
        for k in range(count - 1):
            want = total * (k + 1) / count
            lo = cuts[-1] + 1                              # every piece gets at least one layer
            hi = n_layers - (count - 1 - k)                # and so does every piece still to come
            best = min(range(lo, hi + 1), key=lambda i: abs(cum[i] - want))
            cuts.append(best)
        cuts.append(n_layers)
        return [cuts[i + 1] - cuts[i] for i in range(count)]
    base, extra = divmod(n_layers, count)
    return [base + (1 if i < extra else 0) for i in range(count)]


def build_package_plans(scan: ScanResult, settings: QuantSettings, output: OutputSettings, output_root: Path) -> List[PackagePlan]:
    included = [t for t in scan.tensors if t.decision != "SKIP"]

    if output.mode == "Remote two-computer split":
        n_layers = infer_n_layers(scan)
        tail = int(output.remote_tail_layers or 0)
        if tail <= 0:
            tail = max(1, n_layers // 2)
        tail = max(1, min(tail, max(1, n_layers - 1)))
        split = n_layers - tail

        local = PackagePlan(
            package_dir=str(output_root / f"{output.output_basename}.local.nxmodel"),
            role="local_primary",
            layer_start=0,
            layer_end=split,
            notes=[f"language layers 0-{split-1}; remote starts at {split}"],
        )
        remote = PackagePlan(
            package_dir=str(output_root / f"{output.output_basename}.remote.nxmodel"),
            role="remote_worker",
            layer_start=split,
            layer_end=n_layers,
            notes=[f"language layers {split}-{n_layers-1}; copy this package to worker"],
        )

        for t in included:
            if t.component == COMPONENT_LANGUAGE and t.layer is not None:
                (local if t.layer < split else remote).tensors.append(t)
            elif t.component == COMPONENT_LANGUAGE and is_embedding(t):
                local.tensors.append(t)
                if output.duplicate_embeddings_to_remote:
                    remote.tensors.append(t)
                    remote.notes.append("embedding duplicated by option")
            elif t.component == COMPONENT_LANGUAGE and is_final_output(t):
                remote.tensors.append(t)
            elif t.component == COMPONENT_LANGUAGE:
                # Unknown/global language tensor. Duplicate safely.
                local.tensors.append(t)
                remote.tensors.append(t)
            else:
                # Full-model feature modules stay on the primary side. The worker
                # shard only needs the tail language layers it actually executes.
                local.tensors.append(t)
                if t.component in (COMPONENT_VISUAL, COMPONENT_AUDIO, COMPONENT_MTP, COMPONENT_OTHER):
                    if f"{t.component} feature tensors kept in local primary package" not in local.notes:
                        local.notes.append(f"{t.component} feature tensors kept in local primary package")

        for p in [local, remote]:
            p.estimated_tensor_bytes = sum(estimated_output_nbytes(t) for t in p.tensors)
        return [local, remote]

    if output.mode == SPLIT_MODE:
        n_layers = infer_n_layers(scan)
        # Everything that is not one numbered layer ("globals": embeddings, final norm, output layer, vision/audio/MTP, other) goes to piece 1.
        # Layers go to whichever piece their number falls in.  Nothing is duplicated, nothing is guessed: a tensor is in exactly one piece.
        layer_bytes = [0] * max(1, n_layers)
        global_bytes = 0
        for t in included:
            nb = estimated_output_nbytes(t)
            if t.component == COMPONENT_LANGUAGE and t.layer is not None and 0 <= int(t.layer) < len(layer_bytes):
                layer_bytes[int(t.layer)] += nb
            else:
                global_bytes += nb
        sizes = split_layer_sizes(n_layers, int(output.split_count or 2), output.split_layers, layer_bytes, global_bytes)
        count = len(sizes)
        plans: List[PackagePlan] = []
        a = 0
        for k, sz in enumerate(sizes):
            plans.append(PackagePlan(
                package_dir=str(output_root / f"{output.output_basename}.part{k + 1}of{count}.nxmodel"),
                role="primary" if k == 0 else f"worker_{k}",
                layer_start=a, layer_end=a + sz,
                notes=[f"piece {k + 1} of {count}: language layers {a}-{a + sz - 1}",
                       "self-contained: model description, tokenizer and chat template are inside this piece"]))
            a += sz
        for t in included:
            if t.component == COMPONENT_LANGUAGE and t.layer is not None:
                for pl in plans:
                    if pl.layer_start <= int(t.layer) < pl.layer_end:
                        pl.tensors.append(t); break
                else:
                    plans[-1].tensors.append(t)
            else:
                plans[0].tensors.append(t)
        plans[0].notes.append("also holds the embeddings, final norm, output layer and any vision/audio tensors (each stored once, in this piece only)")
        for pl in plans:
            pl.estimated_tensor_bytes = sum(estimated_output_nbytes(t) for t in pl.tensors)
        return plans

    p = PackagePlan(
        package_dir=str(output_root / f"{output.output_basename}.nxmodel"),
        role="single",
        layer_start=0,
        layer_end=infer_n_layers(scan),
        tensors=included,
        notes=["single native package"],
    )
    p.estimated_tensor_bytes = sum(estimated_output_nbytes(t) for t in p.tensors)
    return [p]


# ----------------------------
# Native writer
# ----------------------------

class BuildCancelled(Exception):
    pass


class NativePackageBuilder:
    def __init__(
        self,
        scan: ScanResult,
        settings: QuantSettings,
        output_settings: OutputSettings,
        output_root: Path,
        log: Callable[[str], None],
        progress: Callable[[str, float], None],
        cancel_event: threading.Event,
    ):
        self.scan = scan
        self.settings = settings
        self.output_settings = output_settings
        self.output_root = output_root
        self.log = log
        self.progress = progress
        self.cancel_event = cancel_event

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise BuildCancelled("Build cancelled")

    def build(self) -> List[str]:
        try:
            import numpy as np  # noqa
        except Exception as e:
            raise RuntimeError("numpy is required for quantized build: python3 -m pip install numpy") from e

        self.output_root.mkdir(parents=True, exist_ok=True)
        plans = build_package_plans(self.scan, self.settings, self.output_settings, self.output_root)
        built = []

        single_file_mode = self.output_settings.mode in ("Single native file", SPLIT_MODE)

        for i, plan in enumerate(plans):
            self.check_cancel()
            self.log(f"Building {plan.role}: {plan.package_dir}")
            self._build_one_package(plan, package_index=i, package_count=len(plans))

            if single_file_mode:
                package_dir = Path(plan.package_dir)
                nxm_path = (self.output_root / f"{self.output_settings.output_basename}.nxm") if plan.role == "single" else \
                           (self.output_root / (package_dir.name[:-len(".nxmodel")] + ".nxm"))
                self.log("Tensor payload stage complete.")
                self.log(f"Packaging temporary .nxmodel folder into single native file: {nxm_path}")
                self.progress("Packaging single .nxm file", (i + 0.97) / max(1, len(plans)))
                self._pack_package_dir_to_nxm(package_dir, nxm_path, package_index=i, package_count=len(plans))
                self.log(f"Removing temporary build folder: {package_dir}")
                shutil.rmtree(package_dir)
                built.append(str(nxm_path))
            else:
                built.append(plan.package_dir)

        return built

    def _pack_package_dir_to_nxm(self, package_dir: Path, nxm_path: Path, package_index: int = 0, package_count: int = 1):
        """
        Pack a native .nxmodel directory into one .nxm file.

        File layout:
          uint8  magic[16] = NXM_MAGIC
          uint64 manifest_json_len
          uint64 file_table_json_len
          uint64 reserved = 0
          bytes  manifest_json          # nexus_model.json, immediately readable
          bytes  file_table_json        # paths + payload offsets/sizes
          bytes  payload files          # metadata + weight bytes

        Offsets in file_table_json are relative to payload_start.
        """
        self.check_cancel()

        if len(NXM_MAGIC) != 16:
            raise RuntimeError(f"Internal error: NXM_MAGIC must be 16 bytes, got {len(NXM_MAGIC)}")

        manifest_path = package_dir / "nexus_model.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["container"] = {
            "format": "nexus_native_single_file",
            "version": 1,
            "magic": "NEXUS_NXM_V1",
            "file": nxm_path.name,
            "manifest_embedded_after_header": True,
            "file_table_embedded_after_manifest": True,
            "payload_offsets_relative_to": "payload_start",
        }

        # Write updated manifest back so the embedded file copy matches the
        # immediate manifest section exactly.
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)

        files: List[Path] = []
        for p in sorted(package_dir.rglob("*")):
            if p.is_file():
                files.append(p)

        entries = []
        offset = 0
        for p in files:
            rel = str(p.relative_to(package_dir)).replace(os.sep, "/")
            size = p.stat().st_size
            if rel == "nexus_model.json":
                kind = "manifest"
            elif rel.startswith("metadata/"):
                kind = "metadata"
            elif rel.endswith(".nxw"):
                kind = "weights"
            else:
                kind = "file"

            entries.append({
                "path": rel,
                "kind": kind,
                "offset": offset,
                "nbytes": size,
            })
            offset += size

        file_table = {
            "format": "nexus_native_single_file_table",
            "version": 1,
            "file_count": len(entries),
            "payload_bytes": offset,
            "files": entries,
        }
        table_bytes = json.dumps(file_table, indent=2).encode("utf-8")

        tmp_path = nxm_path.with_suffix(nxm_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        if nxm_path.exists():
            nxm_path.unlink()

        payload_total = max(1, sum(p.stat().st_size for p in files))
        header_bytes = struct.calcsize(NXM_HEADER_STRUCT) + len(manifest_bytes) + len(table_bytes)
        self.log(f"Packaging header/table: {fmt_bytes(header_bytes)}")
        self.log(f"Packaging payload files: {len(files)} files, {fmt_bytes(payload_total)}")

        with tmp_path.open("wb") as out:
            out.write(struct.pack(NXM_HEADER_STRUCT, NXM_MAGIC, len(manifest_bytes), len(table_bytes), 0))
            out.write(manifest_bytes)
            out.write(table_bytes)

            written = 0
            last_report_pct = -1
            for p in files:
                self.check_cancel()
                rel = p.relative_to(package_dir)
                self.log(f"  pack {rel} ({fmt_bytes(p.stat().st_size)})")
                self.progress(f"Packing .nxm: {rel}", (package_index + 0.97) / max(1, package_count))
                with p.open("rb") as src:
                    while True:
                        chunk = src.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)

                        pct = int((written * 100) / payload_total)
                        if pct >= last_report_pct + 5 or pct == 100:
                            last_report_pct = pct
                            self.log(f"  packaging progress: {pct}% ({fmt_bytes(written)} / {fmt_bytes(payload_total)})")

                        # Keep progress within the final visible slice of this package.
                        pack_frac = 0.97 + 0.03 * min(1.0, written / payload_total)
                        self.progress(
                            f"Packing .nxm: {pct}% ({fmt_bytes(written)} / {fmt_bytes(payload_total)})",
                            (package_index + pack_frac) / max(1, package_count),
                        )

        tmp_path.rename(nxm_path)
        self.progress(f"Packed .nxm: {nxm_path.name}", (package_index + 1.0) / max(1, package_count))
        final_size = nxm_path.stat().st_size
        self.log(f"Packed .nxm: {nxm_path} ({fmt_bytes(final_size)})")
        if final_size > FAT32_FILE_SIZE_WARN_BYTES:
            self.log(f"  NOTE: {nxm_path.name} is at or over 4 GiB -- a FAT32-formatted USB "
                      f"drive will refuse to hold this file. Use exFAT or NTFS for it instead.")

    @staticmethod
    def _weights_stem(package_dir: Path) -> str:
        """Weight files carry the model's name (so a folder of them can be told apart): MyModel.part2of3-000.nxw"""
        n = package_dir.name
        return n[:-len(".nxmodel")] if n.lower().endswith(".nxmodel") else n

    def _build_one_package(self, plan: PackagePlan, package_index: int, package_count: int):
        package_dir = Path(plan.package_dir)
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)

        metadata_dir = package_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        if self.output_settings.copy_metadata_files:
            self._copy_metadata_files(metadata_dir)

        manifest: Dict[str, Any] = {
            "format": "nexus_native_model",
            "version": 2,
            "created_unix": int(time.time()),
            "model": build_model_info(self.scan, plan),
            "package": {
                "role": plan.role,
                "layer_start": plan.layer_start,
                "layer_end": plan.layer_end,
                "package_dir_name": package_dir.name,
            },
            # Compatibility fields kept flat for older experimental loaders.
            "source_path": self.scan.path,
            "source_type": self.scan.source_type,
            "role": plan.role,
            "layer_start": plan.layer_start,
            "layer_end": plan.layer_end,
            "n_layers": infer_n_layers(self.scan),
            "preserves_hf_tensor_names": True,
            "tensor_layout": "safetensors_c_order",
            "native_loader_hints": {
                "container": "directory",
                "manifest": "nexus_model.json",
                "weights": "*.nxw",
                "metadata_dir": "metadata",
                "tensor_names": "original_hf_names",
                "shape_order": "same_as_safetensors",
                "payload_endianness": "little",
                "no_renaming": True,
                "no_fusing": True,
                "no_flattening": True,
                "quantized_axis": "last_dim",
                "runtime_should_read_config": "metadata/config.json",
            },
            "quantization": {
                "group_size": 64,
                "schemes": {
                    "NQ4_G64": "symmetric signed int4, f16 scale per 64 values",
                    "NQ5_G64": "symmetric signed int5, f16 scale per 64 values",
                    "NQ6_G64": "symmetric signed int6, f16 scale per 64 values",
                    "NQ8_G64": "symmetric signed int8, f16 scale per 64 values",
                },
            },
            "settings": self.settings.to_dict(),
            "output_settings": self.output_settings.to_dict(),
            "notes": plan.notes,
            "metadata_files": sorted([str(p.relative_to(package_dir)) for p in metadata_dir.glob("*") if p.is_file()]),
            "parts": [],
            "tensors": [],
        }

        max_part_bytes = int(max(0.1, self.output_settings.max_part_gib) * 1024**3)
        current_part_idx = 0
        current_part_path = package_dir / f"{self._weights_stem(package_dir)}-{current_part_idx:03d}.nxw"
        current = current_part_path.open("wb")
        current_size = 0
        manifest["parts"].append({"file": current_part_path.name, "bytes": 0})

        total_est = max(1, sum(estimated_output_nbytes(t) for t in plan.tensors))
        done_bytes = 0

        try:
            for ti, t in enumerate(plan.tensors):
                self.check_cancel()

                est = estimated_output_nbytes(t)
                if (
                    current_size > 0
                    and current_size + est > max_part_bytes
                    and est < max_part_bytes
                    and self.output_settings.mode == "Multipart by size"
                ):
                    current.close()
                    manifest["parts"][current_part_idx]["bytes"] = current_size
                    current_part_idx += 1
                    current_part_path = package_dir / f"{self._weights_stem(package_dir)}-{current_part_idx:03d}.nxw"
                    current = current_part_path.open("wb")
                    current_size = 0
                    manifest["parts"].append({"file": current_part_path.name, "bytes": 0})

                offset = current_size
                storage_type = decision_to_storage_type(t)
                before = current.tell()

                self.log(f"[{plan.role}] {ti+1}/{len(plan.tensors)} {t.decision} {t.name} {fmt_bytes(est)}")
                self.progress(f"{plan.role}: {t.name}", (package_index + done_bytes / total_est) / max(1, package_count))

                if t.decision == "KEEP_SOURCE":
                    written = self._copy_source_tensor(t, current)
                elif t.decision == "KEEP_F32":
                    written = self._write_f32_tensor(t, current)
                elif t.decision in QUANT_BITS:
                    written = self._write_quant_tensor(t, current, QUANT_BITS[t.decision])
                else:
                    continue

                after = current.tell()
                if written != after - before:
                    written = after - before

                rec = {
                    "name": t.name,
                    "shape": t.shape,
                    "source_dtype": t.dtype,
                    "storage_type": storage_type,
                    "decision": t.decision,
                    "component": t.component,
                    "class": t.tensor_class,
                    "layer": t.layer,
                    "part": current_part_path.name,
                    "offset": offset,
                    "nbytes": written,
                    "source_shard": t.shard,
                    "source_offsets": [t.offset_start, t.offset_end],
                }
                if t.decision in QUANT_BITS:
                    rec["quant"] = {
                        "scheme": storage_type,
                        "bits": QUANT_BITS[t.decision],
                        "group_size": 64,
                        "axis": "last_dim",
                        "scale_dtype": "F16",
                        "zero_point": "signed_offset",
                    }

                manifest["tensors"].append(rec)
                current_size += written
                done_bytes += max(est, written)
                self.progress(f"{plan.role}: {t.name}", (package_index + min(1.0, done_bytes / total_est)) / max(1, package_count))

            current.close()
            manifest["parts"][current_part_idx]["bytes"] = current_size
            actual_tensor_bytes = sum(p["bytes"] for p in manifest["parts"])
            storage_type_counts: Dict[str, int] = {}
            storage_type_bytes: Dict[str, int] = {}
            for rec in manifest["tensors"]:
                st = str(rec.get("storage_type", "UNKNOWN"))
                storage_type_counts[st] = storage_type_counts.get(st, 0) + 1
                storage_type_bytes[st] = storage_type_bytes.get(st, 0) + int(rec.get("nbytes", 0))
            manifest["summary"] = {
                "tensor_count": len(manifest["tensors"]),
                "estimated_tensor_bytes": plan.estimated_tensor_bytes,
                "actual_tensor_bytes": actual_tensor_bytes,
                "parts_count": len(manifest["parts"]),
                "component_counts": count_by(plan.tensors, lambda x: x.component),
                "class_counts": count_by(plan.tensors, lambda x: x.tensor_class),
                "decision_counts": count_by(plan.tensors, lambda x: x.decision),
                "storage_type_counts": storage_type_counts,
                "storage_type_bytes": storage_type_bytes,
            }

            (package_dir / "nexus_model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.log(f"Tensor payload written: {package_dir} ({fmt_bytes(manifest['summary']['actual_tensor_bytes'])})")
        except Exception:
            try:
                current.close()
            except Exception:
                pass
            raise

    def _copy_metadata_files(self, metadata_dir: Path):
        source = Path(self.scan.path)
        if source.is_file():
            source = source.parent
        if not source.exists():
            return

        # Ensure the output directory exists
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for p in source.iterdir():
            if not p.is_file():
                continue
                
            # Skip heavy weights binaries
            if p.suffix.lower() == ".safetensors":
                continue
            if p.name.endswith(".bin") or p.name.endswith(".pt"):
                continue
            if p.stat().st_size > 256 * 1024 * 1024:
                continue
                
            try:
                # Direct copy for all standard metadata files (including merges.txt, tokenizer.json, etc.)
                shutil.copy2(p, metadata_dir / p.name)
                self.log(f"Metadata copied: {p.name}")
            except Exception as e:
                self.log(f"Metadata copy skipped {p.name}: {e}")


        # Automatically extract and generate merges.txt if missing but present in tokenizer.json
        tokenizer_json_path = metadata_dir / "tokenizer.json"
        merges_txt_path = metadata_dir / "merges.txt"

        if tokenizer_json_path.exists() and not merges_txt_path.exists():
            try:
                self.log("Checking tokenizer.json for embedded BPE merges...")
                with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                
                # Check standard HuggingFace tokenizers BPE structure
                model_data = tok_data.get("model", {})
                merges_source = model_data.get("merges")

                if merges_source and isinstance(merges_source, list):
                    merges_lines = []
                    for item in merges_source:
                        if isinstance(item, str):
                            merges_lines.append(item)
                        elif isinstance(item, list) and len(item) == 2:
                            # Handles Qwen3.5/pair array variant
                            merges_lines.append(f"{item[0]} {item[1]}")
                    
                    if merges_lines:
                        with open(merges_txt_path, "w", encoding="utf-8") as f_out:
                            f_out.write("\n".join(merges_lines) + "\n")
                        self.log(f"Generated synthetic merges.txt containing {len(merges_lines)} rules from tokenizer.json")
            except Exception as e:
                self.log(f"Warning: Failed to extract merges from tokenizer.json: {e}")


    def _abs_tensor_start(self, t: TensorInfo) -> int:
        return int(t.data_start + t.offset_start)

    def _copy_source_tensor(self, t: TensorInfo, out):
        with open(t.shard, "rb") as f:
            f.seek(self._abs_tensor_start(t))
            remaining = t.nbytes
            written = 0
            while remaining > 0:
                self.check_cancel()
                chunk = f.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"Unexpected EOF reading {t.name}")
                out.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
            return written

    def _source_to_float32(self, raw: bytes, dtype: str):
        import numpy as np
        if dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        if dtype == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            u32 = u16 << 16
            return u32.view(np.float32)
        raise ValueError(f"Unsupported source dtype for conversion: {dtype}")

    def _write_f32_tensor(self, t: TensorInfo, out):
        if t.dtype == "F32":
            return self._copy_source_tensor(t, out)

        item_bytes = DTYPE_BYTES.get(t.dtype)
        if not item_bytes:
            raise ValueError(f"Unsupported dtype for F32 conversion: {t.dtype}")

        written = 0
        with open(t.shard, "rb") as f:
            f.seek(self._abs_tensor_start(t))
            remaining = t.nbytes
            chunk_bytes = max(item_bytes, int(self.output_settings.chunk_mib) * 1024 * 1024)
            chunk_bytes -= chunk_bytes % item_bytes
            while remaining > 0:
                self.check_cancel()
                raw = f.read(min(chunk_bytes, remaining))
                if not raw:
                    raise IOError(f"Unexpected EOF reading {t.name}")
                arr = self._source_to_float32(raw, t.dtype)
                b = arr.astype("<f4", copy=False).tobytes()
                out.write(b)
                written += len(b)
                remaining -= len(raw)
        return written

    def _write_quant_tensor(self, t: TensorInfo, out, bits: int):
        if t.ndim < 2:
            # Safety: quantizing a vector is almost never what we want.
            return self._copy_source_tensor(t, out)

        if t.dtype not in ("F32", "F16", "BF16"):
            raise ValueError(f"Unsupported dtype for quantization: {t.dtype}")

        import numpy as np

        cols = int(t.shape[-1])
        rows = int(t.elements // max(1, cols))
        item_bytes = DTYPE_BYTES[t.dtype]
        row_bytes = cols * item_bytes
        target_chunk_bytes = max(row_bytes, int(self.output_settings.chunk_mib) * 1024 * 1024)
        rows_per_chunk = max(1, min(rows, target_chunk_bytes // max(1, row_bytes)))

        written = 0
        abs_start = self._abs_tensor_start(t)

        with open(t.shard, "rb") as f:
            for r0 in range(0, rows, rows_per_chunk):
                self.check_cancel()
                nr = min(rows_per_chunk, rows - r0)
                f.seek(abs_start + r0 * row_bytes)
                raw = f.read(nr * row_bytes)
                if len(raw) != nr * row_bytes:
                    raise IOError(f"Unexpected EOF reading rows for {t.name}")

                x = self._source_to_float32(raw, t.dtype).reshape(nr, cols)
                block_bytes = self._quantize_rows_g64(x, bits)
                out.write(block_bytes)
                written += len(block_bytes)

        return written

    def _pack_unsigned_bits(self, q, bits: int):
        import numpy as np

        q = q.astype(np.uint16, copy=False)
        ng, group = q.shape
        packed_len = (group * bits + 7) // 8
        out = np.zeros((ng, packed_len), dtype=np.uint8)

        for i in range(group):
            bit = i * bits
            byte = bit >> 3
            shift = bit & 7
            v = q[:, i]
            out[:, byte] |= ((v << shift) & 0xFF).astype(np.uint8)
            spill = shift + bits - 8
            if spill > 0:
                out[:, byte + 1] |= ((v >> (8 - shift)) & 0xFF).astype(np.uint8)
        return out

    def _quantize_rows_g64(self, x, bits: int) -> bytes:
        import numpy as np

        group = 64
        rows, cols = x.shape
        groups_per_row = (cols + group - 1) // group
        padded_cols = groups_per_row * group

        if padded_cols != cols:
            xp = np.zeros((rows, padded_cols), dtype=np.float32)
            xp[:, :cols] = x
        else:
            xp = x.astype(np.float32, copy=False)

        g = xp.reshape(rows * groups_per_row, group)

        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
        offset = 1 << (bits - 1)

        max_abs = np.max(np.abs(g), axis=1).astype(np.float32)
        scale = max_abs / float(qmax)
        scale[scale == 0.0] = 1.0

        q = np.rint(g / scale[:, None])
        q = np.clip(q, qmin, qmax).astype(np.int16)
        qu = (q + offset).astype(np.uint16)

        packed = self._pack_unsigned_bits(qu, bits)
        scales = scale.astype("<f2").view(np.uint8).reshape(-1, 2)

        rec_size = 2 + packed.shape[1]
        records = np.empty((packed.shape[0], rec_size), dtype=np.uint8)
        records[:, 0:2] = scales
        records[:, 2:] = packed
        return records.tobytes()


# ----------------------------
# GUI
# ----------------------------

class NexusNativeQuantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nexus Native Quant Builder")
        self.geometry("1500x860")
        self.minsize(1180, 700)

        self.settings = QuantSettings()
        self.output_settings = OutputSettings()
        self.scan: Optional[ScanResult] = None
        self.filter_manual_only = False
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.build_thread: Optional[threading.Thread] = None

        self._build_style()
        self._build_ui()
        self.after(150, self._poll_log_queue)

    def _build_style(self):
        self.configure(bg="#171922")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#171922", foreground="#e8eaf0", fieldbackground="#10131a")
        style.configure("Treeview", background="#0e1017", foreground="#e8eaf0", fieldbackground="#0e1017", rowheight=24)
        style.configure("Treeview.Heading", background="#252938", foreground="#e8eaf0")
        style.map("Treeview", background=[("selected", "#3b3278")])
        style.configure("TButton", padding=5)
        style.configure("TLabel", background="#171922", foreground="#e8eaf0")
        style.configure("TFrame", background="#171922")
        style.configure("TLabelframe", background="#171922", foreground="#e8eaf0")
        style.configure("TLabelframe.Label", background="#171922", foreground="#aab0c4")

        # *** fix: readability of the dropdowns (closed field AND the open list).
        # The "." style above covers plain ttk widgets, but ttk's Combobox splits
        # into two pieces that "." doesn't reach: the closed field (its own ttk
        # element, themed separately per-state) and the open dropdown list, which
        # is a *plain Tk Listbox* underneath, not a ttk widget at all -- it only
        # takes colors through the classic option database (option_add), never
        # through ttk Style. Leaving both at their theme/system defaults is what
        # produced light-grey panels with light text and poor contrast.
        style.configure("TCombobox", fieldbackground="#20232e", background="#20232e",
                         foreground="#e8eaf0", arrowcolor="#e8eaf0", selectforeground="#ffffff",
                         selectbackground="#3b3278")
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#20232e"), ("disabled", "#171922")],
                  foreground=[("readonly", "#e8eaf0"), ("disabled", "#6a6f82")],
                  background=[("readonly", "#20232e")])
        self.option_add("*TCombobox*Listbox.background", "#20232e")
        self.option_add("*TCombobox*Listbox.foreground", "#e8eaf0")
        self.option_add("*TCombobox*Listbox.selectBackground", "#3b3278")
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Button(top, text="Open HF Folder / Safetensors", command=self.open_path).pack(side="left")
        ttk.Button(top, text="Rescan", command=self.rescan).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Export Policy JSON", command=self.export_policy).pack(side="left", padx=(18, 0))
        ttk.Button(top, text="Import Policy JSON", command=self.import_policy).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="BUILD", command=self.start_build).pack(side="left", padx=(18, 0))

        self.path_var = tk.StringVar(value="No model loaded")
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=(14, 0), fill="x", expand=True)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=5)

        self._build_left(left)
        self._build_right(right)

        # *** fix: the left column opened too narrow to read its own labels/values
        # until you dragged it wider by hand. A ttk.PanedWindow's initial sash
        # position isn't driven by its children's natural/requested size the way
        # pack() would -- it just splits on weight, which gave the left pane only
        # a sliver. Setting the sash explicitly, once the window has a real size,
        # gives a sane starting width without changing anything about how it
        # resizes afterward (you can still drag it either way).
        self.after(0, lambda: main.sashpos(0, 430))

    def _build_left(self, parent):
        # The left control column can be taller than the window. Put it inside
        # a scrollable canvas so the Build section can never disappear.
        canvas = tk.Canvas(parent, bg="#171922", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_frame_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            # Linux usually sends Button-4/5; Windows/macOS usually send MouseWheel.
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_rules(scroll_frame)
        self._build_output(scroll_frame)
        self._build_build(scroll_frame)
        self._build_summary(scroll_frame)

    def _combo_row(self, parent, label, attr, values):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar(value=getattr(self.settings, attr))
        cb = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=12)
        cb.pack(side="right")
        setattr(self, f"var_{attr}", var)

    def _build_rules(self, parent):
        frm = ttk.LabelFrame(parent, text="Auto Quant Rules")
        frm.pack(fill="x", pady=(0, 8))

        qvals = ["Q4G64", "Q5G64", "Q6G64", "Q8G64", "KEEP_SOURCE", "KEEP_F32"]
        self._combo_row(frm, "Default matrix", "default_quant", qvals)
        self._combo_row(frm, "Embeddings", "embedding_quant", qvals)
        self._combo_row(frm, "Output/lm_head", "output_quant", qvals)
        self._combo_row(frm, "FFN", "ffn_quant", qvals)
        self._combo_row(frm, "Attention", "attention_quant", qvals)
        self._combo_row(frm, "SSM matrix", "ssm_matrix_quant", qvals)
        self._combo_row(frm, "Full precision", "full_precision_decision", ["KEEP_SOURCE", "KEEP_F32"])

        row = ttk.Frame(frm)
        row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Min matrix MiB", width=18).pack(side="left")
        self.var_min_mib = tk.DoubleVar(value=self.settings.min_matrix_mib)
        tk.Spinbox(row, from_=0.0, to=4096.0, increment=0.25, textvariable=self.var_min_mib, width=8).pack(side="right")

        comp = ttk.LabelFrame(parent, text="Components")
        comp.pack(fill="x", pady=(0, 8))
        self.var_include_language = tk.BooleanVar(value=True)
        self.var_include_visual = tk.BooleanVar(value=True)
        self.var_include_audio = tk.BooleanVar(value=True)
        self.var_include_mtp = tk.BooleanVar(value=True)
        self.var_include_other = tk.BooleanVar(value=True)
        for text, var in [
            ("Language model", self.var_include_language),
            ("Visual tensors", self.var_include_visual),
            ("Audio/speech tensors", self.var_include_audio),
            ("MTP/speculative tensors", self.var_include_mtp),
            ("Other globals", self.var_include_other),
        ]:
            ttk.Checkbutton(comp, text=text, variable=var).pack(anchor="w", padx=8, pady=1)

        keep = ttk.LabelFrame(parent, text="Keep Source Precision")
        keep.pack(fill="x", pady=(0, 8))
        self.var_keep_1d = tk.BooleanVar(value=True)
        self.var_keep_norms = tk.BooleanVar(value=True)
        self.var_keep_biases = tk.BooleanVar(value=True)
        self.var_keep_ssm_control = tk.BooleanVar(value=True)
        self.var_keep_conv1d = tk.BooleanVar(value=True)
        self.var_keep_small = tk.BooleanVar(value=True)
        for text, var in [
            ("1D tensors", self.var_keep_1d),
            ("Norms", self.var_keep_norms),
            ("Biases", self.var_keep_biases),
            ("SSM/DeltaNet controls", self.var_keep_ssm_control),
            ("Conv1D kernels", self.var_keep_conv1d),
            ("Below threshold", self.var_keep_small),
        ]:
            ttk.Checkbutton(keep, text=text, variable=var).pack(anchor="w", padx=8, pady=1)

        ttk.Button(parent, text="Apply Auto Decisions", command=self.apply_auto).pack(fill="x", pady=(0, 5))
        ttk.Button(parent, text="Clear Manual Overrides", command=self.clear_manual).pack(fill="x", pady=(0, 5))

        manual = ttk.LabelFrame(parent, text="Manual Decision")
        manual.pack(fill="x", pady=(4, 8))
        self.manual_decision_var = tk.StringVar(value="Q4G64")
        ttk.Combobox(manual, textvariable=self.manual_decision_var, values=DECISIONS[1:], state="readonly").pack(fill="x", padx=8, pady=5)
        ttk.Button(manual, text="Set Selected", command=self.set_selected_decision).pack(fill="x", padx=8, pady=(0, 5))
        ttk.Button(manual, text="Revert Selected to Auto", command=self.revert_selected_auto).pack(fill="x", padx=8, pady=(0, 8))

    def _build_output(self, parent):
        frm = ttk.LabelFrame(parent, text="Native Output")
        frm.pack(fill="x", pady=(0, 8))

        self.var_output_mode = tk.StringVar(value=self.output_settings.mode)
        self.var_output_basename = tk.StringVar(value=self.output_settings.output_basename)
        self.var_output_dir = tk.StringVar(value=str(Path.home()))
        self.var_max_part_gib = tk.DoubleVar(value=self.output_settings.max_part_gib)
        self.var_remote_tail_layers = tk.IntVar(value=0)
        self.var_duplicate_embeddings = tk.BooleanVar(value=False)
        self.var_chunk_mib = tk.IntVar(value=32)
        self.var_split_count = tk.IntVar(value=2)
        self.var_split_layers = tk.StringVar(value="")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Mode", width=18).pack(side="left")
        ttk.Combobox(row, textvariable=self.var_output_mode, values=[
            "Single native file",
            "Native directory package",
            "Multipart by size",
            "Remote two-computer split",
            SPLIT_MODE,
        ], state="readonly", width=28).pack(side="right")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Base name", width=18).pack(side="left")
        ttk.Entry(row, textvariable=self.var_output_basename, width=24).pack(side="right")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Output folder", width=18).pack(side="left")
        ttk.Button(row, text="Browse…", command=self.browse_output_dir).pack(side="right")
        ttk.Entry(row, textvariable=self.var_output_dir, width=24).pack(side="right", padx=(0, 4))

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Max part GiB", width=18).pack(side="left")
        tk.Spinbox(row, from_=0.25, to=2048.0, increment=0.25, textvariable=self.var_max_part_gib, width=8).pack(side="right")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Remote tail layers", width=18).pack(side="left")
        tk.Spinbox(row, from_=0, to=9999, increment=1, textvariable=self.var_remote_tail_layers, width=8).pack(side="right")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Chunk MiB", width=18).pack(side="left")
        tk.Spinbox(row, from_=4, to=512, increment=4, textvariable=self.var_chunk_mib, width=8).pack(side="right")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Pieces (computers)", width=18).pack(side="left")
        tk.Spinbox(row, from_=2, to=64, increment=1, textvariable=self.var_split_count, width=8).pack(side="right")

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text="Layers per piece", width=18).pack(side="left")
        ttk.Entry(row, textvariable=self.var_split_layers, width=24).pack(side="right")
        ttk.Label(frm, text="Split mode: every piece is its own self-contained .nxm (model description, tokenizer and chat template inside).\n"
                            "Layers per piece: e.g. 20,22,22 - blank = balanced by size (piece 1 also holds the embeddings and output layer).\n"
                            "Each tensor is stored exactly once; nothing is duplicated.",
                  justify="left").pack(anchor="w", padx=8, pady=(0, 2))

        ttk.Button(frm, text="Refresh Estimate", command=self.refresh_estimates).pack(fill="x", padx=8, pady=(6, 8))

        self.plan_text = tk.Text(frm, height=9, bg="#0e1017", fg="#d8dbea", insertbackground="white", relief="flat", wrap="word")
        self.plan_text.pack(fill="both", padx=6, pady=(0, 6))

    def _build_build(self, parent):
        frm = ttk.LabelFrame(parent, text="Build")
        frm.pack(fill="x", pady=(0, 8))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(frm, variable=self.progress_var, maximum=1.0)
        self.progress.pack(fill="x", padx=8, pady=(8, 3))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).pack(anchor="w", padx=8)

        row = ttk.Frame(frm); row.pack(fill="x", padx=8, pady=6)
        self.build_btn = ttk.Button(row, text="Build Native Model", command=self.start_build)
        self.build_btn.pack(side="left", fill="x", expand=True)
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.cancel_build)
        self.cancel_btn.pack(side="left", padx=(6, 0))

    def _build_summary(self, parent):
        frm = ttk.LabelFrame(parent, text="Summary")
        frm.pack(fill="both", expand=True)
        self.summary_text = tk.Text(frm, height=10, bg="#0e1017", fg="#d8dbea", insertbackground="white", relief="flat", wrap="word")
        self.summary_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_right(self, parent):
        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(0, 6))

        ttk.Label(tools, text="Filter").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.refresh_table())
        ttk.Entry(tools, textvariable=self.filter_var, width=40).pack(side="left", padx=(6, 10))

        ttk.Label(tools, text="Component").pack(side="left")
        self.component_filter = tk.StringVar(value="all")
        cb = ttk.Combobox(tools, textvariable=self.component_filter, values=["all", COMPONENT_LANGUAGE, COMPONENT_VISUAL, COMPONENT_AUDIO, COMPONENT_MTP, COMPONENT_OTHER], state="readonly", width=12)
        cb.pack(side="left", padx=(6, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())

        ttk.Label(tools, text="Decision").pack(side="left")
        self.decision_filter = tk.StringVar(value="all")
        cb = ttk.Combobox(tools, textvariable=self.decision_filter, values=["all"] + DECISIONS[1:], state="readonly", width=14)
        cb.pack(side="left", padx=(6, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())

        ttk.Button(tools, text="Manual Only", command=self.show_manual_only).pack(side="left", padx=(8, 0))
        ttk.Button(tools, text="Clear Filters", command=self.clear_filters).pack(side="left", padx=(6, 0))

        cols = ("layer", "component", "class", "decision", "auto", "src", "src_mib", "out_mib", "shape", "name", "reason")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="extended")
        self.tree.pack(fill="both", expand=True, side="left")

        heads = {
            "layer": "Layer",
            "component": "Component",
            "class": "Class",
            "decision": "Decision",
            "auto": "Auto",
            "src": "Src",
            "src_mib": "Src MiB",
            "out_mib": "Out MiB",
            "shape": "Shape",
            "name": "Tensor",
            "reason": "Reason",
        }
        widths = {
            "layer": 58, "component": 90, "class": 110, "decision": 100, "auto": 100,
            "src": 70, "src_mib": 80, "out_mib": 80, "shape": 150,
            "name": 470, "reason": 340,
        }
        for c in cols:
            self.tree.heading(c, text=heads[c], command=lambda cc=c: self.sort_by(cc))
            self.tree.column(c, width=widths[c], anchor="w")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.bind("<Double-1>", lambda e: self.set_selected_decision())

        log_frame = ttk.LabelFrame(parent, text="Build Log")
        log_frame.pack(fill="x", pady=(8, 0))
        self.log_widget = scrolledtext.ScrolledText(log_frame, height=8, bg="#090b10", fg="#c8cad8", insertbackground="white", relief="flat")
        self.log_widget.pack(fill="both", expand=True)

    # ----------------------------
    # UI state
    # ----------------------------

    def update_settings_from_ui(self):
        for attr in ["default_quant", "embedding_quant", "output_quant", "ffn_quant", "attention_quant", "ssm_matrix_quant", "full_precision_decision"]:
            getattr(self.settings, attr)
            setattr(self.settings, attr, getattr(self, f"var_{attr}").get())
        self.settings.min_matrix_mib = float(self.var_min_mib.get())
        self.settings.keep_1d = bool(self.var_keep_1d.get())
        self.settings.keep_norms = bool(self.var_keep_norms.get())
        self.settings.keep_biases = bool(self.var_keep_biases.get())
        self.settings.keep_ssm_control = bool(self.var_keep_ssm_control.get())
        self.settings.keep_conv1d = bool(self.var_keep_conv1d.get())
        self.settings.keep_small_tensors = bool(self.var_keep_small.get())
        self.settings.include_language = bool(self.var_include_language.get())
        self.settings.include_visual = bool(self.var_include_visual.get())
        self.settings.include_audio = bool(self.var_include_audio.get())
        self.settings.include_mtp = bool(self.var_include_mtp.get())
        self.settings.include_other = bool(self.var_include_other.get())

    def update_output_from_ui(self):
        self.output_settings.mode = self.var_output_mode.get()
        self.output_settings.output_basename = self.var_output_basename.get().strip() or "model-nexus-native"
        self.output_settings.max_part_gib = float(self.var_max_part_gib.get())
        self.output_settings.remote_tail_layers = int(self.var_remote_tail_layers.get())
        self.output_settings.duplicate_embeddings_to_remote = bool(self.var_duplicate_embeddings.get())
        self.output_settings.chunk_mib = int(self.var_chunk_mib.get())
        self.output_settings.split_count = int(self.var_split_count.get())
        self.output_settings.split_layers = self.var_split_layers.get().strip()

    # ----------------------------
    # Scanning/actions
    # ----------------------------

    def open_path(self):
        choice = messagebox.askyesnocancel(
            "Open Model",
            "Open a folder?\n\nYes = HF model folder with safetensors shards\nNo = single .safetensors file\nCancel = stop"
        )
        if choice is None:
            return
        if choice:
            p = filedialog.askdirectory(title="Choose HF model folder")
        else:
            p = filedialog.askopenfilename(title="Choose safetensors", filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
        if p:
            self.load_path(Path(p))

    def rescan(self):
        if not self.scan:
            self.open_path()
        else:
            self.load_path(Path(self.scan.path))

    def load_path(self, path: Path):
        self.path_var.set(f"Scanning {path}")
        self.status_var.set("Scanning...")
        self.update_idletasks()

        def worker():
            try:
                result = scan_path(path)
                self.after(0, lambda: self.on_scan_complete(result))
            except Exception as e:
                tb = traceback.format_exc()
                err_msg = str(e)
                self.after(0, lambda: messagebox.showerror("Scan failed", f"{err_msg}\n\n{tb}"))

        threading.Thread(target=worker, daemon=True).start()

    def on_scan_complete(self, result: ScanResult):
        self.scan = result
        self.path_var.set(f"{result.source_type}: {result.path} | shards/files: {result.metadata.get('source_shard_count', '?')}")
        self.status_var.set("Scan complete")
        # *** fix: default the output name to the model you just opened instead of
        # the generic "model-nexus-native" every build used to start with. Still
        # just a text field afterward -- type over it if you want something else.
        self.var_output_basename.set(derive_basename(Path(result.path)))
        self.apply_auto()

    def apply_auto(self):
        if not self.scan:
            return
        self.update_settings_from_ui()
        for t in self.scan.tensors:
            d, reason = auto_decide(t, self.settings)
            t.auto_decision = d
            if not t.manual:
                t.decision = d
                t.reason = reason
            else:
                t.reason = "manual override"
        self.refresh_all()

    def clear_manual(self):
        if not self.scan:
            return
        for t in self.scan.tensors:
            t.manual = False
            t.decision = t.auto_decision
            _, reason = auto_decide(t, self.settings)
            t.reason = reason
        self.refresh_all()

    def selected_indices(self) -> List[int]:
        out = []
        for item in self.tree.selection():
            tags = self.tree.item(item, "tags")
            if tags:
                out.append(int(tags[0]))
        return out

    def set_selected_decision(self):
        if not self.scan:
            return
        d = self.manual_decision_var.get()
        for idx in self.selected_indices():
            t = self.scan.tensors[idx]
            t.decision = d
            t.manual = True
            t.reason = "manual override"
        self.refresh_all()

    def revert_selected_auto(self):
        if not self.scan:
            return
        for idx in self.selected_indices():
            t = self.scan.tensors[idx]
            t.manual = False
            t.decision = t.auto_decision
            _, reason = auto_decide(t, self.settings)
            t.reason = reason
        self.refresh_all()

    def browse_output_dir(self):
        d = filedialog.askdirectory(title="Choose output folder", initialdir=self.var_output_dir.get())
        if d:
            self.var_output_dir.set(d)
            self.refresh_estimates()

    def show_manual_only(self):
        self.filter_manual_only = True
        self.refresh_table()

    def clear_filters(self):
        self.filter_var.set("")
        self.component_filter.set("all")
        self.decision_filter.set("all")
        self.filter_manual_only = False
        self.refresh_table()

    def refresh_all(self):
        self.refresh_table()
        self.update_summary()
        self.refresh_estimates()

    def refresh_table(self):
        if not self.scan:
            return
        q = self.filter_var.get().strip().lower()
        cf = self.component_filter.get()
        df = self.decision_filter.get()

        self.tree.delete(*self.tree.get_children())
        for idx, t in enumerate(self.scan.tensors):
            if self.filter_manual_only and not t.manual:
                continue
            if cf != "all" and t.component != cf:
                continue
            if df != "all" and t.decision != df:
                continue
            if q and q not in f"{t.name} {t.component} {t.tensor_class} {t.decision} {t.reason}".lower():
                continue

            self.tree.insert("", "end", values=(
                "" if t.layer is None else t.layer,
                t.component,
                t.tensor_class,
                ("*" if t.manual else "") + t.decision,
                t.auto_decision,
                t.dtype,
                f"{t.mib:.2f}",
                f"{estimated_output_nbytes(t) / 1024**2:.2f}",
                format_shape(t.shape),
                t.name,
                t.reason,
            ), tags=(str(idx),))

    def sort_by(self, col):
        if not self.scan:
            return
        key_map = {
            "layer": lambda t: -1 if t.layer is None else t.layer,
            "component": lambda t: t.component,
            "class": lambda t: t.tensor_class,
            "decision": lambda t: t.decision,
            "auto": lambda t: t.auto_decision,
            "src": lambda t: t.dtype,
            "src_mib": lambda t: t.mib,
            "out_mib": lambda t: estimated_output_nbytes(t),
            "shape": lambda t: format_shape(t.shape),
            "name": lambda t: t.name,
            "reason": lambda t: t.reason,
        }
        self.scan.tensors.sort(key=key_map.get(col, lambda t: t.name))
        self.refresh_table()

    def update_summary(self):
        if not self.scan:
            return
        s = summarize(self.scan)
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", f"Shards/files: {self.scan.metadata.get('source_shard_count', '?')}\n")
        self.summary_text.insert("end", f"Tensors scanned: {s['tensor_count']}\n")
        self.summary_text.insert("end", f"Language layers: {s['n_layers']}\n")
        self.summary_text.insert("end", f"Source files: {fmt_bytes(s['source_file_bytes'])}\n")
        self.summary_text.insert("end", f"Source tensor payload: {fmt_bytes(s['source_tensor_bytes'])}\n")
        self.summary_text.insert("end", f"Estimated selected output: {fmt_bytes(s['estimated_output_bytes'])}\n\n")

        self.summary_text.insert("end", "By decision:\n")
        for k, v in sorted(s["by_decision"].items()):
            self.summary_text.insert("end", f"  {k}: {v}\n")
        self.summary_text.insert("end", "\nBy component:\n")
        for k, v in sorted(s["by_component"].items()):
            self.summary_text.insert("end", f"  {k}: {v}\n")
        self.summary_text.insert("end", "\nBy class:\n")
        for k, v in sorted(s["by_class"].items()):
            self.summary_text.insert("end", f"  {k}: {v}\n")

    def refresh_estimates(self):
        if not self.scan:
            return
        self.update_settings_from_ui()
        self.update_output_from_ui()

        # Recompute auto decisions here too. Previously the checkbox state only
        # changed settings; the visible estimate did not change until Apply Auto
        # Decisions or Build ran. This keeps Refresh Estimate honest without
        # calling apply_auto(), which would recurse back into refresh_all().
        for t in self.scan.tensors:
            d, reason = auto_decide(t, self.settings)
            t.auto_decision = d
            if not t.manual:
                t.decision = d
                t.reason = reason

        output_root = Path(self.var_output_dir.get()).expanduser()
        plans = build_package_plans(self.scan, self.settings, self.output_settings, output_root)
        s = summarize(self.scan)

        self.plan_text.delete("1.0", "end")
        self.plan_text.insert("end", f"Source files: {fmt_bytes(s['source_file_bytes'])}\n")
        self.plan_text.insert("end", f"Estimated selected output: {fmt_bytes(s['estimated_output_bytes'])}\n")

        component_bytes: Dict[str, int] = {}
        component_counts: Dict[str, int] = {}
        for t in self.scan.tensors:
            component_counts[t.component] = component_counts.get(t.component, 0) + 1
            component_bytes[t.component] = component_bytes.get(t.component, 0) + estimated_output_nbytes(t)
        self.plan_text.insert("end", "Selected bytes by component:\n")
        for comp in [COMPONENT_LANGUAGE, COMPONENT_VISUAL, COMPONENT_AUDIO, COMPONENT_MTP, COMPONENT_OTHER]:
            self.plan_text.insert("end", f"  {comp}: {component_counts.get(comp, 0)} tensors, {fmt_bytes(component_bytes.get(comp, 0))}\n")

        self.plan_text.insert("end", f"Output root: {output_root}\n")
        if self.output_settings.mode == "Single native file":
            est_final = s['estimated_output_bytes'] + 1024 * 1024  # rough manifest/table/metadata cushion
            self.plan_text.insert("end", f"Final file: {self.output_settings.output_basename}.nxm (~{fmt_bytes(est_final)})\n")
            self.plan_text.insert("end", "Build flow: temporary .nxmodel folder -> packaged .nxm -> temp folder removed\n")
        self.plan_text.insert("end", "\n")
        single_file_mode = self.output_settings.mode in ("Single native file", SPLIT_MODE)
        for p in plans:
            layer_range = ""
            if p.layer_start is not None:
                layer_range = f" layers {p.layer_start}-{max(p.layer_start, (p.layer_end or p.layer_start)-1)}"
            self.plan_text.insert("end", f"{Path(p.package_dir).name}\n")
            self.plan_text.insert("end", f"  role: {p.role}{layer_range}\n")
            self.plan_text.insert("end", f"  tensors: {len(p.tensors)}\n")
            self.plan_text.insert("end", f"  est payload: {fmt_bytes(p.estimated_tensor_bytes)}\n")
            for note in list(dict.fromkeys(p.notes))[:4]:
                self.plan_text.insert("end", f"  note: {note}\n")
            # *** fix: single-file modes (Single native file / split-by-computer) each pack
            # down to ONE .nxm file. FAT32 -- still the default format a lot of USB sticks
            # ship pre-formatted as -- caps any single file at just under 4 GiB. A plain
            # "No space left on device" partway through a USB copy with plenty of free space
            # left is this limit, not a full drive, so flag it here where it's decided, not
            # after someone's already hit it.
            if single_file_mode and p.estimated_tensor_bytes > FAT32_FILE_SIZE_WARN_BYTES:
                self.plan_text.insert(
                    "end",
                    f"  NOTE: this piece's .nxm will be at or over 4 GiB -- won't fit on a "
                    f"FAT32-formatted USB drive. Use exFAT or NTFS for the USB release, or "
                    f"split into more pieces.\n")
            self.plan_text.insert("end", "\n")

    # ----------------------------
    # Export/import
    # ----------------------------

    def export_policy(self):
        if not self.scan:
            messagebox.showwarning("No model", "Scan first.")
            return
        self.update_settings_from_ui()
        self.update_output_from_ui()
        p = filedialog.asksaveasfilename(defaultextension=".json", initialfile="nexus_native_quant_policy.json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        data = {
            "format": "nexus_native_quant_policy",
            "version": 1,
            "source_path": self.scan.path,
            "source_type": self.scan.source_type,
            "settings": self.settings.to_dict(),
            "output_settings": self.output_settings.to_dict(),
            "summary": summarize(self.scan),
            "tensors": [
                {
                    "name": t.name,
                    "shape": t.shape,
                    "dtype": t.dtype,
                    "shard": t.shard,
                    "nbytes": t.nbytes,
                    "layer": t.layer,
                    "component": t.component,
                    "class": t.tensor_class,
                    "auto_decision": t.auto_decision,
                    "decision": t.decision,
                    "manual": t.manual,
                    "reason": t.reason,
                } for t in self.scan.tensors
            ],
        }
        Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")
        messagebox.showinfo("Exported", p)

    def import_policy(self):
        if not self.scan:
            messagebox.showwarning("No model", "Scan matching model first.")
            return
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        data = json.loads(Path(p).read_text(encoding="utf-8"))

        settings = data.get("settings", {})
        for k, v in settings.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
                var_name = f"var_{k}"
                if hasattr(self, var_name):
                    getattr(self, var_name).set(v)

        out_settings = data.get("output_settings", {})
        if out_settings:
            self.var_output_mode.set(out_settings.get("mode", self.var_output_mode.get()))
            self.var_output_basename.set(out_settings.get("output_basename", self.var_output_basename.get()))
            self.var_max_part_gib.set(out_settings.get("max_part_gib", self.var_max_part_gib.get()))
            self.var_remote_tail_layers.set(out_settings.get("remote_tail_layers", self.var_remote_tail_layers.get()))
            self.var_duplicate_embeddings.set(out_settings.get("duplicate_embeddings_to_remote", self.var_duplicate_embeddings.get()))
            self.var_chunk_mib.set(out_settings.get("chunk_mib", self.var_chunk_mib.get()))
            self.var_split_count.set(out_settings.get("split_count", self.var_split_count.get()))
            self.var_split_layers.set(out_settings.get("split_layers", self.var_split_layers.get()))

        by_name = {r.get("name"): r for r in data.get("tensors", [])}
        count = 0
        for t in self.scan.tensors:
            r = by_name.get(t.name)
            if not r:
                continue
            t.decision = self._normalize_decision(r.get("decision", t.decision))
            t.manual = bool(r.get("manual", True))
            t.reason = "imported policy"
            count += 1

        self.refresh_all()
        messagebox.showinfo("Imported", f"Imported {count} tensor decisions.")

    def _normalize_decision(self, d: str) -> str:
        mapping = {
            "KEEP_F16": "KEEP_SOURCE",
            "KEEP_BF16": "KEEP_SOURCE",
            "NQ4_G64": "Q4G64",
            "NQ5_G64": "Q5G64",
            "NQ6_G64": "Q6G64",
            "NQ8_G64": "Q8G64",
        }
        return mapping.get(d, d if d in DECISIONS else "KEEP_SOURCE")

    def export_csv(self):
        if not self.scan:
            messagebox.showwarning("No model", "Scan first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="nexus_native_quant_report.csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "component", "layer", "class", "shape", "dtype", "source_bytes", "estimated_output_bytes", "decision", "auto_decision", "manual", "reason", "shard"])
            for t in self.scan.tensors:
                w.writerow([t.name, t.component, t.layer, t.tensor_class, format_shape(t.shape), t.dtype, t.nbytes, estimated_output_nbytes(t), t.decision, t.auto_decision, t.manual, t.reason, t.shard])
        messagebox.showinfo("Exported", p)

    # ----------------------------
    # Build
    # ----------------------------

    def start_build(self):
        if not self.scan:
            messagebox.showwarning("No model", "Scan a model first.")
            return
        if self.build_thread and self.build_thread.is_alive():
            messagebox.showinfo("Build running", "A build is already running.")
            return

        self.update_settings_from_ui()
        self.update_output_from_ui()
        output_root = Path(self.var_output_dir.get()).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        # Apply auto once so component checkboxes/threshold changes are reflected.
        self.apply_auto()

        self.cancel_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("Building...")
        self.build_btn.configure(state="disabled")
        self._log("=== Build started ===")

        def worker():
            try:
                builder = NativePackageBuilder(
                    scan=self.scan,
                    settings=self.settings,
                    output_settings=self.output_settings,
                    output_root=output_root,
                    log=lambda s: self.log_queue.put(("log", s)),
                    progress=lambda s, p: self.log_queue.put(("progress", s, p)),
                    cancel_event=self.cancel_event,
                )
                built = builder.build()
                self.log_queue.put(("done", built))
            except BuildCancelled:
                self.log_queue.put(("cancelled",))
            except Exception:
                self.log_queue.put(("error", traceback.format_exc()))

        self.build_thread = threading.Thread(target=worker, daemon=True)
        self.build_thread.start()

    def cancel_build(self):
        self.cancel_event.set()
        self.status_var.set("Cancelling...")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if not item:
                    continue
                kind = item[0]
                if kind == "log":
                    self._log(item[1])
                elif kind == "progress":
                    _, status, p = item
                    self.status_var.set(status)
                    self.progress_var.set(max(0.0, min(1.0, float(p))))
                elif kind == "done":
                    _, built = item
                    self.progress_var.set(1.0)
                    self.status_var.set("Build complete")
                    self.build_btn.configure(state="normal")
                    self._log("=== Build complete ===")
                    for b in built:
                        self._log(f"Built: {b}")
                    messagebox.showinfo("Build complete", "\n".join(built))
                elif kind == "cancelled":
                    self.status_var.set("Cancelled")
                    self.build_btn.configure(state="normal")
                    self._log("=== Build cancelled ===")
                elif kind == "error":
                    self.status_var.set("Build failed")
                    self.build_btn.configure(state="normal")
                    self._log(item[1])
                    messagebox.showerror("Build failed", item[1])
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    def _log(self, text: str):
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", str(text) + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")


def main():
    app = NexusNativeQuantApp()
    app.mainloop()


if __name__ == "__main__":
    main()
