#!/usr/bin/env python3
"""Reconstructs the exact PerceptionSuccessProbe architecture for a trained run directory,
so a sweep of many runs (varying pool_type/key_dim/n_queries/modalities/etc., not just the
one hand-run default-architecture model compute_calibration_scores.py originally assumed)
can each be loaded correctly instead of silently instantiating PerceptionSuccessProbe()
defaults and getting a shape-mismatched (or, if shapes happen to coincide, silently wrong)
model.

Three tiers, tried in order per run_dir:

  1. command.txt -- sweep.py's run_cell() writes the exact `train_probe_time_dependent.py`
     invocation as a sibling of the run's timestamped output dir. If present, this is 100%
     ground truth: re-parse it with train_probe_time_dependent.parse_args(argv) and build the
     model with train_probe_time_dependent.build_model(args) -- the literal function training
     itself used, so there's no separate kwarg-mapping to keep in sync.
  2. probe.onnx -- exported from probe_best.pt (confirmed: train_probe_time_dependent.py
     reloads probe_best.pt's weights right before exporting, no reload of probe_last.pt in
     between) for essentially every completed run, sweep or not. Unlike the raw state_dict,
     torch.onnx.export traces the actual executed graph -- so hyperparameters invisible to
     the state_dict alone (pool_type "attention" vs "topk", since they share identical
     params; the topk cutoff; pool_temperature) show up as literal graph nodes/constants,
     scoped by submodule path (e.g. "/pool_base/Softmax", "/pool_lang/TopK"). Combined with
     state_dict tensor shapes (hidden_dim, n_hidden_layers, key_dim, n_queries,
     input_proj_dim, embed_dim/embed_hidden, share_image_pool via tensor equality), this
     closes every remaining gap.
  3. Neither present -- fall back to PerceptionSuccessProbe() defaults (today's behavior),
     with a clear logged warning naming the run, instead of silently assuming.

In every tier, the final model.load_state_dict(state_dict, strict=True) stays as the safety
net: a reconstruction that's wrong in any way that changes a parameter's shape raises
immediately instead of silently loading mismatched weights.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

import onnx
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_probe_time_dependent as train_mod  # noqa: E402
from probe_model import CANONICAL_MODALITIES, PerceptionSuccessProbe  # noqa: E402

# Recovered via state_dict shapes alone, no ONNX needed.
_STATE_DICT_ONLY_ARCH = "state_dict shapes only (no probe.onnx -- pool_type/topk/temperature unverified)"


def find_command_txt(run_dir: Path) -> Path | None:
    """command.txt sits next to the *timestamped* run dir, i.e. run_dir.parent/command.txt --
    handles run_dir being either that timestamped dir directly or a `latest` symlink to it."""
    resolved = run_dir.resolve()
    candidate = resolved.parent / "command.txt"
    return candidate if candidate.is_file() else None


def load_args_from_command_txt(path: Path):
    """Re-parses the exact saved invocation. The file is one line:
    <python> <train_probe_time_dependent.py path> --flag value ... -- drop the first two
    tokens (interpreter + script path) and hand the rest to parse_args(argv)."""
    tokens = shlex.split(path.read_text())
    script_idx = next(i for i, t in enumerate(tokens) if t.endswith("train_probe_time_dependent.py"))
    argv = tokens[script_idx + 1:]
    return train_mod.parse_args(argv)


def _resolve_constant(model: onnx.ModelProto, name: str) -> float | int | None:
    """Best-effort: look up `name` first among graph initializers, then among Constant node
    outputs, and return its scalar value (int/float) if it has exactly one element."""
    for init in model.graph.initializer:
        if init.name == name:
            arr = onnx.numpy_helper.to_array(init)
            return arr.reshape(-1)[0].item() if arr.size == 1 else None
    for node in model.graph.node:
        if node.op_type == "Constant" and name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    arr = onnx.numpy_helper.to_array(attr.t)
                    return arr.reshape(-1)[0].item() if arr.size == 1 else None
    return None


def _infer_pool_branch(model: onnx.ModelProto, prefix: str) -> dict | None:
    """Inspects every traced node scoped under f"/{prefix}/" (e.g. "/pool_base/") and
    returns {"pool_type", "topk", "temperature"} for that modality's pool, or None if the
    modality has no nodes at all (i.e. wasn't in --modalities)."""
    nodes = [n for n in model.graph.node if n.name.startswith(f"/{prefix}/")]
    if not nodes:
        return None
    op_types = {n.op_type for n in nodes}

    if "TopK" in op_types:
        pool_type = "topk"
        topk_node = next(n for n in nodes if n.op_type == "TopK")
        k = _resolve_constant(model, topk_node.input[1])
        topk = int(k) if k is not None else None
    elif "Tanh" in op_types and "Sigmoid" in op_types:
        pool_type, topk = "gated", None
    elif "Softmax" in op_types:
        pool_type, topk = "attention", None
    elif "ReduceMean" in op_types:
        pool_type, topk = "mean", None
    elif "ReduceMax" in op_types:
        pool_type, topk = "max", None
    else:
        logging.warning("Unrecognized op set %s under %s -- assuming 'attention'.", op_types, prefix)
        pool_type, topk = "attention", None

    temperature = 1.0
    if pool_type in ("attention", "gated", "topk"):
        div_node = next((n for n in nodes if n.op_type == "Div"), None)
        if div_node is not None:
            val = _resolve_constant(model, div_node.input[1])
            if val is not None:
                temperature = float(val)
    return {"pool_type": pool_type, "topk": topk, "temperature": temperature}


def infer_arch_from_onnx(onnx_path: Path, state_dict: dict[str, torch.Tensor]) -> dict:
    """Combines the traced ONNX graph (for pool_type/topk/temperature, which the raw
    state_dict can't distinguish) with state_dict tensor shapes (for everything else) into
    PerceptionSuccessProbe constructor kwargs."""
    model = onnx.load(str(onnx_path))

    modality_attr = {"base": "pool_base", "wrist": "pool_wrist", "lang": "pool_lang"}
    branches = {m: _infer_pool_branch(model, modality_attr[m]) for m in CANONICAL_MODALITIES}
    present_modalities = tuple(m for m in CANONICAL_MODALITIES if branches[m] is not None)
    if not present_modalities:
        raise ValueError(f"No pool_{{base,wrist,lang}} nodes found at all in {onnx_path}")

    # pool_type/topk/temperature must agree across every present modality -- the model only
    # has one pool_type/topk/pool_temperature for all of them (per PerceptionSuccessProbe's
    # constructor signature).
    pool_types = {branches[m]["pool_type"] for m in present_modalities}
    if len(pool_types) > 1:
        raise ValueError(f"Inconsistent pool_type across modalities in {onnx_path}: {branches}")
    pool_type = pool_types.pop()
    topk = next((branches[m]["topk"] for m in present_modalities if branches[m]["topk"] is not None), None)
    temperature = branches[present_modalities[0]]["temperature"]

    prefix = modality_attr[present_modalities[0]]
    key_dim = None
    n_queries = 1
    if f"{prefix}.query" in state_dict:
        q = state_dict[f"{prefix}.query"]
        key_dim = q.shape[-1]
        n_queries = q.shape[0] if q.ndim == 2 else 1
    elif f"{prefix}.gate_v.weight" in state_dict:
        key_dim = state_dict[f"{prefix}.gate_v.weight"].shape[0]
        n_queries = state_dict[f"{prefix}.gate_w.weight"].shape[0]

    share_image_pool = False
    if "base" in present_modalities and "wrist" in present_modalities:
        base_keys = {k[len("pool_base."):] for k in state_dict if k.startswith("pool_base.")}
        wrist_keys = {k[len("pool_wrist."):] for k in state_dict if k.startswith("pool_wrist.")}
        share_image_pool = base_keys == wrist_keys and all(
            torch.equal(state_dict[f"pool_base.{k}"], state_dict[f"pool_wrist.{k}"]) for k in base_keys
        )

    input_proj_dim = state_dict["input_proj"].shape[1] if "input_proj" in state_dict else None

    has_embed = any(k.startswith("embed.") for k in state_dict)
    if has_embed:
        embed_linears = _sorted_linear_out_features(state_dict, "embed")
        embed_dim = embed_linears[-1]
        embed_hidden = embed_linears[0] if len(embed_linears) > 1 else None
        head_linears = _sorted_linear_out_features(state_dict, "classifier")
    else:
        embed_dim = embed_hidden = None
        head_linears = _sorted_linear_out_features(state_dict, "head")

    hidden_dim = head_linears[0]
    n_hidden_layers = len(head_linears) - 1

    return {
        "hidden_dim": hidden_dim,
        "n_hidden_layers": n_hidden_layers,
        "key_dim": key_dim,
        "pool_type": pool_type,
        "n_queries": n_queries,
        "topk": topk,
        "pool_temperature": temperature,
        "modalities": present_modalities,
        "share_image_pool": share_image_pool,
        "input_proj_dim": input_proj_dim,
        "embed_dim": embed_dim,
        "embed_hidden": embed_hidden,
    }


def _sorted_linear_out_features(state_dict: dict[str, torch.Tensor], module_prefix: str) -> list[int]:
    """All f"{module_prefix}.{i}.weight" 2-D (Linear) entries, in increasing i order ->
    their out_features. Skips LayerNorm weights (1-D)."""
    import re

    entries = []
    pattern = re.compile(rf"^{re.escape(module_prefix)}\.(\d+)\.weight$")
    for key, tensor in state_dict.items():
        m = pattern.match(key)
        if m and tensor.ndim == 2:
            entries.append((int(m.group(1)), tensor.shape[0]))
    entries.sort()
    if not entries:
        raise ValueError(f"No Linear weights found under '{module_prefix}.*' in state_dict")
    return [out_features for _, out_features in entries]


def load_probe_model(run_dir: Path, checkpoint: str, device: str) -> tuple[nn.Module, str]:
    """Returns (model, architecture_source) where architecture_source is one of
    "command.txt", "onnx", or "default (unverified)" -- recorded in the sweep summary JSON
    so it's visible which tier recovered each model's architecture."""
    checkpoint_path = run_dir / checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    command_txt = find_command_txt(run_dir)
    if command_txt is not None:
        args = load_args_from_command_txt(command_txt)
        model = train_mod.build_model(args)
        source = "command.txt"
    else:
        onnx_path = run_dir / "probe.onnx"
        if onnx_path.is_file():
            kwargs = infer_arch_from_onnx(onnx_path, state_dict)
            model = PerceptionSuccessProbe(**kwargs)
            source = "onnx"
        else:
            logging.warning(
                "No command.txt or probe.onnx found for %s -- assuming default architecture "
                "(unverified). If this run used non-default sweep args, loading will likely "
                "fail below.", run_dir,
            )
            model = PerceptionSuccessProbe()
            source = "default (unverified)"

    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, source
