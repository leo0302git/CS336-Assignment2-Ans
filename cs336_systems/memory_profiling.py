from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import cs336_basics.model as Model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import time
from einops import rearrange
import numpy as np
from pathlib import Path
# -------------------------
# Config
# -------------------------
config = {
    "vocab_size": 10000,
    "context_length": 128,
    "batch_size": 2,
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
    "rope_theta": 10000,
}

MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    # "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    # "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

def baseline_active_bytes(snapshot: dict) -> int:
    """
    Sum sizes of blocks that are already active at snapshot time.
    This captures the big baseline (e.g., model parameters) that happened
    before _record_memory_history() started.
    """
    segs = snapshot.get("segments", [])
    base = 0
    for seg in segs:
        for blk in seg.get("blocks", []):
            state = blk.get("state", "")
            # Common value in snapshots: "active_allocated"
            if isinstance(state, str) and "active" in state and "allocated" in state:
                # size key can vary: "size", "requested_size", etc.
                sz = blk.get("size", None)
                if sz is None:
                    sz = blk.get("requested_size", None)
                if isinstance(sz, int):
                    base += sz
    return base


def peak_active_bytes_from_snapshot(snapshot: dict) -> int:
    """
    peak = max_t (baseline + delta_active_from_events(t))
    """
    base = baseline_active_bytes(snapshot)

    events = _find_event_list(snapshot)  # use your existing event-finder
    active = base
    peak = base

    for evt in events:
        norm = _event_kind_and_size(evt)  # your existing normalizer
        if norm is None:
            continue
        kind, sz = norm
        if kind == "alloc":
            active += sz
            peak = max(peak, active)
        else:
            active = max(base, active - sz)  # don't go below baseline

    return peak

# -------------------------
# Single train step timing
# -------------------------
def benchmark_step(
    model,
    model_name,
    opt,
    x,
    y,
    use_autocast: bool,
    autocast_dtype=torch.bfloat16,
    forward_only = True,
    use_mem_prof = False
):
    L = x.shape[-1]
    mix = 'mix' if use_autocast else 'nomix'
    f_only = 'forward' if forward_only else 'train'
    ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if use_autocast
        else nullcontext()
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    
    if use_mem_prof: torch.cuda.memory._record_memory_history(max_entries=200000)
    
    with ctx:
        pred = model(x)
        loss = cross_entropy(
            rearrange(pred, "b l v -> (b l) v"),
            rearrange(y, "b l -> (b l)")
        )
    

    if forward_only and use_mem_prof: 
        torch.cuda.memory._dump_snapshot(f"../data/mem_prof_{model_name}_{f_only}_L{L}_{mix}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if not forward_only and use_mem_prof:
        torch.cuda.memory._dump_snapshot(f"../data/mem_prof_{model_name}_{f_only}_L{L}_{mix}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        
    torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


# -------------------------
# Benchmark loop
# -------------------------
def run_benchmark(
    model,
    model_name,
    opt,
    x,
    y,
    warmup_steps: int,
    measure_steps: int,
    use_autocast: bool,
    forward_only: bool,
    use_mem_prof: bool
):
    # warmup
    for _ in range(warmup_steps):
        benchmark_step(model, model_name, opt, x, y, use_autocast, forward_only= forward_only, use_mem_prof=use_mem_prof)

    times = []
    for _ in range(measure_steps):
        t = benchmark_step(model, model_name,opt, x, y, use_autocast, forward_only= forward_only, use_mem_prof=use_mem_prof)
        times.append(t)

    return sum(times) / len(times)

import argparse
import glob
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


FILENAME_RE = re.compile(
    r"""mem_prof_(?:(?P<model>[^_]+)_)?(?P<phase>forward|train)_L(?P<L>\d+)_(?P<mix>mix|nomix)\.pickle$"""
)

# Example accepted by regex:
#   mem_prof_forward_L128_nomix.pickle           (model missing -> unknown)
#   mem_prof_large_forward_L256_mix.pickle
#   mem_prof_2.7b_forward_L128_nomix.pickle
#   mem_prof_large_train_L512_nomix.pickle


@dataclass
class Meta:
    path: str
    model_size: str
    context_length: int
    mixed_precision: bool
    forward_only: bool


def parse_meta(path: str) -> Optional[Meta]:
    base = os.path.basename(path)
    m = FILENAME_RE.match(base)
    if not m:
        return None
    model = m.group("model") or "unknown"
    phase = m.group("phase")
    L = int(m.group("L"))
    mix = m.group("mix") == "mix"
    forward_only = phase == "forward"
    return Meta(
        path=path,
        model_size=model,
        context_length=L,
        mixed_precision=mix,
        forward_only=forward_only,
    )


def load_snapshot(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: snapshot is not a dict (got {type(obj)})")
    return obj


def _find_event_list(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Try to locate the allocation/free event list in a torch memory snapshot.
    Different torch versions store traces differently. We search common locations.
    """
    # Common candidates (depending on torch version)
    candidates: List[Any] = []

    # Newer snapshots often have 'device_traces' (list per device)
    if "device_traces" in snapshot and isinstance(snapshot["device_traces"], list):
        candidates.extend(snapshot["device_traces"])

    # Some versions use 'traces'
    if "traces" in snapshot and isinstance(snapshot["traces"], list):
        candidates.extend(snapshot["traces"])

    # Some embed directly
    if "events" in snapshot and isinstance(snapshot["events"], list):
        candidates.append(snapshot["events"])

    # Try to extract events list from candidate structures
    for cand in candidates:
        # cand could be a dict with 'events'/'trace'/'records'
        if isinstance(cand, dict):
            for k in ("events", "trace", "records", "allocations"):
                v = cand.get(k)
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
        # cand itself might already be a list[dict]
        if isinstance(cand, list) and cand and isinstance(cand[0], dict):
            return cand

    # If we get here, we couldn't find a list of dict events
    keys = sorted(snapshot.keys())
    raise KeyError(
        "Could not locate an event list in snapshot.\n"
        f"Top-level keys: {keys}\n"
        "Hint: open one pickle and inspect its structure; "
        "you may need to adjust _find_event_list() for your torch version."
    )


def _event_kind_and_size(evt: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """
    Normalize an event to (kind, size_bytes).
    kind: 'alloc' or 'free'
    Returns None if event isn't an alloc/free we can interpret.
    """
    # Common key names
    # kind/action/type examples: "alloc", "free", "malloc", "free_requested", ...
    kind = None
    for k in ("kind", "action", "type", "event", "name"):
        if k in evt and isinstance(evt[k], str):
            kind = evt[k].lower()
            break

    # Size key variants
    size = None
    for k in ("size", "bytes", "nbytes", "alloc_size", "requested_size"):
        if k in evt and isinstance(evt[k], int):
            size = evt[k]
            break

    if kind is None or size is None:
        return None

    # Map kind to alloc/free
    # Very defensive matching
    if "alloc" in kind or "malloc" in kind or kind in ("a",):
        return ("alloc", size)
    if "free" in kind or "release" in kind or kind in ("f",):
        return ("free", size)

    return None


def peak_active_bytes_from_events(events: List[Dict[str, Any]]) -> int:
    """
    Reconstruct an 'active bytes' curve from alloc/free events.
    This ignores caching/segments and tries to match what "Active Memory Timeline" means:
    active = sum(alloc sizes) - sum(free sizes) over time.
    """
    active = 0
    peak = 0

    for evt in events:
        norm = _event_kind_and_size(evt)
        if norm is None:
            continue
        kind, sz = norm
        if kind == "alloc":
            active += sz
            if active > peak:
                peak = active
        else:
            # clamp (some snapshots can have unmatched frees)
            active = max(0, active - sz)

    return peak


def human_bytes(n: int) -> str:
    # Binary units (GiB)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def parse_mem_profile():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default="../data",
        help="Directory containing mem_prof_*.pickle files",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="mem_prof_*.pickle",
        help="Glob pattern relative to --dir",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="../data/mem_peaks.csv",
        help="Output CSV filename",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.dir, args.glob)}")

    rows = []
    skipped = []

    for p in paths:
        meta = parse_meta(p)
        if meta is None:
            skipped.append(p)
            continue

        snap = load_snapshot(p)
        peak = peak_active_bytes_from_snapshot(snap)

        # events = _find_event_list(snap)
        # peak = peak_active_bytes_from_events(events)

        rows.append(
            {
                "file": os.path.basename(p),
                "model_size": meta.model_size,
                "context_length": meta.context_length,
                "mixed_precision": meta.mixed_precision,
                "forward_only": meta.forward_only,
                "peak_active_bytes": int(peak),
                "peak_active_human": human_bytes(int(peak)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No files were parsed successfully. "
            "Check filename format or adjust FILENAME_RE."
        )

    # Sort nicely for reading
    df = df.sort_values(
        by=["model_size", "context_length", "mixed_precision", "forward_only"],
        ascending=[True, True, True, True],
    )

    df.to_csv(args.out_csv, index=False)

    # Print a readable table
    print("\n=== Peak Active Memory (reconstructed from snapshot events) ===")
    print(df[["model_size", "context_length", "mixed_precision", "forward_only", "peak_active_human", "file"]].to_string(index=False))
    print(f"\nSaved CSV -> {args.out_csv}")

    if skipped:
        print("\n[Skipped files that didn't match expected naming pattern:]")
        for s in skipped:
            print(" -", os.path.basename(s))


def mem_profiling():
    assert torch.cuda.is_available(), "This benchmark requires a GPU."

    device = torch.device("cuda")
    torch.manual_seed(0)
    path = Path('../data')
    path.mkdir(parents=True, exist_ok=True)
    
    B = config["batch_size"]
    L_list = [128, 256, 512]
    warmup_steps = 1
    measure_steps = 1
    for L in L_list:
        x = torch.randint(
            0, config["vocab_size"],
            (B, L),
            device=device
        )
        y = torch.randint(
            0, config["vocab_size"],
            (B, L),
            device=device
        )
        

        print(f"Running BF16 mixed-precision benchmark on {device}, context_length: {L}\n")

        for name, spec in MODEL_SPECS.items():
            print(f"Model size: {name}")

            model = Model.BasicsTransformerLM(
                vocab_size=config["vocab_size"],
                context_length=L,
                d_model=spec["d_model"],
                d_ff=spec["d_ff"],
                num_heads=spec["num_heads"],
                num_layers=spec["num_layers"],
                rope_theta=config["rope_theta"],
            ).to(device)

            opt = AdamW(
                model.parameters(),
                lr=config["lr"],
                betas=config["betas"],
                eps=config["eps"],
                weight_decay=config["weight_decay"],
            )
            try: 
                # full precision forward only
                t_fp32_f_only = run_benchmark(
                    model, name, opt, x, y,
                    warmup_steps, measure_steps,
                    use_autocast= False,
                    forward_only= True, 
                    use_mem_prof= True
                )
                
                # full precision full train step
                t_fp32_train_step = run_benchmark(
                    model, name, opt, x, y,
                    warmup_steps, measure_steps,
                    use_autocast= False,
                    forward_only= False, 
                    use_mem_prof= True
                )

                # BF16 mixed precision forward only
                t_bf16_f_only = run_benchmark(
                    model, name, opt, x, y,
                    warmup_steps, measure_steps,
                    use_autocast= True,
                    forward_only= True, 
                    use_mem_prof= True
                )
                
                # BF16 mixed precision full train step
                t_bf16_train_step = run_benchmark(
                    model, name, opt, x, y,
                    warmup_steps, measure_steps,
                    use_autocast= True,
                    forward_only= False, 
                    use_mem_prof= True
                )
                
                print(f"  FP32  avg forward time: {t_fp32_f_only:.4f} s")
                print(f"  FP32  avg step time: {t_fp32_train_step:.4f} s")
                
                print(f"  BF16  avg forward time: {t_bf16_f_only:.4f} s")
                print(f"  BF16  avg step time: {t_bf16_train_step:.4f} s")
                
                print(f"  Forward only Speedup: {t_fp32_f_only / t_bf16_f_only:.2f}x")
                print(f"  Train step Speedup: {t_fp32_train_step / t_bf16_train_step:.2f}x\n")
            except Exception as e:
                print('Cannot perform memory profiling: ', e)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # mem_profiling()
    parse_mem_profile()