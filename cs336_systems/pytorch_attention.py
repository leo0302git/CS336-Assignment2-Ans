import math
import time
import csv
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from cs336_basics.model import scaled_dot_product_attention
import torch

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric: str, title: str, y_label: str, out_png: str, logx: bool = True, logy: bool = True):
    plt.figure()
    for d in sorted(df["d_model"].unique()):
        sub = df[df["d_model"] == d].sort_values("seq_len")
        plt.plot(sub["seq_len"], sub[metric], marker="o", label=f"d_model={d}")

    plt.xlabel("sequence length L")
    plt.ylabel(y_label)
    plt.title(title)
    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_csv():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to attention_bench.csv",default='../data/pytorch_attention.csv')
    ap.add_argument("--out_dir", default="../data", help="Directory to save plots")
    ap.add_argument("--no_logy", action="store_false", help="Disable log scale on y-axis")
    ap.add_argument("--no_logx", action="store_true", help="Disable log scale on x-axis")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Keep only successful runs for smooth curves
    ok = df[df["status"] == "ok"].copy()

    # Ensure numeric
    for col in ["forward_ms", "backward_ms", "mem_before_backward_MiB", "seq_len", "d_model"]:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    ok = ok.dropna(subset=["seq_len", "d_model", "forward_ms", "backward_ms", "mem_before_backward_MiB"])

    logx = not args.no_logx
    logy = not args.no_logy

    out1 = f"{args.out_dir.rstrip('/')}/forward_ms_vs_L.png"
    out2 = f"{args.out_dir.rstrip('/')}/backward_ms_vs_L.png"
    out3 = f"{args.out_dir.rstrip('/')}/mem_before_backward_vs_L.png"

    plot_metric(
        ok,
        metric="forward_ms",
        title="Attention forward time vs sequence length",
        y_label="forward time (ms)",
        out_png=out1,
        logx=logx,
        logy=logy,
    )

    plot_metric(
        ok,
        metric="backward_ms",
        title="Attention backward time vs sequence length",
        y_label="backward time (ms)",
        out_png=out2,
        logx=logx,
        logy=logy,
    )

    plot_metric(
        ok,
        metric="mem_before_backward_MiB",
        title="Memory before backward vs sequence length",
        y_label="memory before backward (MiB)",
        out_png=out3,
        logx=logx,
        logy=logy,
    )

    print("Saved plots:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)




def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    return time.perf_counter()



@dataclass
class Result:
    d_model: int
    seq_len: int
    forward_ms: Optional[float]
    backward_ms: Optional[float]
    mem_before_backward_mib: Optional[float]
    status: str  # "ok", "oom_forward", "oom_backward", "error"
    note: str = ""


def bench_one(
    d_model: int, L: int, iters: int, warmup: int,
    dtype: torch.dtype, device: str,
    attn_fn: Callable,  # 关键：传进来已决定是否compile的函数
) -> Result:
    B = 8

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        Q = torch.randn((B, L, d_model), device=device, dtype=dtype)
        K = torch.randn((B, L, d_model), device=device, dtype=dtype)
        V = torch.randn((B, L, d_model), device=device, dtype=dtype)

        # warmup forward (no grad)
        with torch.no_grad():
            for _ in range(warmup):
                _ = attn_fn(Q, K, V)
                sync()

        # time forward (no grad)
        sync()
        t0 = now()
        with torch.no_grad():
            for _ in range(iters):
                _ = attn_fn(Q, K, V)
                sync()
        t1 = now()
        forward_ms = (t1 - t0) * 1000.0 / iters

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            return Result(d_model, L, None, None, None, "oom_forward", note=msg[:200])
        return Result(d_model, L, None, None, None, "error", note=msg[:200])

    mem_before_list: List[int] = []
    backward_times: List[float] = []

    try:
        # warmup backward
        for _ in range(warmup):
            Qw = torch.randn((B, L, d_model), device=device, dtype=dtype, requires_grad=True)
            Kw = torch.randn((B, L, d_model), device=device, dtype=dtype, requires_grad=True)
            Vw = torch.randn((B, L, d_model), device=device, dtype=dtype, requires_grad=True)
            out = attn_fn(Qw, Kw, Vw)
            loss = out.sum()
            sync()
            loss.backward()
            sync()

        # timed backward
        for _ in range(iters):
            Qb = torch.randn((B, L, d_model), device=device, dtype=dtype, requires_grad=True)
            Kb = torch.randn((B, L, d_model), device=device, dtype=dtype, requires_grad=True)
            Vb = torch.randn((B, L, d_model), device=device, dtype=dtype, requires_grad=True)

            out = attn_fn(Qb, Kb, Vb)
            loss = out.sum()

            sync()
            mem_before_list.append(torch.cuda.memory_allocated())

            t0 = now()
            loss.backward()
            sync()
            t1 = now()
            backward_times.append(t1 - t0)

        mem_before_backward_mib = (sum(mem_before_list) / len(mem_before_list)) / (1024**2)
        backward_ms = (sum(backward_times) / len(backward_times)) * 1000.0

        return Result(d_model, L, forward_ms, backward_ms, mem_before_backward_mib, "ok")

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            return Result(d_model, L, forward_ms, None, None, "oom_backward", note=msg[:200])
        return Result(d_model, L, forward_ms, None, None, "error", note=msg[:200])
def build_attn_fn(compiled: bool, backend: str):
    if not compiled:
        return scaled_dot_product_attention
    return torch.compile(scaled_dot_product_attention, backend=backend, fullgraph=False)



def main():
    from pathlib import Path
    out_path = Path('../data')
    out_path.mkdir(parents=True, exist_ok=True) 
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="../data/pytorch_attention.csv")
    ap.add_argument('--compile', default=False)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required for this benchmark."
    device = args.device

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    d_models = [16, 32, 64, 128]
    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]
    
    attn_fn = build_attn_fn(compiled=args.compile, backend="aot_eager")  # 或 "eager"
    
    results: List[Result] = []
    for d in d_models:
        for L in seq_lens:
            print(f"[Complile: {args.compile}] Running d_model={d}, L={L} ...")
            r = bench_one(d, L, args.iters, args.warmup, dtype, device,attn_fn)
            print(f"  status={r.status}, fwd_ms={r.forward_ms}, bwd_ms={r.backward_ms}, mem_pre_bwd_MiB={r.mem_before_backward_mib}")
            results.append(r)

    # Save CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_model", "seq_len", "dtype", "forward_ms", "backward_ms", "mem_before_backward_MiB", "status", "note"])
        for r in results:
            w.writerow([r.d_model, r.seq_len, args.dtype, r.forward_ms, r.backward_ms, r.mem_before_backward_mib, r.status, r.note])

    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
    plot_csv()
