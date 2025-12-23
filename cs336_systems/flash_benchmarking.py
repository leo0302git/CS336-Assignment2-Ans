import math
import csv
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

import torch
import triton
import triton.testing

from cs336_systems.flash_forward import MyTritonFlashAttentionAutogradFunctionClass as FLASH_AUTOGRAD_FN
from cs336_systems.flash_forward import MyFlashAttnAutogradFunctionClass as PYTORCH_AUTOGRAD_FN
from cs336_systems.pytorch_attention import scaled_dot_product_attention



@dataclass
class BenchResult:
    impl: str                # "flash" or "pytorch"
    mode: str                # "fwd" / "bwd" / "e2e"
    dtype: str               # "bf16" / "fp32"
    seqlen: int
    d: int
    ms: Optional[float]      # latency in ms, None if OOM/error
    peak_mem_mb: Optional[float]
    status: str              # "ok" / "oom" / "error"
    note: str = ""


def _dtype_name(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float32:
        return "fp32"
    return str(dt)


def make_causal_mask(n: int, device: torch.device) -> torch.Tensor:
    # mask shape: (queries, keys) boolean
    # True means "allowed" (lower triangle)
    return torch.ones((n, n), device=device, dtype=torch.bool).tril()


def flash_forward(Q, K, V, is_causal: bool) -> torch.Tensor:
    # 你的 autograd function: apply(Q,K,V,is_causal)
    return FLASH_AUTOGRAD_FN.apply(Q, K, V, is_causal)


# def pytorch_forward(Q, K, V, is_causal: bool) -> torch.Tensor:
#     # 你的 autograd function: apply(Q,K,V,is_causal)
#     return PYTORCH_AUTOGRAD_FN.apply(Q, K, V, is_causal)
def pytorch_forward(Q, K, V, mask) -> torch.Tensor:
    # 你的 autograd function: apply(Q,K,V,is_causal)
    return scaled_dot_product_attention(Q, K, V, mask)

def bench_one_case(
    impl: str,
    mode: str,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool,
    mask: Optional[torch.Tensor],
    warmup: int,
    rep: int,
) -> Tuple[Optional[float], Optional[float], str, str]:
    """
    Returns: (ms, peak_mem_mb, status, note)
    """
    device = Q.device
    assert device.type == "cuda"

    # 为了更稳定：先清 peak mem
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    try:
        if mode == "fwd":
            # forward latency：不建图、不算梯度
            with torch.inference_mode():
                if impl == "flash":
                    fn = lambda: flash_forward(Q, K, V, is_causal)
                else:
                    fn = lambda: pytorch_forward(Q, K, V, mask)

                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                return float(ms), float(peak), "ok", ""

        elif mode == "bwd":
            # backward-only latency：
            # 1) 先 forward 一次，保留 graph
            # 2) benchmark 只跑 backward，并 retain_graph=True 复用 graph
            Q.requires_grad_(True)
            K.requires_grad_(True)
            V.requires_grad_(True)

            # 先 forward 一次，保留 graph
            if impl == "flash":
                out = flash_forward(Q, K, V, is_causal)
            else:
                out = pytorch_forward(Q, K, V, mask)

            # 固定一个 grad_output，避免 loss.sum() 引入额外 kernel
            dO = torch.randn_like(out)

            def bwd_only():
                # 不触碰 .grad，避免清零开销：用 autograd.grad 直接取梯度
                torch.autograd.grad(
                    out, (Q, K, V),
                    grad_outputs=dO,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )

            ms = triton.testing.do_bench(bwd_only, warmup=warmup, rep=rep)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            return float(ms), float(peak), "ok", ""

        elif mode == "e2e":
            # end-to-end：每次 forward+backward（重新建图）
            def e2e_once():
                # 这里不要 inference_mode，需要建图
                Q_ = Q.detach().requires_grad_(True)
                K_ = K.detach().requires_grad_(True)
                V_ = V.detach().requires_grad_(True)

                if impl == "flash":
                    out_ = flash_forward(Q_, K_, V_, is_causal)
                else:
                    out_ = pytorch_forward(Q_, K_, V_, mask)

                dO_ = torch.randn_like(out_)
                torch.autograd.grad(
                    out_, (Q_, K_, V_),
                    grad_outputs=dO_,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            ms = triton.testing.do_bench(e2e_once, warmup=warmup, rep=rep)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            return float(ms), float(peak), "ok", ""

        else:
            return None, None, "error", f"unknown mode {mode}"

    except torch.cuda.OutOfMemoryError:
        # 清理显存，避免后续 case 受影响
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        return None, None, "oom", "cuda oom"
    except Exception as e:
        return None, None, "error", f"{type(e).__name__}: {e}"


def main():
    import pathlib
    data_path = pathlib.Path('../data/fash_benchmarking/')
    parser = argparse.ArgumentParser()
    data_path.mkdir(parents=True,exist_ok=True)
    parser.add_argument("--out_csv", type=str, default="../data/fash_benchmarking/flash_bench_results.csv")
    parser.add_argument("--out_png", type=str, default="../data/fash_benchmarking/")
    
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if FLASH_AUTOGRAD_FN is None:
        raise RuntimeError("Please set FLASH_AUTOGRAD_FN to your FlashAttention autograd Function class.")
    if PYTORCH_AUTOGRAD_FN is None:
        raise RuntimeError("Please set PYTORCH_AUTOGRAD_FN to your pytorch function class.")
    if scaled_dot_product_attention is None:
        raise RuntimeError("Please import and set scaled_dot_product_attention baseline function.")

    device = torch.device(args.device)
    assert device.type == "cuda", "Benchmark must run on GPU."

    B = 1
    is_causal = True

    seqlens = [2 ** p for p in range(7, 17)]  # 128..65536
    ds = [2 ** p for p in range(4, 8)]        # 16..128
    dtypes = [torch.bfloat16, torch.float32]
    impls = ["flash", "pytorch"]
    modes = ["fwd", "bwd", "e2e"]

    results: list[BenchResult] = []

    for dt in dtypes:
        for n in seqlens:
            # baseline causal mask (N,N) 会非常大；但 baseline 本身也是 O(N^2)。
            # 对大 N 基本一定 OOM，所以这里仍然构造，OOM 会被捕获并继续。
            mask = make_causal_mask(n, device=device)

            for d in ds:
                # 随机输入：在 benchmark 前生成
                # 注意：baseline 和 flash 都用同一份输入（便于对比）
                Q = torch.randn((B, n, d), device=device, dtype=dt)
                K = torch.randn((B, n, d), device=device, dtype=dt)
                V = torch.randn((B, n, d), device=device, dtype=dt)

                for impl in impls:
                    for mode in modes:
                        # baseline 才需要 mask；flash 的 causal 在 kernel 内实现
                        this_mask = mask if impl == "pytorch" else None

                        ms, peak_mb, status, note = bench_one_case(
                            impl=impl,
                            mode=mode,
                            Q=Q,
                            K=K,
                            V=V,
                            is_causal=is_causal,
                            mask=this_mask,
                            warmup=args.warmup,
                            rep=args.rep,
                        )

                        results.append(BenchResult(
                            impl=impl,
                            mode=mode,
                            dtype=_dtype_name(dt),
                            seqlen=n,
                            d=d,
                            ms=ms,
                            peak_mem_mb=peak_mb,
                            status=status,
                            note=note,
                        ))

                        print(f"[{status}] impl={impl:7s} mode={mode:3s} "
                              f"dtype={_dtype_name(dt):4s} N={n:6d} d={d:3d} "
                              f"ms={ms if ms is not None else 'NA'} "
                              f"peakMB={peak_mb if peak_mb is not None else 'NA'} "
                              f"{note}")

            # mask 很大，释放一下
            del mask
            torch.cuda.empty_cache()

    # 写 CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "mode", "dtype", "seqlen", "d", "ms", "peak_mem_mb", "status", "note"])
        for r in results:
            writer.writerow([r.impl, r.mode, r.dtype, r.seqlen, r.d,
                             r.ms if r.ms is not None else "",
                             r.peak_mem_mb if r.peak_mem_mb is not None else "",
                             r.status, r.note])

    print(f"\nSaved results to: {args.out_csv}")
    for datatype in dtypes:
        plot_from_csv(args.out_csv, args.out_png, _dtype_name(datatype))
    





def plot_from_csv(csv_path: str, out_dir: str = ".", dtype: str = "bf16"):
    df = pd.read_csv(csv_path)

    # 只画成功的点
    df = df[df["status"] == "ok"].copy()
    df["ms"] = pd.to_numeric(df["ms"], errors="coerce")
    df = df.dropna(subset=["ms"])

    # 你可以按需改：挑一个 d 做 ms vs seqlen
    ds = sorted(df["d"].unique())
    if not ds:
        raise RuntimeError("No valid rows to plot.")
    d_pick = min(ds)  # 例如默认挑最小 d
    print(f"Plotting dtype={dtype}, d={d_pick}")

    df1 = df[(df["dtype"] == dtype) & (df["d"] == d_pick)]

    # 画每个 mode 一张：ms vs seqlen，对比 impl
    for mode in ["fwd", "bwd", "e2e"]:
        sub = df1[df1["mode"] == mode]
        if sub.empty:
            continue

        plt.figure()
        for impl in ["flash", "pytorch"]:
            curve = sub[sub["impl"] == impl].sort_values("seqlen")
            if curve.empty:
                continue
            plt.plot(curve["seqlen"], curve["ms"], marker="o", label=impl)

        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Sequence length (N)")
        plt.ylabel("Latency (ms)")
        plt.title(f"{mode} latency vs N (dtype={dtype}, d={d_pick})")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        out_path = f"{out_dir}/latency_vs_N_{mode}_{dtype}_d{d_pick}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)

    # 再挑一个 N 做 ms vs d
    ns = sorted(df[df["dtype"] == dtype]["seqlen"].unique())
    if ns:
        n_pick = min(ns)  # 默认挑最小 N
        df2 = df[(df["dtype"] == dtype) & (df["seqlen"] == n_pick)]

        for mode in ["fwd", "bwd", "e2e"]:
            sub = df2[df2["mode"] == mode]
            if sub.empty:
                continue

            plt.figure()
            for impl in ["flash", "pytorch"]:
                curve = sub[sub["impl"] == impl].sort_values("d")
                if curve.empty:
                    continue
                plt.plot(curve["d"], curve["ms"], marker="o", label=impl)

            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.xlabel("Head dim (d)")
            plt.ylabel("Latency (ms)")
            plt.title(f"{mode} latency vs d (dtype={dtype}, N={n_pick})")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            out_path = f"{out_dir}/latency_vs_d_{mode}_{dtype}_N{n_pick}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print("Saved:", out_path)


def plot_from_csv2(csv_path: str, out_dir: str = ".", dtype: str = "bf16"):
    df = pd.read_csv(csv_path)

    # 只画成功的点
    df = df[df["status"] == "ok"].copy()
    df["ms"] = pd.to_numeric(df["ms"], errors="coerce")
    df = df.dropna(subset=["ms"])

    # 你可以按需改：挑一个 d 做 ms vs seqlen
    ds = sorted(df["d"].unique())
    if not ds:
        raise RuntimeError("No valid rows to plot.")
    d_pick = min(ds)  # 例如默认挑最小 d
    print(f"Plotting dtype={dtype}, d={d_pick}")

    df1 = df[(df["dtype"] == dtype) & (df["d"] == d_pick)]

    # 画每个 mode 一张：ms vs seqlen，对比 impl
    for mode in ["fwd", "bwd", "e2e"]:
        sub = df1[df1["mode"] == mode]
        if sub.empty:
            continue

        plt.figure()
        for impl in ["flash", "pytorch"]:
            curve = sub[sub["impl"] == impl].sort_values("seqlen")
            if curve.empty:
                continue
            plt.plot(curve["seqlen"], curve["ms"], marker="o", label=impl)

        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Sequence length (N)")
        plt.ylabel("Latency (ms)")
        plt.title(f"{mode} latency vs N (dtype={dtype}, d={d_pick})")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        out_path = f"{out_dir}/latency_vs_N_{mode}_{dtype}_d{d_pick}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)

    # 再挑一个 N 做 ms vs d
    ns = sorted(df[df["dtype"] == dtype]["seqlen"].unique())
    if ns:
        n_pick = min(ns)  # 默认挑最小 N
        df2 = df[(df["dtype"] == dtype) & (df["seqlen"] == n_pick)]

        for mode in ["fwd", "bwd", "e2e"]:
            sub = df2[df2["mode"] == mode]
            if sub.empty:
                continue

            plt.figure()
            for impl in ["flash", "pytorch"]:
                curve = sub[sub["impl"] == impl].sort_values("d")
                if curve.empty:
                    continue
                plt.plot(curve["d"], curve["ms"], marker="o", label=impl)

            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.xlabel("Head dim (d)")
            plt.ylabel("Latency (ms)")
            plt.title(f"{mode} latency vs d (dtype={dtype}, N={n_pick})")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            out_path = f"{out_dir}/latency_vs_d_{mode}_{dtype}_N{n_pick}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print("Saved:", out_path)


    

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # 如果是 H100 上通常建议打开（fp32 matmul 会更快）
    main()
