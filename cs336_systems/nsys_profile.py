import argparse
import os
import time
from contextlib import nullcontext

import torch
from einops import rearrange

import cs336_basics.model as Model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import scaled_dot_product_attention

# ----------------------------
# NVTX helper (works even if NVTX isn't available)
# ----------------------------
try:
    import torch.cuda.nvtx as nvtx  # type: ignore

    def nvtx_range(name: str):
        return nvtx.range(name)

except Exception:
    def nvtx_range(name: str):
        return nullcontext()


MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------
# Optional: annotate attention internals for (e)
# ----------------------------
def make_annotated_attention(original_fn):
    """
    Wrap scaled_dot_product_attention with NVTX ranges:
      - attn_scores_matmul
      - attn_softmax
      - attn_out_matmul

    If the signature differs in your codebase, you can tweak this wrapper.
    """
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        # Many CS336 implementations do: scores = Q @ K^T / sqrt(dk); scores = mask(scores); P = softmax(scores); out = P @ V
        dk = Q.shape[-1]
        with nvtx_range("attn_scores_matmul"):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

        if mask is not None:
            # mask could be additive (large negative) or boolean. We try to handle both.
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask

        with nvtx_range("attn_softmax"):
            probs = torch.softmax(scores, dim=-1)

        with nvtx_range("attn_out_matmul"):
            out = torch.matmul(probs, V)
        return out

    # If anything goes wrong, fall back to original.
    try:
        # quick sanity: call signature check is hard; we just return the wrapper and let runtime tell us.
        return annotated_scaled_dot_product_attention
    except Exception:
        return original_fn


def maybe_patch_attention(enable: bool):
    if not enable:
        return
    # Patch the function if present in the module
    if hasattr(Model, "scaled_dot_product_attention"):
        Model.scaled_dot_product_attention = make_annotated_attention(Model.scaled_dot_product_attention)
    elif hasattr(Model, "BasicsTransformerLM") and hasattr(Model.BasicsTransformerLM, "scaled_dot_product_attention"):
        # Rare alternative layout
        Model.BasicsTransformerLM.scaled_dot_product_attention = make_annotated_attention(
            Model.BasicsTransformerLM.scaled_dot_product_attention
        )


# ----------------------------
# One step: forward-only or full train step
# ----------------------------
def run_forward(model, x, y):
    with nvtx_range("forward"):
        logits = model(x)
    # Flatten for CE
    logits_flat = rearrange(logits, "b t v -> (b t) v")
    y_flat = rearrange(y, "b t -> (b t)")
    loss = cross_entropy(logits_flat, y_flat)
    return loss


def run_train_step(model, opt, x, y):
    with nvtx_range("train_step"):
        with nvtx_range("forward"):
            logits = model(x)

        logits_flat = rearrange(logits, "b t v -> (b t) v")
        y_flat = rearrange(y, "b t -> (b t)")

        with nvtx_range("loss"):
            loss = cross_entropy(logits_flat, y_flat)

        opt.zero_grad(set_to_none=True)
        with nvtx_range("backward"):
            loss.backward()

        with nvtx_range("optimizer_step"):
            opt.step()

    return loss


def main():
    p = argparse.ArgumentParser(description="CS336 A2 Systems: nsys profiling script (small reports, NVTX ranges).")
    p.add_argument("--model_size", type=str, default="small", choices=list(MODEL_SPECS.keys()))
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--rope_theta", type=int, default=10000)

    p.add_argument("--mode", type=str, default="forward", choices=["forward", "train"])
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--measure_steps", type=int, default=1)  # keep report small by default

    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--annotate_attention", action="store_true")

    # Optimizer params (only used in train mode)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.01)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script is intended for CUDA profiling. CUDA not available.")

    maybe_patch_attention(args.annotate_attention)

    spec = MODEL_SPECS[args.model_size]
    model = Model.BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        d_ff=spec["d_ff"],
        num_heads=spec["num_heads"],
        num_layers=spec["num_layers"],
        rope_theta=args.rope_theta,
    ).to(device)

    opt = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Fixed random batch (generated once) to keep profile focused on model compute.
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=g).to(device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=g).to(device)

    # Autocast context (optional)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if args.mixed_precision else nullcontext()

    # Warmup
    model.train() if args.mode == "train" else model.eval()
    with nvtx_range("warmup"):
        for _ in range(args.warmup_steps):
            with amp_ctx:
                if args.mode == "forward":
                    with torch.no_grad():
                        _ = run_forward(model, x, y)
                else:
                    _ = run_train_step(model, opt, x, y)
            _sync_if_cuda()

    # Measured steps (keep small!)
    with nvtx_range("measured"):
        for _ in range(args.measure_steps):
            with amp_ctx:
                if args.mode == "forward":
                    model.eval()
                    with torch.no_grad():
                        _ = run_forward(model, x, y)
                else:
                    model.train()
                    _ = run_train_step(model, opt, x, y)
            _sync_if_cuda()

    # A tiny print so you know it finished.
    print(
        f"Done. mode={args.mode} size={args.model_size} "
        f"ctx={args.context_length} bs={args.batch_size} "
        f"warmup={args.warmup_steps} measured={args.measure_steps} "
        f"amp={args.mixed_precision} attn_nvtx={args.annotate_attention}"
    )


if __name__ == "__main__":
    main()




# forward: abc
'''
/root/data1/Nsight/nsight-systems/bin/nsys profile \
--trace=cuda,nvtx --sample=none --cpuctxsw=none \
--force-overwrite true \
-o forward_small_ctx128 \
python nsys_profile.py --model_size small --context_length 128 --mode forward --warmup_steps 2 --measure_steps 5 --annotate_attention
'''

# Train step（用于 b/d）
'''
/root/data1/Nsight/nsight-systems/bin/nsys profile \
--trace=cuda,nvtx --sample=none --cpuctxsw=none \
--force-overwrite true \
-o train_small_ctx128 \
python nsys_profile.py --model_size small --context_length 128 --mode train --warmup_steps 2 --measure_steps 5 --annotate_attention
'''