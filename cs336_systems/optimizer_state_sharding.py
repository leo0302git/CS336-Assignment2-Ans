from typing import Any, Dict, Iterable, List, Optional, Type, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer
import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tests.common import ToyModel





def _dist_info() -> tuple[int, int]:
    """Return (rank, world_size). Works even if torch.distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _as_param_list(x: Union[Tensor, torch.nn.Parameter, Iterable]) -> List[torch.nn.Parameter]:
    """Normalize param_group['params'] into a list of Parameters."""
    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        return [x]  # type: ignore[list-item]
    return list(x)  # type: ignore[arg-type]


class ShardedStateOptimizer(Optimizer):
    """
    ZeRO-1 style optimizer-state sharding:
      - All ranks keep a full replica of model parameters.
      - Each rank owns (and holds optimizer state for) a shard of parameters.
      - After local shard update, broadcast updated params from owning rank to all ranks.

    This is correctness-first (simple broadcast-per-param). It matches the assignment interface:
      - __init__(params, optimizer_cls, **kwargs)
      - step(closure=None, **kwargs)
      - add_param_group(param_group)
    """

    def __init__(
        self,
        params,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)

        self._rank, self._world_size = _dist_info()

        # Global deterministic sharding counter across ALL param groups
        self._global_param_index: int = 0

        # param -> owning rank
        self._param_owner: Dict[torch.nn.Parameter, int] = {}

        # Local optimizer is built from local param groups collected in add_param_group
        self._local_param_groups: List[Dict[str, Any]] = []
        self._inner_optim: Optional[Optimizer] = None

        # IMPORTANT: call torch.optim.Optimizer super-class ctor.
        # In PyTorch, "defaults" is typically the hyperparam dict passed to the optimizer ctor.
        super().__init__(params, defaults=dict(kwargs))

        # Now build the real optimizer on local shard only.
        # (Outer param_groups already contain the full param groups.)
        self._rebuild_inner_optimizer()

        # Expose sharded state (so `len(optimizer.state)` reflects sharding).
        # The canonical state is kept inside the inner optimizer.
        self.state = {} if self._inner_optim is None else self._inner_optim.state  # type: ignore[assignment]

    def _rebuild_inner_optimizer(self) -> None:
        """(Re)create the inner optimizer from current local param groups."""
        # Filter out empty groups; most torch optimizers don't like empty param groups.
        non_empty = [g for g in self._local_param_groups if len(_as_param_list(g["params"])) > 0]
        if len(non_empty) == 0:
            self._inner_optim = None
            return

        # Create wrapped optimizer on local shard.
        # We pass **kwargs as defaults; per-group overrides are kept inside group dicts.
        self._inner_optim = self._optimizer_cls(non_empty, **self._optimizer_kwargs)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Called by Optimizer.__init__ and may be called later.
        We:
          1) Let the base class record the full param group (API semantics).
          2) Assign each param an owning rank (round-robin by global index).
          3) Add the local subset of this group into the inner optimizer (sharded state).
        """
        # 1) record full group in outer optimizer
        super().add_param_group(param_group)

        # 2) assign owners deterministically
        full_params: List[torch.nn.Parameter] = _as_param_list(param_group["params"])
        for p in full_params:
            if p not in self._param_owner:
                owner = self._global_param_index % self._world_size
                self._param_owner[p] = owner
                self._global_param_index += 1

        # 3) build local subset group
        local_params = [p for p in full_params if self._param_owner[p] == self._rank]
        if len(local_params) == 0:
            # This rank owns none from this group; nothing to add to inner optimizer.
            return

        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        self._local_param_groups.append(local_group)

        # If inner optimizer already exists (e.g., add_param_group during training),
        # add directly; otherwise rebuild later.
        if self._inner_optim is not None:
            self._inner_optim.add_param_group(local_group)
            self.state = self._inner_optim.state  # type: ignore[assignment]

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        """
        1) Run inner optimizer step on local shard only.
        2) Broadcast updated parameters from owning rank to all ranks.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._inner_optim is not None:
            # Many optimizers accept **kwargs (rarely used); forward them.
            inner_loss = self._inner_optim.step(closure=None, **kwargs)
            if loss is None:
                loss = inner_loss

        # Sync updated params so all ranks have identical model weights.
        if self._world_size > 1:
            # Broadcast each parameter from its owner rank.
            # (Correctness-first; not the most efficient.)
            for group in self.param_groups:
                for p in _as_param_list(group["params"]):
                    owner = self._param_owner[p]
                    dist.broadcast(p.data, src=owner)

        return loss

    # Optional: forward common attributes/methods to wrapped optimizer where sensible.
    def zero_grad(self, set_to_none: bool = True) -> None:
        # Zero grads for ALL params (outer groups). This is what users expect.
        for group in self.param_groups:
            for p in _as_param_list(group["params"]):
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return *sharded* state_dict (only local optimizer state), plus full param_groups
        (so it can be reasoned about). This is often acceptable for ZeRO-style sharding.
        """
        if self._inner_optim is None:
            return {"state": {}, "param_groups": self.param_groups}
        sd = self._inner_optim.state_dict()
        # Keep outer param_groups for completeness (full model view).
        sd["param_groups_full"] = self.param_groups
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load *local shard* state if present.
        NOTE: Full distributed checkpointing is beyond this minimal assignment wrapper.
        """
        if self._inner_optim is not None and "state" in state_dict and "param_groups" in state_dict:
            self._inner_optim.load_state_dict({"state": state_dict["state"], "param_groups": state_dict["param_groups"]})
            self.state = self._inner_optim.state  # type: ignore[assignment]

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ============ Model (simple GPT-ish Transformer for benchmarking) ============

@dataclass
class ModelCfg:
    vocab: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int


CFG: Dict[str, ModelCfg] = {
    # 这些只是“可跑的基准配置”。如果作业给了标准 small/large/xl，替换成作业里的即可。
    "small":  ModelCfg(vocab=32000, n_layers=12, d_model=768,  n_heads=12, d_ff=3072),
    "medium": ModelCfg(vocab=32000, n_layers=24, d_model=1024, n_heads=16, d_ff=4096),
    "large":  ModelCfg(vocab=32000, n_layers=32, d_model=1536, n_heads=24, d_ff=6144),
    "xl":     ModelCfg(vocab=32000, n_layers=48, d_model=2048, n_heads=32, d_ff=8192),
}


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        h = self.ln1(x)
        # attn_mask: (T, T) with True meaning "blocked" for MHA in PyTorch if using bool mask
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(4096, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, idx: torch.Tensor):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        # causal mask: block attending to future
        # For batch_first MultiheadAttention, attn_mask shape can be (T,T) bool, True means masked.
        causal = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, causal)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ============ Helpers ============

def init_dist():
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def bytes_to_mib(x: int) -> float:
    return x / (1024 ** 2)


@torch.no_grad()
def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def model_param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def grad_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += tensor_bytes(p.grad)
    return total


def optim_state_bytes(optim) -> int:
    # Sum tensor bytes inside optimizer.state
    total = 0
    for state in optim.state.values():
        if isinstance(state, dict):
            for v in state.values():
                if torch.is_tensor(v):
                    total += tensor_bytes(v)
    return total


def reduce_max_i64(x: int) -> int:
    t = torch.tensor([x], device="cuda", dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def reduce_avg_f64(x: float) -> float:
    t = torch.tensor([x], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def mem_snapshot(tag: str) -> Tuple[int, int]:
    # (allocated, reserved) in bytes
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    return alloc, reserv


def peak_since_reset() -> Tuple[int, int]:
    # (max_allocated, max_reserved) in bytes
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


# ============ Main benchmark ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=list(CFG.keys()), default="large")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shard", type=int, choices=[0, 1], default=0, help="1=optimizer state sharding, 0=normal AdamW")
    args = ap.parse_args()

    rank, world = init_dist()

    cfg = CFG[args.model]
    torch.manual_seed(0)

    model = TinyGPT(cfg).cuda().to(dtype=torch.bfloat16)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # ---- Optimizer (swap here to use your sharded optimizer) ----
    if args.shard == 1:
        optim = ShardedStateOptimizer(ddp.parameters(), torch.optim.AdamW, lr=args.lr)
    else:
        optim = torch.optim.AdamW(ddp.parameters(), lr=args.lr)

    dist.barrier()

    # ========== Memory after init ==========
    alloc0, reserv0 = mem_snapshot("after_init")
    # Optimizer state may be lazy (AdamW creates state at first step), so state bytes often ~0 here.
    p_bytes = model_param_bytes(ddp.module)
    g_bytes = grad_bytes(ddp.module)
    o_bytes = optim_state_bytes(optim)

    # report max across ranks
    alloc0_m = bytes_to_mib(reduce_max_i64(alloc0))
    reserv0_m = bytes_to_mib(reduce_max_i64(reserv0))
    p_m = bytes_to_mib(reduce_max_i64(p_bytes))
    g_m = bytes_to_mib(reduce_max_i64(g_bytes))
    o_m = bytes_to_mib(reduce_max_i64(o_bytes))

    if rank == 0:
        print(f"\n=== Setting: shard={args.shard} world={world} model={args.model} batch={args.batch} seq={args.seq} ===")
        print(f"[After init] allocated={alloc0_m:.1f} MiB, reserved={reserv0_m:.1f} MiB")
        print(f"           breakdown(bytes->MiB): params={p_m:.1f}, grads={g_m:.1f}, optim_state={o_m:.1f}")

    # ========== One iteration memory points ==========
    # Synthesize data
    b, t = args.batch, args.seq
    vocab = cfg.vocab
    x = torch.randint(0, vocab, (b, t), device="cuda")
    y = torch.randint(0, vocab, (b, t), device="cuda")

    # --- Peak before optimizer step: peak during fwd+bwd ---
    ddp.train()
    optim.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    logits = ddp(x)
    loss = F.cross_entropy(logits.view(-1, vocab).float(), y.view(-1))
    loss.backward()

    peak_fwbw_alloc, peak_fwbw_res = peak_since_reset()
    # point snapshot directly before step (current allocated/reserved)
    alloc_pre, reserv_pre = mem_snapshot("pre_step")

    # --- Peak after optimizer step: peak during step (and immediately after) ---
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    optim.step()

    peak_step_alloc, peak_step_res = peak_since_reset()
    alloc_post, reserv_post = mem_snapshot("post_step")

    # breakdown after backward (grads exist)
    p_bytes2 = model_param_bytes(ddp.module)
    g_bytes2 = grad_bytes(ddp.module)
    o_bytes2 = optim_state_bytes(optim)  # Adam state usually materializes after first step

    # reduce max for memory, avg for loss
    peak_fwbw_alloc_m = bytes_to_mib(reduce_max_i64(peak_fwbw_alloc))
    peak_fwbw_res_m = bytes_to_mib(reduce_max_i64(peak_fwbw_res))
    alloc_pre_m = bytes_to_mib(reduce_max_i64(alloc_pre))
    reserv_pre_m = bytes_to_mib(reduce_max_i64(reserv_pre))

    peak_step_alloc_m = bytes_to_mib(reduce_max_i64(peak_step_alloc))
    peak_step_res_m = bytes_to_mib(reduce_max_i64(peak_step_res))
    alloc_post_m = bytes_to_mib(reduce_max_i64(alloc_post))
    reserv_post_m = bytes_to_mib(reduce_max_i64(reserv_post))

    p_m2 = bytes_to_mib(reduce_max_i64(p_bytes2))
    g_m2 = bytes_to_mib(reduce_max_i64(g_bytes2))
    o_m2 = bytes_to_mib(reduce_max_i64(o_bytes2))
    loss_avg = reduce_avg_f64(float(loss.item()))

    if rank == 0:
        print(f"[Before step] peak_alloc(fwd+bwd)={peak_fwbw_alloc_m:.1f} MiB, peak_reserved={peak_fwbw_res_m:.1f} MiB")
        print(f"             snapshot alloc={alloc_pre_m:.1f} MiB, reserved={reserv_pre_m:.1f} MiB, loss={loss_avg:.4f}")
        print(f"             breakdown: params={p_m2:.1f}, grads={g_m2:.1f}, optim_state={o_m2:.1f}")
        print(f"[After step ] peak_alloc(step)={peak_step_alloc_m:.1f} MiB, peak_reserved={peak_step_res_m:.1f} MiB")
        print(f"             snapshot alloc={alloc_post_m:.1f} MiB, reserved={reserv_post_m:.1f} MiB")

    # ========== Timing: average per-iter ==========
    # warmup
    dist.barrier()
    for _ in range(args.warmup):
        optim.zero_grad(set_to_none=True)
        logits = ddp(x)
        loss = F.cross_entropy(logits.view(-1, vocab).float(), y.view(-1))
        loss.backward()
        optim.step()
    dist.barrier()
    torch.cuda.synchronize()

    it_times = []
    for _ in range(args.iters):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optim.zero_grad(set_to_none=True)
        logits = ddp(x)
        loss = F.cross_entropy(logits.view(-1, vocab).float(), y.view(-1))
        loss.backward()
        optim.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        it_times.append(t1 - t0)

    # report: take max over ranks per-iter (straggler decides)
    # easiest: average locally then all_reduce max of average
    avg_local = sum(it_times) / len(it_times)
    avg_max = torch.tensor([avg_local], device="cuda", dtype=torch.float64)
    dist.all_reduce(avg_max, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[Timing] avg_iter_time(max_rank) = {avg_max.item()*1000:.2f} ms")

    dist.barrier()


if __name__ == "__main__":
    main()
