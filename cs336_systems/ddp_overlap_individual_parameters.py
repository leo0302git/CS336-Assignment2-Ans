import torch
import cs336_basics.model as Model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import time
from typing import Callable
from jaxtyping import Int, Float
from torch import Tensor
from einops import rearrange
import numpy as np
import os
import logging
from copy import deepcopy
from typing import Type, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import pathlib

from cs336_systems.distributed_communication_single_node import write_csv_row
from cs336_systems.naive_ddp import (
    naive_ddp_on_after_backward,
    barrier,
    validate_ddp_net_equivalence,
    get_naive_ddp,
    _setup_process_group,
    batched_all_reduce_grad
)

class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module): 
        '''Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks.'''
        # assume dist already initialized
        super().__init__() #这里不能直接赋值，必须要经过继承
        self.module = module
        self._handles: List[Tuple[nn.Parameter, dist.Work]] = []
        def hook(param):
            if param.grad is None: return
            handle= dist.all_reduce(param.grad,op=dist.ReduceOp.SUM,async_op=True)
            if handle == None: 
                raise ValueError
            else: self._handles.append((param, handle))
            
        seen = set()
        for p in module.parameters():
            if id(p) in seen: 
                continue
            seen.add(id(p))
            if p.requires_grad == True:
                p.register_post_accumulate_grad_hook(hook)
            dist.broadcast(p.data, src=0) 
        for b in module.buffers():
            dist.broadcast(b.data, src=0)
        return 

    def forward(self, *inputs, **kwargs): 
        '''Calls the wrapped module’s forward() method with the provided positional and keyword arguments.'''
        return self.module.forward(*inputs, **kwargs) # 记得return而不是只是算出来
        
    def finish_gradient_synchronization(self): 
        '''When called, wait for asynchronous communication calls to be queued on GPU. 这会在optimizer step之前被调用，确保各个rank上都有更新后的梯度'''
        world_size = dist.get_world_size()
        for _ , handle in self._handles:
            handle.wait()
        for param, _ in self._handles:
            if param.requires_grad == True and param.grad != None:
                param.grad.div_(world_size)
        self._handles.clear()

def _test_NaiveDistributedDataParallel(rank: int, world_size: int, warmup, step_num, config, CSV_FILE):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="nccl")

    barrier(backend='nccl',device=device)

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    non_parallel_model = Model.BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        rope_theta=config["rope_theta"],
    ).to(device)
    ddp_model = get_naive_ddp(non_parallel_model)



    # 就用同一份数据一直算，反正benchmarking不需要数据有多大意义
    all_x = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    all_y = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)

    # Each rank will see only 10 examples (out of the total dataset size of 20)
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = cross_entropy

    # Optimizer for the DDP model
    ddp_optimizer = AdamW(
        params=non_parallel_model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )
    print(f'[rank {rank}] Start training...')
    try:
        for i in range(warmup):
            ddp_optimizer.zero_grad()
            offset = rank * local_bs
            ddp_data = all_x[offset : offset + local_bs, :].to(device)
            ddp_labels = all_y[offset : offset + local_bs, :].to(device)
            ddp_outputs = ddp_model(ddp_data)
            ddp_loss = loss_fn(ddp_outputs, ddp_labels)
            ddp_loss.backward()
            # 这一步将各个rank上的梯度做了all reduce，使每个rank上都得到了所有数据上的平均梯度
            naive_ddp_on_after_backward(ddp_model)
            ddp_optimizer.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        step_times = []
        comm_times = []
        for i in range(step_num):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ddp_optimizer.zero_grad()
            offset = rank * local_bs
            ddp_data = all_x[offset : offset + local_bs, :].to(device)
            ddp_labels = all_y[offset : offset + local_bs, :].to(device)
            ddp_outputs = ddp_model(ddp_data)
            ddp_loss = loss_fn(ddp_outputs, ddp_labels)
            ddp_loss.backward()

            # 这一步将各个rank上的梯度做了all reduce，使每个rank上都得到了所有数据上的平均梯度
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            naive_ddp_on_after_backward(ddp_model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            # Measure the total time per training step and the proportion of time spent on communicating gradients.
            ddp_optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            step_times.append(t3 - t0)
            comm_times.append(t2 - t1)

        # 到这里，所有rank已经获得step_times以及comm_times
        step_t = torch.tensor(step_times, device=device)
        comm_t = torch.tensor(comm_times, device=device)
        dist.all_reduce(step_t, op=dist.ReduceOp.MAX) # all_reduce只能对张量做，不能对python list做
        dist.all_reduce(comm_t, op=dist.ReduceOp.MAX)

        if rank == 0:
            avg_step_time = step_t.mean().item() # mean完之后还要取数，用item
            avg_comm_time = comm_t.mean().item()
            comm_ratio = avg_comm_time / avg_step_time
            write_csv_row({
                "setting": "1node_2gpus",
                'overlap': False,
                "model": config['model_size'],
                "global_bs": config["batch_size"],
                "local_bs": local_bs, # local batch size
                "context_length": config["context_length"],
                "steps": step_num,
                "warmup": warmup,
                "avg_step_time_s": avg_step_time,
                "avg_comm_time_s": avg_comm_time,
                "comm_ratio": comm_ratio,
            }, CSV_FILE=CSV_FILE)
        barrier(backend='nccl',device=device)
    finally:
        # 无论中途谁抛异常，都尽量 clean shutdown。否则一个进程异常会导致其他进程中断
        print(f'[rank {rank}] Done.')
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_OverlapDistributedDataParallel(rank: int, world_size: int, warmup, step_num, config, CSV_FILE):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="nccl")

    barrier(backend='nccl',device=device)

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    # print(f'[rank {rank}] Initializing model...')
    non_parallel_model = Model.BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        rope_theta=config['rope_theta']
    ).to(device)


    ddp_model = DDPIndividualParameters(non_parallel_model)

    # # Make sure all the ranks have the same model state
    # validate_ddp_net_equivalence(ddp_model)

    # 就用同一份数据一直算，反正benchmarking不需要数据有多大意义
    all_x = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    all_y = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)

    # Each rank will see only 10 examples (out of the total dataset size of 20)
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = cross_entropy

    # Optimizer for the DDP model
    ddp_optimizer = AdamW(
        params=non_parallel_model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )
    print(f'[rank {rank}] Start training...')
    try:
        for i in range(warmup):
            ddp_optimizer.zero_grad()
            offset = rank * local_bs
            ddp_data = all_x[offset : offset + local_bs, :].to(device)
            ddp_labels = all_y[offset : offset + local_bs, :].to(device)
            ddp_outputs = ddp_model(ddp_data)
            ddp_loss = loss_fn(ddp_outputs, ddp_labels)
            ddp_loss.backward()
            # 由于注册了钩子，所以可以不用手动写同步梯度
            ddp_model.finish_gradient_synchronization()
            ddp_optimizer.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        step_times = []
        comm_times = []
        for i in range(step_num):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ddp_optimizer.zero_grad()
            offset = rank * local_bs
            ddp_data = all_x[offset : offset + local_bs, :].to(device)
            ddp_labels = all_y[offset : offset + local_bs, :].to(device)
            ddp_outputs = ddp_model(ddp_data)
            ddp_loss = loss_fn(ddp_outputs, ddp_labels)
            ddp_loss.backward()

            # 这一步将各个rank上的梯度做了all reduce，使每个rank上都得到了所有数据上的平均梯度
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            ddp_model.finish_gradient_synchronization()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            # Measure the total time per training step and the proportion of time spent on communicating gradients.
            ddp_optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            step_times.append(t3 - t0)
            comm_times.append(t2 - t1)

        # 到这里，所有rank已经获得step_times以及comm_times
        step_t = torch.tensor(step_times, device=device)
        comm_t = torch.tensor(comm_times, device=device)
        dist.all_reduce(step_t, op=dist.ReduceOp.MAX) # all_reduce只能对张量做，不能对python list做
        dist.all_reduce(comm_t, op=dist.ReduceOp.MAX)

        if rank == 0:
            avg_step_time = step_t.mean().item() # mean完之后还要取数，用item
            avg_comm_time = comm_t.mean().item()
            comm_ratio = avg_comm_time / avg_step_time
            write_csv_row({
                "setting": "1node_2gpus",
                'overlap': True,
                "model": config['model_size'],
                "global_bs": config["batch_size"],
                "local_bs": local_bs, # local batch size
                "context_length": config["context_length"],
                "steps": step_num,
                "warmup": warmup,
                "avg_step_time_s": avg_step_time,
                "avg_comm_time_s": avg_comm_time,
                "comm_ratio": comm_ratio,
            }, CSV_FILE=CSV_FILE)
        barrier(backend='nccl',device=device)
    finally:
        # 无论中途谁抛异常，都尽量 clean shutdown。否则一个进程异常会导致其他进程中断
        print(f'[rank {rank}] Done.')
        if dist.is_initialized():
            dist.destroy_process_group()

def ddp_overlap_benchmarking_single_model(config):
    torch.set_float32_matmul_precision('high')
    step_num = 20
    warmup_num = 2
    world_size = 2
    mp.spawn(# type: ignore[attr-defined]
        _test_OverlapDistributedDataParallel,
        args=(world_size, warmup_num, step_num, config, CSV_FILE),
        nprocs=world_size,
        join=True,
    )
    mp.spawn(# type: ignore[attr-defined]
        _test_NaiveDistributedDataParallel,
        args=(world_size, warmup_num, step_num, config, CSV_FILE),
        nprocs=world_size,
        join=True,
    )
    
def ddp_overlap_individual_parameters_benchmarking():
    data_path = pathlib.Path('../data/ddp_overlap_individual_parameters_benchmarking/')
    data_path.mkdir(parents=True, exist_ok=True)
    CSV_FILE = "../data/ddp_overlap_individual_parameters_benchmarking/ddp_overlap_individual_parameters_benchmarking.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    # "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    # "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
    for k,v in MODEL_SPECS.items():       
        config={
        'device': device,
        'model_size': k,
        'vocab_size':10000,
        'context_length':128,
        'd_model':v['d_model'],
        'd_ff':v['d_ff'],
        'num_layers':v['num_layers'],
        'num_heads':v['num_heads'],
        'rope_theta':10000,
        'lr':1e-3,
        'betas':(0.9, 0.999),
        'eps':1e-8,
        'weight_decay':0.01,
        'batch_size': 2
        }
        print("Start benchmarking model: ", k, ' Using device: ', device)
        ddp_overlap_benchmarking_single_model(config)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import os


    df = pd.read_csv(CSV_FILE)

    metrics = [
        ("avg_step_time_s", "Step time (s)"),
        ("avg_comm_time_s", "Comm time (s)"),
        ("comm_ratio", "Comm ratio"),
    ]

    model_order = ["small", "medium", "large", "xl", "2.7B"]

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for Overlap, label in [(False, "Per-parameter all-reduce"),
                                (True, "Overlap all-reduce")]:
            sub = df[df["overlap"] == Overlap]
            sub = sub.sort_values("model")

            ax.plot(
                sub["model"],
                sub[metric],
                marker="o",
                label=label,
            )

        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        ax.legend()

    axes[-1].set_xlabel("Model size")
    axes[0].set_title("Naive DDP: Per-parameter vs Overlap All-Reduce")

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(CSV_FILE))
    out_path = os.path.join(out_dir, "naive_ddp_Overlap_vs_individual.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


# cs336_systems/ddp_overlap_profile_benchmark.py
import os
import time
import csv
import math
import socket
import argparse
import pathlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cs336_basics.model as Model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

from cs336_systems.naive_ddp import (
    barrier,
    naive_ddp_on_after_backward,
    get_naive_ddp,
)
from cs336_systems.ddp_overlap_individual_parameters import (
    DDPIndividualParameters,
)

# -----------------------------
# helpers
# -----------------------------
def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def setup_process_group_nccl(rank: int, world_size: int, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    assert torch.cuda.is_available(), "Need 2 GPUs for this benchmark"
    ngpu = torch.cuda.device_count()
    assert ngpu >= world_size, f"Need >= {world_size} GPUs, but have {ngpu}"

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,   # important for correct NCCL barrier mapping
    )
    return device

def nvtx_range(msg: str, enable: bool):
    # lightweight NVTX markers for Nsight Systems
    if not enable:
        return
    torch.cuda.nvtx.range_push(msg)

def nvtx_pop(enable: bool):
    if not enable:
        return
    torch.cuda.nvtx.range_pop()

def write_rows_csv(path: str, fieldnames: list[str], rows: list[dict]):
    is_new = not os.path.exists(path)
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r)

@dataclass
class XLConfig:
    model_size: str = "xl"
    vocab_size: int = 10000
    context_length: int = 128
    d_model: int = 1600
    d_ff: int = 6400
    num_layers: int = 48
    num_heads: int = 25
    rope_theta: int = 10000
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    global_batch_size: int = 2  # per your earlier code; keep small for profiling

@dataclass
class LargeConfig:
    model_size: str = "large"
    vocab_size: int = 10000
    context_length: int = 128
    d_model: int = 1280
    d_ff: int = 5120
    num_layers: int = 36
    num_heads: int = 20
    rope_theta: int = 10000
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    global_batch_size: int = 2  # per your earlier code; keep small for profiling
    
def run_one_rank(
    rank: int,
    world_size: int,
    impl: str,
    warmup: int,
    steps: int,
    out_dir: str,
    master_port: int,
    enable_nvtx: bool,
):
    device = setup_process_group_nccl(rank, world_size, master_port)
    barrier("nccl", device)

    # Make runs deterministic-ish
    torch.manual_seed(1234 + rank)
    torch.cuda.manual_seed_all(5678 + rank)

    cfg = LargeConfig()

    model = Model.BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        rope_theta=cfg.rope_theta,
    ).to(device)

    if impl == "naive":
        ddp_model = get_naive_ddp(model)
    elif impl == "overlap":
        ddp_model = DDPIndividualParameters(model)
    else:
        raise ValueError(f"unknown impl: {impl}")

    opt = AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # fixed synthetic data
    x_all = torch.randint(0, cfg.vocab_size, (cfg.global_batch_size, cfg.context_length), device=device)
    y_all = torch.randint(0, cfg.vocab_size, (cfg.global_batch_size, cfg.context_length), device=device)

    assert x_all.size(0) % world_size == 0
    local_bs = x_all.size(0) // world_size
    x = x_all[rank * local_bs : (rank + 1) * local_bs]
    y = y_all[rank * local_bs : (rank + 1) * local_bs]

    # Warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        out = ddp_model(x)
        loss = cross_entropy(out, y)
        loss.backward()
        if impl == "naive":
            naive_ddp_on_after_backward(ddp_model)
        else:
            ddp_model.finish_gradient_synchronization()
        opt.step()

    torch.cuda.synchronize()
    barrier("nccl", device)

    # Timed steps (record per-step)
    step_times = []
    for step in range(steps):
        barrier("nccl", device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)

        nvtx_range("forward", enable_nvtx)
        out = ddp_model(x)
        nvtx_pop(enable_nvtx)

        nvtx_range("loss", enable_nvtx)
        loss = cross_entropy(out, y)
        nvtx_pop(enable_nvtx)

        nvtx_range("backward", enable_nvtx)
        loss.backward()
        nvtx_pop(enable_nvtx)

        # gradient sync timing point (where overlap differs)
        nvtx_range("grad_sync_finish", enable_nvtx)
        if impl == "naive":
            naive_ddp_on_after_backward(ddp_model)   # synchronous all-reduce after backward
        else:
            ddp_model.finish_gradient_synchronization()  # waits for async all-reduces kicked off by hooks
        nvtx_pop(enable_nvtx)

        nvtx_range("optimizer_step", enable_nvtx)
        opt.step()
        nvtx_pop(enable_nvtx)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    # reduce across ranks: take MAX per step (true iteration time)
    step_t = torch.tensor(step_times, device=device, dtype=torch.float64)
    dist.all_reduce(step_t, op=dist.ReduceOp.MAX)

    if rank == 0:
        csv_path = os.path.join(out_dir, f"ddp_{impl}_xl_1node2gpus_steps.csv")
        rows = []
        for i, s in enumerate(step_t.tolist()):
            rows.append({
                "impl": impl,
                "model": cfg.model_size,
                "world_size": world_size,
                "global_bs": cfg.global_batch_size,
                "local_bs": local_bs,
                "context_length": cfg.context_length,
                "step": i,
                "step_time_s": s,
            })
        write_rows_csv(
            csv_path,
            fieldnames=list(rows[0].keys()),
            rows=rows
        )

        mean_s = float(step_t.mean().item())
        print(f"[rank0] impl={impl} mean_step_time = {mean_s*1000:.3f} ms  (saved {csv_path})")

    barrier("nccl", device)
    dist.destroy_process_group()

def ddp_overlap_nsys_benchmarking():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=str, choices=["naive", "overlap"], required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="../data/ddp_overlap_individual_parameters_benchmarking")
    parser.add_argument("--nvtx", action="store_true", help="add NVTX ranges for Nsight Systems")
    args = parser.parse_args()

    world_size = 2
    mp.set_start_method("spawn", force=True)

    master_port = find_free_port()
    mp.spawn(
        run_one_rank,
        args=(world_size, args.impl, args.warmup, args.steps, args.out_dir, master_port, args.nvtx),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    ddp_overlap_nsys_benchmarking()

'''
在assignment2/cs336_systems路径下运行
/root/data1/Nsight/nsight-systems/bin/nsys profile \
--trace=cuda,nvtx --sample=none --cpuctxsw=none \
--force-overwrite true \
-o ../data/ddp_overlap_individual_parameters_benchmarking/ddp_overlap_benchmark_naive \
python ddp_overlap_individual_parameters.py --nvtx --impl naive
以及
/root/data1/Nsight/nsight-systems/bin/nsys profile \
--trace=cuda,nvtx --sample=none --cpuctxsw=none \
--force-overwrite true \
-o ../data/ddp_overlap_individual_parameters_benchmarking/ddp_overlap_benchmark_overlap \
python ddp_overlap_individual_parameters.py --nvtx --impl overlap
'''