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
from typing import Type

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

data_path = pathlib.Path('../data/naive_ddp_benchmarking/')
data_path.mkdir(parents=True, exist_ok=True)
CSV_FILE = "../data/naive_ddp_benchmarking/naive_ddp_benchmarking.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _test_NaiveDistributedDataParallel(rank: int, world_size: int, warmup, step_num, config):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="nccl")

    barrier(backend='nccl',device=device)

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    # Create a toy model and move it to the proper device.
    # This is our non-parallel baseline.
    print(f'[rank {rank}] Initializing model...')
    non_parallel_model = Model.BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        rope_theta=config['rope_theta']
    )

    non_parallel_model = non_parallel_model.to(device)

    ddp_model = get_naive_ddp(non_parallel_model) 

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
            # 这一步将各个rank上的梯度做了all reduce，使每个rank上都得到了所有数据上的平均梯度
            if not config['batched_all_reduce']: naive_ddp_on_after_backward(ddp_model)
            else: batched_all_reduce_grad(ddp_model)
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
                'batched_all_reduce': config['batched_all_reduce'],
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


def benchmarking_naive_ddp(config):
    torch.set_float32_matmul_precision('high')
    step_num = 20
    warmup_num = 2
    world_size = 2
    mp.spawn(# type: ignore[attr-defined]
        _test_NaiveDistributedDataParallel,
        args=(world_size, warmup_num, step_num, config),
        nprocs=world_size,
        join=True,
    )

def minimal_ddp_flat_benchmarking(config):
    torch.set_float32_matmul_precision('high')
    step_num = 50
    warmup_num = 5
    world_size = 2
    mp.spawn(# type: ignore[attr-defined]
        _test_NaiveDistributedDataParallel,
        args=(world_size, warmup_num, step_num, config),
        nprocs=world_size,
        join=True,
    )
if __name__ == "__main__":

    MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    # "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    # "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
    for i in range(2):
        for k,v in MODEL_SPECS.items():       
            config={
            'device': device,
            'batched_all_reduce': i == 0,
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
            print("Start benchmarking model: ", k, " ,batched_all_reduce: ", config['batched_all_reduce'])
            benchmarking_naive_ddp(config)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    CSV_PATH = "../data/naive_ddp_benchmarking/naive_ddp_benchmarking.csv"

    df = pd.read_csv(CSV_PATH)

    metrics = [
        ("avg_step_time_s", "Step time (s)"),
        ("avg_comm_time_s", "Comm time (s)"),
        ("comm_ratio", "Comm ratio"),
    ]

    model_order = ["small", "medium", "large", "xl", "2.7B"]

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for batched, label in [(False, "Per-parameter all-reduce"),
                                (True, "Batched all-reduce")]:
            sub = df[df["batched_all_reduce"] == batched]
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
    axes[0].set_title("Naive DDP: Per-parameter vs Batched All-Reduce")

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(CSV_PATH))
    out_path = os.path.join(out_dir, "naive_ddp_batched_vs_individual.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


