import os
import time
import socket
import argparse
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import csv
from pathlib import Path


data_path = Path('../data/distributed_communication_single_node/')
data_path.mkdir(parents=True, exist_ok=True)
CSV_FILE = "../data/distributed_communication_single_node/allreduce_benchmark.csv"

def write_csv_row(row: dict, CSV_FILE = CSV_FILE):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# def setup(rank: int, world_size: int, backend: str, host: str, port: int):
#     # 显式关掉 libuv（关键！）
#     is_master = (rank == 0)
#     store = dist.TCPStore(# type: ignore[attr-defined]
#         host_name=host,
#         port=port,
#         world_size=world_size,
#         is_master=is_master,
#         timeout=datetime.timedelta(seconds=120),
#         wait_for_workers=True,
#         use_libuv=False,   # <<< 关键
#     )
#     dist.init_process_group(
#         backend=backend,
#         store=store,
#         rank=rank,
#         world_size=world_size,
#     )

def iters_for_bytes(nbytes: int) -> tuple[int, int]:
    if nbytes >= 1_000_000_000: return 2, 5
    if nbytes >= 100_000_000:   return 5, 20
    if nbytes >= 10_000_000:    return 10, 50
    return 20, 200

def barrier(backend, device):
    if backend == "nccl":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()

def worker(rank, world_size, backend, device_type, nbytes, host, port):
    # 绑定设备
    if device_type == "cuda":
        assert torch.cuda.is_available()
        ngpu = torch.cuda.device_count()
        local_gpu = rank % ngpu
        torch.cuda.set_device(local_gpu)
        device = torch.device("cuda", local_gpu)
    else:
        device = torch.device("cpu")

    # init pg
    is_master = (rank == 0)
    store = dist.TCPStore(# type: ignore[attr-defined]
        host_name=host,
        port=port,
        world_size=world_size,
        is_master=is_master,
        timeout=datetime.timedelta(seconds=120),
        wait_for_workers=True,
        use_libuv=False,
    )

    if backend == "nccl":
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
            device_id=device,   # 关键
        )
    else:
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
        )

    try:
        # 计时前同步
        barrier(backend, device)
        if device_type == "cuda":
            torch.cuda.synchronize()

        numel = nbytes // 4
        x = torch.empty(numel, device=device, dtype=torch.float32).uniform_()

        warmup, iters = iters_for_bytes(nbytes)
        for _ in range(warmup):
            dist.all_reduce(x)

        barrier(backend, device)
        if device_type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        barrier(backend, device)
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if rank == 0:
            avg_ms = (t1 - t0) * 1000.0 / iters
            gbps = (nbytes / 1e9) / ((t1 - t0) / iters)

            print(f"[{backend} {device_type}] size={nbytes/1e6:.0f}MB world={world_size} "
                f"iters={iters} avg={avg_ms:.3f}ms ~{gbps:.2f}GB/s")

            write_csv_row({
                "backend": backend,
                "device": device_type,
                "world_size": world_size,
                "size_mb": nbytes / 1024 / 1024,
                "avg_ms": avg_ms,
                "gbps": gbps,
            })
    finally:
        # 无论中途谁抛异常，都尽量 clean shutdown。否则一个进程异常会导致其他进程中断
        if dist.is_initialized():
            dist.destroy_process_group()

def run_one(world_size: int, backend: str, device_type: str, nbytes: int):
    host = "127.0.0.1"
    port = find_free_port()
    mp.spawn( # type: ignore[attr-defined]
        worker,
        args=(world_size, backend, device_type, nbytes, host, port),
        nprocs=world_size, join=True)

def main():
    # 有gpu时使用--cpu --gpu命令运行脚本
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2,4,6])
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[1,10,100,1000])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    tests = []
    if args.cpu or (not args.cpu and not args.gpu):
        tests.append(("gloo", "cpu"))
    if args.gpu:
        print('Has gpu')
        tests.append(("nccl", "cuda"))

    for backend, device_type in tests:
        print(f'[{backend} {device_type}]')
        for mb in args.sizes_mb:
            nbytes = mb * 1024 * 1024
            for ws in args.world_sizes:
                if device_type == "cuda" and torch.cuda.is_available():
                    if ws > torch.cuda.device_count():
                        print(f"Skip: world_size={ws} > num_gpus={torch.cuda.device_count()}")
                        continue
                run_one(ws, backend, device_type, nbytes)

if __name__ == "__main__":
    # main()
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("../data/distributed_communication_single_node/allreduce_benchmark.csv")

    # 示例 1：固定 size，比较 world_size
    size_mb = 10
    sub = df[df["size_mb"] == size_mb]

    for (backend, device), g in sub.groupby(["backend", "device"]):
        plt.plot(g["world_size"], g["gbps"], marker="o", label=f"{backend}-{device}")

    plt.xlabel("World size")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title(f"All-reduce bandwidth vs world size ({size_mb}MB)")
    plt.legend()
    plt.grid(True)
    out_path = f"{data_path}/All_reduce_bandwidth_vs_world_size_{size_mb}MB.png"
    plt.savefig(out_path, dpi=200)

    sub = df[df["world_size"] == 2]

    for (backend, device), g in sub.groupby(["backend", "device"]):
        plt.plot(g["size_mb"], g["gbps"], marker="o", label=f"{backend}-{device}")

    plt.xscale("log")
    plt.xlabel("Tensor size (MB)")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title("All-reduce bandwidth vs tensor size (world_size=2)")
    plt.legend()
    plt.grid(True)
    out_path = f"{data_path}/All_reduce_bandwidth_vs_tensor_size_world_size_2.png"
    plt.savefig(out_path, dpi=200)
