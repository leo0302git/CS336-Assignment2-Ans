import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile

# 直接使用作业文档中的代码无法在CPU机器上运行
# import os
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp

# def setup(rank, world_size):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def distributed_demo(rank, world_size):
#     setup(rank, world_size)
#     data = torch.randint(0, 10, (3,))
#     print(f"rank {rank} data (before all-reduce): {data}")
#     dist.all_reduce(data, async_op=False)
#     print(f"rank {rank} data (after all-reduce): {data}")

# if __name__ == "__main__":
#     world_size = 4
#     mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
#  PyTorch 编译时未集成 libuv 库（TCPStore 依赖的网络通信库），导致基于 gloo 后端的 TCP 通信无法正常初始化。Windows 系统下的 PyTorch 预编译包常存在这个问题，且 spawn 启动方式 +gloo 后端的组合更容易触发该限制。
# 改用文件系统共享的方式初始化进程组，完全绕开 libuv 依赖。仅适用于单机多进程，跨机器分布式仍需 TCP 通信
def setup(rank, world_size):
    # 创建临时文件路径（所有进程可访问）
    temp_dir = tempfile.gettempdir()
    init_method = f"file://{os.path.join(temp_dir, 'torch_dist_temp')}"
    
    # 初始化进程组（gloo后端 + file初始化）
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,), device='cpu')  # 显式指定CPU
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")
    # 销毁进程组（避免资源泄漏）
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    # Windows下set_start_method需加try-except（避免重复设置）
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    # 启动多进程
    mp.spawn(
        fn=distributed_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )