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
from tests.common import (
    ToyModel,
    ToyModelWithTiedWeights,
)
FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"

def barrier(backend, device):
    if backend == "nccl":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
def validate_ddp_net_equivalence(net):
    '''这个 helper 只支持 DDP wrapper，因为真正 PyTorch 的 DistributedDataParallel 包装后是 ddp_model.module 才是原始模型。本 naive 版本返回的是裸 ToyModel，没有 .module 属性'''
    # Helper to validate synchronization of nets across ranks.
    module = net.module if hasattr(net, "module") else net # 健壮的写法
    net_module_states = list(module.state_dict().values()) # net.module.state_dict().values()会报错：'ToyModel' object has no attribute 'module'。去掉.module
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t) # gather得到所有rank上的对应weights，放在tensor_list
        for tensor in tensor_list:
            assert torch.allclose(tensor, t) # 比较本rank上的weights和其他rank上的对应weights是否一致

  
def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend, rank=rank, world_size=world_size,device_id=device) # GPU下可能这里需要指定device id
        else:
            raise ValueError("Unable to find CUDA devices.")
        # device = f"cuda:{local_rank}"
    else:
        device = "cpu"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    # initialize the process group
    # dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    return device

def get_naive_ddp(model):
    # assume dist already initialized
    seen = set()
    for p in model.parameters():
        if id(p) in seen: 
            continue
        seen.add(id(p))
        dist.broadcast(p.data, src=0) # rank0 是发送源。所有其他 rank 都会接收 rank0 的 tensor 内容，把自己的 p.data 覆盖成 rank0 的值。注意：每个 rank 都必须调用这行代码（同一个顺序、同一组 tensor），只是它们扮演的角色不同：rank0 发送，其它 rank 接收。
        # 所有rank都进入get_naive_ddp函数调用上述代码，这一点很重要。如果只是用if rank == 0: dist.broadcast(...)这种办法的话，只有rank0进入这段代码，其他rank不能进入，导致卡死，因为rank0等不到接收方
    for b in model.buffers():
        dist.broadcast(b.data, src=0)
    return model

def naive_ddp_on_after_backward(model):
    world_size = dist.get_world_size()
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if (not p.requires_grad) or (p.grad is None):
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)

def batched_all_reduce_grad(model):
    world_size = dist.get_world_size()
    seen = set()
    grad_list = []
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if (not p.requires_grad) or (p.grad is None):
            continue
        grad_list.append(p.grad)
    if not grad_list: return
    flat_grads = torch.nn.utils.parameters_to_vector(grad_list)
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(world_size)
    torch.nn.utils.vector_to_parameters(flat_grads, grad_list)

def _test_NaiveDistributedDataParallel(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="nccl")
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    barrier(backend='nccl',device=device)

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    # Create a toy model and move it to the proper device.
    # This is our non-parallel baseline.
    non_parallel_model = model_class().to(device)

    # Create a DDP model. Note that the weights of this model should
    # match the non-parallel baseline above.
    ddp_base = deepcopy(non_parallel_model) # 每个rank都会拿到一个non_parallel_model的完整copy，用来初始化本地模型参数，以及作为对比
    ddp_model = get_naive_ddp(ddp_base) # 这一步是在干嘛？如果只是为了将参数广播到其他rank上，为什么不写一个判断，使得只有master rank广播？并且为什么要返回ddp_model，后面直接在ddp_base上更新参数不行吗？

    # If we're on rank 0, the DDP model should still exactly match the parameters of the
    # non-parallel baseline (since the parameters on rank 0 weren't changed).
    # If we're not on rank 0, the DDP model's parameters should have been updated with
    # the parameters from rank 0. So, double-check that the parameter differ from the
    # local initial state.
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        # This parameter was initialized as [2, 2], so we expect its value to remain the same
        is_no_grad_fixed_param = (
            "no_grad_fixed_param" in ddp_model_param_name or "no_grad_fixed_param" in non_parallel_param_name
        )
        if rank == 0 or is_no_grad_fixed_param:
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(ddp_model)

    # Load the dataset from disk, so we can ensure that every rank has the same
    # overall pool of data.
    # Shape: (20, 10)
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    # Shape: (20, 5)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")

    # Each rank will see only 10 examples (out of the total dataset size of 20)
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    for i in range(5):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # Run the non-parallel model on all the data and take a gradient step
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # At this point, the parameters of non-parallel model should differ
        # from the parameters of the DDP model (since we've applied the
        # gradient step to the non-parallel model, but not to the DDP model).
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    # The only parameters that change are those that require_grad
                    assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
                else:
                    # parameters that don't require_grad shouldn't change
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # While the non-parallel model does a forward pass on all the data (20 examples),
        # each DDP rank only sees 10 (disjoint) examples.
        # However, the end result should be the same as doing a forward pass on all 20 examples.
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()

        # 这一步将各个rank上的梯度做了all reduce，使每个rank上都得到了所有数据上的平均梯度
        naive_ddp_on_after_backward(ddp_model)

        ddp_optimizer.step()

        # At this point, the non-parallel model should exactly match the parameters of the DDP model
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # Shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        # We make sure to use the same seed when shuffling (else the per-rank examples might not be disjoint).
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    # After training is done, we should have the same weights on both the non-parallel baseline
    # and the model trained with DDP.
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
    barrier(backend='nccl',device=device)
    dist.destroy_process_group()

def test_NaiveDistributedDataParallel(model_class):
    world_size = 2
    mp.spawn(# type: ignore[attr-defined]
        _test_NaiveDistributedDataParallel,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )
    
if __name__ == "__main__":
    model_list = [ToyModel, ToyModelWithTiedWeights]
    for i in model_list:
        test_NaiveDistributedDataParallel(i)
    print('Done!')