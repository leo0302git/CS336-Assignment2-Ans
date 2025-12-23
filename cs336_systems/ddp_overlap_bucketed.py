import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import torch
import torch.distributed as dist
import torch.nn as nn
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

import cs336_basics.model as Model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.ddp_overlap_individual_parameters import DDPIndividualParameters
@dataclass
class _BucketState:
    ready_count: int = 0
    launched: bool = False
    work: Optional[dist.Work] = None

class _Bucket:
    """
    A fixed-layout bucket:
      - params: list of Parameters
      - offsets: fixed slice in a flat buffer for each param
      - buffer: flat tensor holding concatenated grads
    Dynamic per-iteration state:
      - param_ready flags, ready_count, launched, work handle
    """

    def __init__(self, params: List[nn.Parameter], device: torch.device, dtype: torch.dtype):
        assert len(params) > 0
        self.params = params

        # fixed offsets in flat buffer (in number of elements)
        self.offsets: Dict[nn.Parameter, Tuple[int, int]] = {}
        off = 0
        for p in params:
            n = p.numel()
            self.offsets[p] = (off, off + n)
            off += n
        self.numel = off

        # flat buffer lives on same device/dtype as params
        self.buffer = torch.empty(self.numel, device=device, dtype=dtype)

        # dynamic state
        self.state = _BucketState()
        self.param_ready: Dict[nn.Parameter, bool] = {p: False for p in params}

    def reset(self):
        self.state = _BucketState()
        for p in self.param_ready:
            self.param_ready[p] = False


class DDPBucketed(nn.Module):
    """
    Bucketed DDP that overlaps backward compute with communication.

    Key properties for correctness:
      - Buckets are built deterministically on ALL ranks, in the same order.
      - Each bucket launches exactly one async all-reduce when all of its params are ready.
      - finish_gradient_synchronization() waits for all in-flight works and writes results back to param.grad.

    Implementation notes:
      - Uses register_post_accumulate_grad_hook like your DDPIndividualParameters,
        so param.grad is already materialized when the hook runs.
      - Buckets are built in *inverse* order of model.parameters() (as requested).
      - bucket_size_mb is float (as requested).
    """

    def __init__(
        self,
        module: nn.Module,
        bucket_size_mb: float = 25.0,
        process_group=None,
    ):
        super().__init__()
        assert dist.is_initialized(), "Process group must be initialized before constructing DDPBucketed."
        self.module = module
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)

        # allow float MB
        self.bucket_cap_bytes: int = max(1, int(bucket_size_mb * 1024 * 1024))

        # build buckets and register hooks
        self._buckets: List[_Bucket] = []
        self._param_to_bucket: Dict[nn.Parameter, _Bucket] = {}
        self._hook_handles = []

        # make sure params/buffers are identical across ranks (same as your other DDPs)
        self._broadcast_module_state()

        # build buckets deterministically (inverse order)
        self._build_buckets_inverse_param_order()

        # register hooks
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # -----------------------------
    # initialization helpers
    # -----------------------------
    def _broadcast_module_state(self):
        # broadcast parameters
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, group=self.process_group)
        # broadcast buffers (e.g., layernorm running stats if any)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0, group=self.process_group)

    def _iter_unique_trainable_params_inverse(self) -> List[nn.Parameter]:
        """
        Return trainable parameters in *inverse* order of model.parameters(),
        deduplicating shared/tied params by id (consistent across ranks).
        """
        params = list(self.module.parameters())
        params.reverse()

        seen = set()
        uniq: List[nn.Parameter] = []
        for p in params:
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)
        return uniq

    def _build_buckets_inverse_param_order(self):
        uniq_params = self._iter_unique_trainable_params_inverse()
        if len(uniq_params) == 0:
            return

        # choose device/dtype for buffers (assume all params are on same device/dtype in this assignment)
        device = uniq_params[0].device
        dtype = uniq_params[0].dtype

        cur: List[nn.Parameter] = []
        cur_bytes = 0

        for p in uniq_params:
            p_bytes = p.numel() * p.element_size()

            # if single param larger than cap: put it alone in its own bucket
            if p_bytes >= self.bucket_cap_bytes and len(cur) == 0:
                b = _Bucket([p], device=device, dtype=dtype)
                self._buckets.append(b)
                self._param_to_bucket[p] = b
                continue

            if cur and (cur_bytes + p_bytes > self.bucket_cap_bytes):
                b = _Bucket(cur, device=device, dtype=dtype)
                self._buckets.append(b)
                for q in cur:
                    self._param_to_bucket[q] = b
                cur = []
                cur_bytes = 0

            cur.append(p)
            cur_bytes += p_bytes

        if cur:
            b = _Bucket(cur, device=device, dtype=dtype)
            self._buckets.append(b)
            for q in cur:
                self._param_to_bucket[q] = b

    def _register_hooks(self):
        """
        Use post-accumulate hook so param.grad is populated when hook runs.
        """
        for p in self._param_to_bucket.keys():
            bucket = self._param_to_bucket[p]

            def make_hook(param: nn.Parameter, b: _Bucket):
                def hook(_param: nn.Parameter):
                    # post_accumulate hook signature provides param; grad is in param.grad now
                    g = _param.grad
                    if g is None:
                        # This can happen if param didn't participate in graph for this iteration.
                        # Don't mark ready; otherwise we'd deadlock waiting for it.
                        return

                    # de-dup within iteration
                    if b.param_ready[param]:
                        return

                    b.param_ready[param] = True
                    b.state.ready_count += 1

                    # pack grad into bucket buffer (handle non-contiguous grads)
                    start, end = b.offsets[param]
                    b.buffer[start:end].copy_(g.reshape(-1))

                    # launch exactly once
                    if (b.state.ready_count == len(b.params)) and (not b.state.launched):
                        b.state.launched = True
                        b.state.work = dist.all_reduce(
                            b.buffer,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                            group=self.process_group,
                        )

                return hook

            h = p.register_post_accumulate_grad_hook(make_hook(p, bucket))
            self._hook_handles.append(h)

    # -----------------------------
    # synchronization API
    # -----------------------------
    def finish_gradient_synchronization(self):
        """
        Wait for all bucket all-reduces, then write reduced grads back to param.grad.
        Safe on CPU and GPU.

        Important:
          - We divide by world_size to match "average gradient" semantics like your other DDPs.
          - We reset bucket dynamic state at the end so next iteration works.
        """
        for b in self._buckets:
            # If bucket never launched (e.g., some params unused), there is nothing to wait on.
            if b.state.work is not None:
                b.state.work.wait()

                # scatter back into each param.grad
                for p in b.params:
                    start, end = b.offsets[p]
                    reduced_flat = b.buffer[start:end] / self.world_size

                    # p.grad might be None if param unused; skip safely
                    if p.grad is None:
                        continue

                    # write back with correct shape (handle possible non-contiguous grad buffer)
                    # safest: overwrite grad data via copy_ into a contiguous view
                    p.grad.copy_(reduced_flat.view_as(p.grad))

            # reset per-iteration state
            b.reset()

    def zero_grad(self, set_to_none: bool = True):
        """
        Optional convenience wrapper.
        Reset bucket state so next iteration can launch again.
        """
        self.module.zero_grad(set_to_none=set_to_none)
        for b in self._buckets:
            b.reset()
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
                'impl': 'overlap',
                'bucket_size_mb':config['bucket_size_mb'],
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

def _test_BucketedDistributedDataParallel(rank: int, world_size: int, warmup, step_num, config, CSV_FILE):
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


    ddp_model = DDPBucketed(non_parallel_model, config['bucket_size_mb'])

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
                'impl': 'bucketed',
                'bucket_size_mb':config['bucket_size_mb'],
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

        # 到这里，所有rank���经获得step_times以及comm_times
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
                'impl': 'naive',
                'bucket_size_mb':config['bucket_size_mb'],
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


def ddp_bucketed_benchmarking_single_model(config, CSV_FILE):
    torch.set_float32_matmul_precision('high')
    step_num = 20
    warmup_num = 2
    world_size = 2
    mp.spawn(# type: ignore[attr-defined]
        _test_BucketedDistributedDataParallel,
        args=(world_size, warmup_num, step_num, config, CSV_FILE),
        nprocs=world_size,
        join=True,
    )
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
    

def ddp_bucketed_benchmarking():
    data_path = pathlib.Path('../data/ddp_bucketed_individual_parameters_benchmarking/')
    data_path.mkdir(parents=True, exist_ok=True)
    CSV_FILE = "../data/ddp_bucketed_individual_parameters_benchmarking/ddp_bucketed_individual_parameters_benchmarking.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    # "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    # "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    # "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
    bucket_sizes_mb = [1,10,100,1000]
    for bucket_size_mb in bucket_sizes_mb:
        for k,v in MODEL_SPECS.items():       
            config={
            'device': device,
            'bucket_size_mb': bucket_size_mb,
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
            ddp_bucketed_benchmarking_single_model(config, CSV_FILE)
    
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_ddp_by_modelsize_with_bucket_lines(
    csv_file: str,
    out_dir: str,
    model_order=("small", "medium", "large"),
    bucket_sizes_mb=(1, 10, 100, 1000),
    setting_filter="1node_2gpus",
):
    os.makedirs(out_dir, exist_ok=True)
    df0 = pd.read_csv(csv_file)

    # 可选：只看指定 setting
    if "setting" in df0.columns and setting_filter is not None:
        df0 = df0[df0["setting"] == setting_filter].copy()

    # 规范类型
    df0["bucket_size_mb"] = pd.to_numeric(df0.get("bucket_size_mb"), errors="coerce")
    df0["model"] = df0["model"].astype(str)
    df0["impl"] = df0["impl"].astype(str)

    # 只保留我们关心的模型
    df0 = df0[df0["model"].isin(model_order)].copy()
    if df0.empty:
        raise ValueError("No rows left after filtering by model_order/setting. Check CSV content.")

    # 聚合：同一 (impl, bucket_size_mb, model) 可能跑多次，取 mean
    agg = (
        df0.groupby(["impl", "bucket_size_mb", "model"], as_index=False)
           .agg(
               avg_step_time_s=("avg_step_time_s", "mean"),
               avg_comm_time_s=("avg_comm_time_s", "mean"),
               comm_ratio=("comm_ratio", "mean"),
           )
    )

    metrics = [
        ("avg_step_time_s", "Avg step time", 1000.0, "ms"),
        ("avg_comm_time_s", "Avg comm time", 1000.0, "ms"),
        ("comm_ratio", "Comm ratio", 1.0, "fraction"),
    ]

    # 6 条线：naive / overlap + bucketed@{size}
    lines = [
        ("naive", None, "Naive (no overlap)"),
        ("overlap", None, "Overlap (per-parameter)"),
        *[("bucketed", float(bs), f"Bucketed @{bs}MB") for bs in bucket_sizes_mb],
    ]

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)

    x_labels = list(model_order)
    x_positions = list(range(len(x_labels)))

    for ax, (metric, title, scale, yunit) in zip(axes, metrics):
        for impl, bs, label in lines:
            if impl in ("naive", "overlap"):
                sub = agg[agg["impl"] == impl].copy()
                sub = sub.groupby("model", as_index=False)[metric].mean()
            else:
                sub = agg[(agg["impl"] == "bucketed") & (agg["bucket_size_mb"] == bs)].copy()

            y = []
            for m in model_order:
                row = sub[sub["model"] == m]
                if row.empty:
                    y.append(float("nan"))
                else:
                    y.append(float(row.iloc[0][metric]) * scale)

            ax.plot(x_positions, y, marker="o", label=label)

        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(yunit)

        if metric == "comm_ratio":
            ax.set_ylim(0.0, 1.0)

        # ✅ 每个子图都显示图例（每条线都有 label）
        ax.legend(loc="best", frameon=False, fontsize=9)

    # ✅ suptitle 放进画布内 + tight_layout 预留顶部空间
    fig.suptitle("DDP comparison across model sizes (1×3 metrics)", y=0.98, fontsize=13)

    # rect=[left, bottom, right, top]：给 suptitle 留空间（top=0.92 左右）
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    figpath = os.path.join(out_dir, "bucketed_benchmark.png")
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # ddp_bucketed_benchmarking()
    plot_ddp_by_modelsize_with_bucket_lines(
csv_file="../data/ddp_bucketed_individual_parameters_benchmarking/ddp_bucketed_individual_parameters_benchmarking.csv",
    out_dir="../data/ddp_bucketed_individual_parameters_benchmarking/",
    model_order=["small", "medium"],
    bucket_sizes_mb=[1, 10, 100, 1000],
)
