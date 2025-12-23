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
# import torch.nn as nn
# from torch.profiler import ProfilerActivity
# from torch.utils.cpp_extension import load_inline
# import triton
# import triton.language as tl
# import math
# import os
import timeit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config={
    'device': device,
    'vocab_size':10000,
    'context_length':128,
    'd_model':768,
    'd_ff':3072,
    'num_heads':12,
    'num_layers':12,
    'rope_theta':10000,
    'lr':1e-3,
    'betas':(0.9, 0.999),
    'eps':1e-8,
    'weight_decay':0.01,
    'batch_size': 32
}
    


def train_sim(model: Model.BasicsTransformerLM, opt: torch.optim.Optimizer,step_num: int, forward_only: bool, device):
    # forward_time = []
    # backward_time = []
    for _ in range(step_num):
        # start_time = time.time()
        pred_y = model(x)
        pred_y_flatten = rearrange(pred_y, 'batch seq_len vocab -> (batch seq_len) vocab')
        y_flatten = rearrange(y, 'batch seq_len -> (batch seq_len)')
        loss = cross_entropy(pred_y_flatten, y_flatten)
        # forward_done = time.time()
        # forward_time.append(forward_done - start_time)
        if not forward_only:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            # backward_done = time.time()
            # backward_time.append(backward_done - forward_done)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # avg_forward_time = sum(forward_time) / len(forward_time)
        # avg_backward_time = sum(backward_time) / len(backward_time)
        # return avg_forward_time, avg_backward_time
def train_sim_xy(model: Model.BasicsTransformerLM, opt: torch.optim.Optimizer,step_num: int, forward_only: bool, device, x, y):
    # forward_time = []
    # backward_time = []
    for _ in range(step_num):
        # start_time = time.time()
        pred_y = model(x)
        pred_y_flatten = rearrange(pred_y, 'batch seq_len vocab -> (batch seq_len) vocab')
        y_flatten = rearrange(y, 'batch seq_len -> (batch seq_len)')
        loss = cross_entropy(pred_y_flatten, y_flatten)

        if not forward_only:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

def benchmark_train(x, y, fun: Callable, trial_num:int, warmup_num: int, model: Model.BasicsTransformerLM, opt: torch.optim.Optimizer,step_num: int, forward_only: bool):
    # warmup
    fun(
        model=model,
        opt=opt,
        step_num=warmup_num,
        forward_only=forward_only,
        device=config['device'],
        x=x, y=y
    )
    time_list = []
    for _ in range(trial_num):
        t = timeit.timeit(
        lambda: fun(
            model=model,
            opt=opt,
            step_num=step_num,
            forward_only=forward_only,
            device=config['device'],x=x,y=y
        ),
        number=1
    )
        time_list.append(t)
    total_time = sum(time_list)
    std = np.std(time_list)
    mean_time = total_time / trial_num
    return mean_time, mean_time / step_num, std

def only_forward(model, opt, step_num, forward_only, device):
    times = []
    with torch.no_grad():
        model.eval()
        for _ in range(step_num):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append(t1 - t0)

    return sum(times) / len(times)
# def benchmark_forward(model, opt, x, num_steps):


def only_backward(model, opt, step_num, forward_only, device):
    times = []
    model.train()
    for _ in range(step_num):
        # start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pred_y = model(x)
        pred_y_flatten = rearrange(pred_y, 'batch seq_len vocab -> (batch seq_len) vocab')
        y_flatten = rearrange(y, 'batch seq_len -> (batch seq_len)')
        loss = cross_entropy(pred_y_flatten, y_flatten)
        
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
    return sum(times) / len(times)


def benchmarking_main():
    x = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    y = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    print(f'Using {device}. Initializing model...')
    model = Model.BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        rope_theta=config['rope_theta']
    ).to(device)
    opt = AdamW(
        params=model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )
    
    step_num = 50
    trial_num = 1
    warmup_num = 5
    forward_only = True
    print(f'Model initialized. Start benchmark. Trial num: {trial_num}')
    
    print('Standard benchmarking')
    mean_time, time_per_step, std= benchmark_train(x, y, train_sim_xy, trial_num=trial_num, warmup_num=warmup_num, model=model, opt=opt, step_num=step_num, forward_only=forward_only)
    print(f'Function: train_sim | : Warmup: {warmup_num} | Step: {step_num} | Forward_only: {forward_only} | Average train time(s): {mean_time:.4f} | Average time per step(s): {time_per_step:.4f} | Std: {std:.4f}')
    
    print('Benchmarking with backward')
    forward_only = False
    mean_time, time_per_step, std= benchmark_train(x, y, train_sim_xy,trial_num=trial_num, warmup_num=warmup_num, model=model, opt=opt, step_num=step_num, forward_only=forward_only)
    print(f'Function: train_sim | Warmup: {warmup_num} | Step: {step_num} | Forward_only: {forward_only} | Average train time(s): {mean_time:.4f} | Average time per step(s): {time_per_step:.4f} | Std: {std:.4f}')
    
    # print('Counting forward time')
    # mean_time, time_per_step, std= benchmark_train(only_forward, trial_num=trial_num, warmup_num=warmup_num, model=model, opt=opt, step_num=step_num, forward_only=forward_only)
    # print(f'Function: only_forward | : Warmup: {warmup_num} | Step: {step_num} | Average train time(s): {mean_time:.4f} | Average time per step(s): {time_per_step:.4f} | Std: {std:.4f}')
    
    # print('Counting backward time')
    # mean_time, time_per_step, std= benchmark_train(only_backward, trial_num=trial_num, warmup_num=warmup_num, model=model, opt=opt, step_num=step_num, forward_only=forward_only)
    # print(f'Function: only_backward | : Warmup: {warmup_num} | Step: {step_num} | Average train time(s): {mean_time:.4f} | Average time per step(s): {time_per_step:.4f} | Std: {std:.4f}')

def benchmarking_compiled():
    x = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    y = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    print(f'Using {device}. Initializing model...')
    model = Model.BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        rope_theta=config['rope_theta']
    )
    opt = AdamW(
        params=model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )
    model = model.to(device)
    compiled_model = torch.compile(model, fullgraph=False)
    # assert isinstance(compiled_model, Model.BasicsTransformerLM)
    torch.set_float32_matmul_precision('high')
    step_num = 50
    trial_num = 2
    warmup_num = 5
    forward_only = True
    print(f'Model initialized. Start compiled benchmark. Trial num: {trial_num}')
    
    print('Benchmarking with only forward')
    mean_time_compiled, time_per_step, std= benchmark_train(x, y, train_sim_xy,trial_num=trial_num, warmup_num=warmup_num, model=compiled_model, opt=opt, step_num=step_num, forward_only=forward_only)
    
    mean_time, time_per_step, std= benchmark_train(x, y, train_sim_xy, trial_num=trial_num, warmup_num=warmup_num, model=model, opt=opt, step_num=step_num, forward_only=forward_only)
    
    print(f'[Forward only] Speed up by compile: {mean_time / mean_time_compiled:.2f}')
    
    print('Benchmarking with full train step')
    mean_time_compiled, time_per_step, std= benchmark_train(x, y, train_sim_xy,trial_num=trial_num, warmup_num=warmup_num, model=compiled_model, opt=opt, step_num=step_num, forward_only=forward_only)
    forward_only = False
    mean_time, time_per_step, std= benchmark_train(x, y, train_sim_xy, trial_num=trial_num, warmup_num=warmup_num, model=model, opt=opt, step_num=step_num, forward_only=forward_only)
    
    print(f'[Full train step] Speed up by compile: {mean_time / mean_time_compiled:.2f}')

if __name__ == "__main__":
    #benchmarking_main()
    benchmarking_compiled()


