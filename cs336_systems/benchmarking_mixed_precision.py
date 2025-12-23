import torch
import torch.nn as nn
import torch.nn.functional as F
import cs336_basics.model as Model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import time
from einops import rearrange
import timeit
import numpy as np
# ToyModel 按作业定义
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train_autocast(x, y, model: Model.BasicsTransformerLM, opt: torch.optim.Optimizer,step_num: int, forward_only: bool, device, autocast_dtype, with_autocast = True):
#     forward_time = []
#     backward_time = []
#     if torch.cuda.is_available(): torch.cuda.synchronize()
#     if with_autocast:
#         with torch.autocast(device_type=device, dtype=autocast_dtype):
#             for _ in range(step_num):
#                 start_time = time.time()
#                 pred_y = model(x)
#                 pred_y_flatten = rearrange(pred_y, 'batch seq_len vocab -> (batch seq_len) vocab')
#                 y_flatten = rearrange(y, 'batch seq_len -> (batch seq_len)')
#                 loss = cross_entropy(pred_y_flatten, y_flatten)
#                 forward_done = time.time()
#                 forward_time.append(forward_done - start_time)
#                 if not forward_only:
#                     opt.zero_grad(set_to_none=True)
#                     loss.backward()
#                     opt.step()
#                     backward_done = time.time()
#                     backward_time.append(backward_done - forward_done)
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#     else:
#         for _ in range(step_num):
#             start_time = time.time()
#             pred_y = model(x)
#             pred_y_flatten = rearrange(pred_y, 'batch seq_len vocab -> (batch seq_len) vocab')
#             y_flatten = rearrange(y, 'batch seq_len -> (batch seq_len)')
#             loss = cross_entropy(pred_y_flatten, y_flatten)
#             forward_done = time.time()
#             forward_time.append(forward_done - start_time)
#             if not forward_only:
#                 opt.zero_grad(set_to_none=True)
#                 loss.backward()
#                 opt.step()
#                 backward_done = time.time()
#                 backward_time.append(backward_done - forward_done)
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#     avg_forward_time = sum(forward_time) / len(forward_time)
#     avg_backward_time = sum(backward_time) / len(backward_time)
#     return avg_forward_time, avg_backward_time

# def benchmarking_train_autocast(trial_num:int, warmup_num: int, model: Model.BasicsTransformerLM, opt: torch.optim.Optimizer,step_num: int, forward_only: bool, with_autocast = True):
#     x = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
#     y = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to(device)
    
#     autocast_dtype = torch.bfloat16
#     # warmup
#     train_autocast(
#         x,y,
#         model=model,
#         opt=opt,
#         step_num=warmup_num,
#         forward_only=forward_only,
#         device=config['device'],
#         autocast_dtype=autocast_dtype,
#         with_autocast = with_autocast
#     )
#     time_list = []
#     for _ in range(trial_num):
#         t = timeit.timeit(
#         lambda: train_autocast(
#             x,y,
#             model=model,
#             opt=opt,
#             step_num=step_num,
#             forward_only=forward_only,
#             device=config['device'],
#             autocast_dtype=autocast_dtype,
#             with_autocast = with_autocast
#         ),
#         number=1
#     )
#         time_list.append(t)
#     total_time = sum(time_list)
#     std = np.std(time_list)
#     mean_time = total_time / trial_num
#     return mean_time, mean_time / step_num, std
    

  
def benchmarking_toymodel():
    B = 10
    in_feature = 5
    out_feature = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This problem expects running on GPU."

    model = ToyModel(in_features=in_feature, out_features=out_feature).to(device)
    print('Architecture tree of model:')
    print(model)
    model.train()

    # float 输入 (因为ToyModel中没有embedding层，所以我们需要直接为其输入embedding后的数据类型，即float)
    # 注意需要显式指定dtype
    x = torch.randn(B, in_feature, device=device, dtype=torch.float32)

    # target 是类别索引 (B,)
    y = torch.randint(0, out_feature, (B,), device=device, dtype=torch.int64)

    # 为了拿到中间激活的 dtype，这里用 hook 抓一下
    saved = {}
    def save_out(name):
        def hook(_module, _inp, out):
            saved[name] = out
        return hook

    h1 = model.fc1.register_forward_hook(save_out("fc1_out"))
    h2 = model.ln.register_forward_hook(save_out("ln_out"))

    print('model.named_modules()')
    for name, module in model.named_modules():
        print(name, module)
    # autocast FP16 / BF16
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)                # predicted logits
        loss = F.cross_entropy(logits, y)

    loss.backward()

    # hooks 用完就移除
    h1.remove()
    h2.remove()

    # 打印题目要的 dtype
    print("param dtype (fc1.weight):", model.fc1.weight.dtype)
    print("fc1 output dtype:", saved["fc1_out"].dtype)
    print("layernorm output dtype:", saved["ln_out"].dtype)
    print("logits dtype:", logits.dtype)
    print("loss dtype:", loss.dtype)
    print("grad dtype (fc1.weight.grad):", model.fc1.weight.grad.dtype)
# if __name__ == "__main__":
#     step_num = 10
#     trial_num = 5
#     warmup_num = 2
#     forward_only = True
#     MODEL_SPECS = {
#         "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
#         "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
#         "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
#         "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
#         "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
#     }
#     print(f'Using {device}. Initializing model...')
#     for k, v in MODEL_SPECS.items():
#         print('Model spec: ', k, v)
#         model = Model.BasicsTransformerLM(
#             vocab_size=config['vocab_size'],
#             context_length=config['context_length'],
#             d_model=v['d_model'],
#             d_ff=v['d_ff'],
#             num_heads=v['num_heads'],
#             num_layers=v['num_layers'],
#             rope_theta=config['rope_theta']
#         ).to(device)
#         opt = AdamW(
#             params=model.parameters(),
#             lr=config['lr'],
#             betas=config['betas'],
#             eps=config['eps'],
#             weight_decay=config['weight_decay']
#         )
#         print(f'Model initialized. Start benchmark. Trial num: {trial_num}')
    
#         print('With_autocast benchmarking')
#         mean_time, time_per_step, std = benchmarking_train_autocast(
#             trial_num=trial_num,
#             warmup_num=warmup_num,
#             model=model,
#             opt=opt,
#             step_num=step_num,
#             forward_only=forward_only,
#             with_autocast=True
#         )
#         print(f'Warmup: {warmup_num} | Step: {step_num} | Forward_only: {forward_only} | Average train time(s): {mean_time:.4f} | Average time per step(s): {time_per_step:.4f} | Std: {std:.4f}')
        
#         print('Without_autocast benchmarking')
#         mean_time, time_per_step, std = benchmarking_train_autocast(
#             trial_num=trial_num,
#             warmup_num=warmup_num,
#             model=model,
#             opt=opt,
#             step_num=step_num,
#             forward_only=forward_only,
#             with_autocast=False
#         )
#         print(f'Warmup: {warmup_num} | Step: {step_num} | Forward_only: {forward_only} | Average train time(s): {mean_time:.4f} | Average time per step(s): {time_per_step:.4f} | Std: {std:.4f}')

from contextlib import nullcontext


# -------------------------
# Config
# -------------------------
config = {
    "vocab_size": 10000,
    "context_length": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
    "rope_theta": 10000,
}

MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


# -------------------------
# Single train step timing
# -------------------------
def benchmark_step(
    model,
    opt,
    x,
    y,
    use_autocast: bool,
    autocast_dtype=torch.bfloat16,
):
    ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if use_autocast
        else nullcontext()
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    with ctx:
        pred = model(x)
        loss = cross_entropy(
            rearrange(pred, "b l v -> (b l) v"),
            rearrange(y, "b l -> (b l)")
        )

    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


# -------------------------
# Benchmark loop
# -------------------------
def run_benchmark(
    model,
    opt,
    x,
    y,
    warmup_steps: int,
    measure_steps: int,
    use_autocast: bool,
):
    # warmup
    for _ in range(warmup_steps):
        benchmark_step(model, opt, x, y, use_autocast)

    times = []
    for _ in range(measure_steps):
        t = benchmark_step(model, opt, x, y, use_autocast)
        times.append(t)

    return sum(times) / len(times)

def benchmarking_mixed_precision():
    assert torch.cuda.is_available(), "This benchmark requires a GPU."

    device = torch.device("cuda")
    torch.manual_seed(0)

    B = config["batch_size"]
    L = config["context_length"]

    x = torch.randint(
        0, config["vocab_size"],
        (B, L),
        device=device
    )
    y = torch.randint(
        0, config["vocab_size"],
        (B, L),
        device=device
    )

    warmup_steps = 2
    measure_steps = 5

    print(f"Running BF16 mixed-precision benchmark on {device}\n")

    for name, spec in MODEL_SPECS.items():
        print(f"Model size: {name}")

        model = Model.BasicsTransformerLM(
            vocab_size=config["vocab_size"],
            context_length=L,
            d_model=spec["d_model"],
            d_ff=spec["d_ff"],
            num_heads=spec["num_heads"],
            num_layers=spec["num_layers"],
            rope_theta=config["rope_theta"],
        ).to(device)

        
        opt = AdamW(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )

        # full precision
        t_fp32 = run_benchmark(
            model, opt, x, y,
            warmup_steps, measure_steps,
            use_autocast=False
        )

        # BF16 mixed precision
        t_bf16 = run_benchmark(
            model, opt, x, y,
            warmup_steps, measure_steps,
            use_autocast=True
        )

        print(f"  FP32  avg step time: {t_fp32:.4f} s")
        print(f"  BF16  avg step time: {t_bf16:.4f} s")
        print(f"  Speedup: {t_fp32 / t_bf16:.2f}x\n")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    benchmarking_mixed_precision()
