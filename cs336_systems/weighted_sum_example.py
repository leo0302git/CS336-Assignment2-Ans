import triton
import triton.language as tl
import torch
from einops import rearrange
from typing import cast
from triton.runtime.jit import JITFunction




@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers
    output_ptr,         # Output pointer
    x_stride_row, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,           # Likely 1
    output_stride_row,           # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for _ in range(0, tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)

        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    
    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))

def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x2d = rearrange(x, "... d -> (...) d").contiguous() # 把任意前置维度变成一维，总共两维

        ctx.save_for_backward(x2d, weight) # save_for_backward 最多只能调用一次，可以在 setup_context() 或 forward() 方法中调用，并且只能使用 tensors。
        ctx.input_shape = input_shape

        assert weight.ndim == 1 and weight.shape[0] == D, "Dimension mismatch" # ndim 跟dim一样，都可以返回维度的数量
        assert x2d.is_cuda and weight.is_cuda # To check if the tensor is stored on CUDA
        assert x2d.is_contiguous()

        ROWS_TILE_SIZE = 16 # 每次(每个tile)读16行
        D_TILE_SIZE = triton.next_power_of_2(D) // 16 # 希望大约16次左右能循环完
        # 比如 D = 768, D_TILE_SIZE=1024//16=64, cdiv(768,64) = 12,也就是说，每tile读16*64的小块，循环12次能把整个tensor处理完
        ctx.ROWS_TILE_SIZE = ROWS_TILE_SIZE
        ctx.D_TILE_SIZE = D_TILE_SIZE

        n_rows = x2d.shape[0]
        y = torch.empty((n_rows,), device=x2d.device, dtype=x2d.dtype)

        grid = (triton.cdiv(n_rows, ROWS_TILE_SIZE),) # 决定了launch多少个program instance

        # Launch our kernel with n instances in our 1D grid.
        weighted_sum_fwd[grid]( 
            x2d, weight, y,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        return y.reshape(output_dims) # 为了写 kernel 简化，把任意形状 [..., D] 的输入 x 展平成了 (N, D)（N=前面维度之积），于是 kernel 输出自然是 (N,)。但从用户视角，输入是 [..., D]，输出应该是 [...]（也就是去掉最后一维的形状），这是weighted sum的数学定义要求的。所以要把 (N,) 再变回 output_dims = x.shape[:-1] （一个列表）
        # -1 表示自动推断该维度，例如 reshape(-1, D) 推断 N

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight = ctx.saved_tensors # 取出保存的张量
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x2d.shape # n_rows是x展平前除最后一维的其他所有维度的乘积

        # 注意，这里的grad_out是loss对y求偏导。因为y已经变成原始维度(D这一维已经被消掉):output_dims = x.shape[:-1](前面可能不止一维)，grad_out跟y是同维的。但是此时kernel是展平的视角，因为我们launch kernel时采用的grid = (triton.cdiv(n_rows, ROWS_TILE_SIZE),)(这是因为我们将x展平成2d: n_rows*D).
        grad_out_1d = grad_out.reshape(-1).contiguous() # '.reshape(-1)' means flatten into a 1D tensor. 这时候，由于y展平后的长度一定就是n_rows，所以grad_out关于forward grid的对齐就准备好了

        n_tiles = triton.cdiv(n_rows, ROWS_TILE_SIZE) # tile的个数
        partial_grad_weight = torch.empty((n_tiles, D), device=x2d.device, dtype=x2d.dtype) # Our strategy is for each thread block to first write to a partial buffer, then we reduce over this buffer to get the final gradient. 所以loss对 w (D，)的偏导理应是和w同维，即(D,)的，但是这里先不计算算完，而是传出一个中间计算结果
        grad_x = torch.empty_like(x2d)

        # launch kernels
        # 注意这里传入的，只有x2d, weight（原始输入数据，通过上下文ctx得到）以及grad_out(pytorch自动管理)是有真实数据的，其余partial_grad_weight以及grad_x都是生成的空张量，仅仅为了传指针进去
        grid = (n_tiles,) # 对齐前向传播的grid维度（铁律）
        weighted_sum_backward[grid](
            x2d, weight,
            grad_out_1d,
            grad_x, partial_grad_weight,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0),
            grad_out_1d.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        grad_weight = partial_grad_weight.sum(dim=0)
        grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, grad_weight

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,              # Input
    grad_output_ptr,                # Grad input
    grad_x_ptr, partial_grad_weight_ptr,  # Grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0) # 表示：在 grid 的第 0 维上，你是第几个 program（类似 CUDA 的 blockIdx.x）
    n_row_tiles = tl.num_programs(0)
    
    # 由于传入的是grad_out_1d，且是contiguous的，所以维度为(n_rows,),stride为1
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0), # 生成内存访问指令时，我们希望 GPU 先遍历第 1 维（列），再遍历第 0 维（行）因为列是连续内存
    ) 
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE), # 注意这里第一维是故意留着不消的，因为在作业文档中说：Our strategy is for each thread block to first write to a partial buffer, then we reduce over this buffer to get the final gradient.
        order=(1, 0),
    )
    # grad_output = tl.load(grad_output_block_ptr, mask=(row_tile_idx * ROWS_TILE_SIZE + tl.arange(0, ROWS_TILE_SIZE)) < NUM_ROWS)
    # 只要 tile 切分是用 cdiv 得到的 grid / loop 次数，就几乎总会遇到“最后一块不满”的情况，就需要 boundary_check。越界读则pad 0，越界写则不写
    for i in range(tl.cdiv(D, D_TILE_SIZE)): # 沿着D维串行扫描。理论上可进一步并行化（繁）
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)

        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,) 只取一小部分来算
        grad_x_row = grad_output[:, None] * weight[None, :] # 相当于做笛卡尔外积。为了索引广播，需要将两个向量变成维度数相同的，这里用None扩充了维度数，使乘法过程变成了(n_rows,1) .* (1, D), 这会广播为 (n_rows,D) .* (n_rows, D)。
        # （这段分析是基于整体而非tile的）grad_x_row_{i,j}是loss 关于x (n_rows, D)的导数的第i行第j列entry，由于grad_output (n_rows,) weight (D,),所以应该是上面的写法，而不应该是weight[:,None] * grad_output[None,:] (这样结果会变成 (D,n_rows)的)。
        # grad_output[:, None]实际上是1*1的，因为grad_output stride = 1
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1)) # 存入对应的一小部分

        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True) # (ROWS_TILE_SIZE, D_TILE_SIZE) .* (ROWS_TILE_SIZE,1), broadcast:  (ROWS_TILE_SIZE, D_TILE_SIZE) .* (ROWS_TILE_SIZE,D_TILE_SIZE), reduce along the first axis(keep_diw): (1, D_TILE_SIZE)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # Never out of bounds for dim 0，因为第0维上的stride已经设为1了

        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE)) # 第0维上的stride虽然为1，但是需要指定其不动
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


def main():
    torch.manual_seed(0)
    device = "cuda"

    N = 256
    D = 128

    x = torch.randn(N, D, device=device, requires_grad=True)
    w = torch.randn(D, device=device, requires_grad=True)

    # Forward
    y = cast(torch.Tensor, WeightedSumFunc.apply(x, w))

    # Simple scalar loss
    loss = y.sum()

    # Backward
    loss.backward()
    assert x.grad is not None
    assert w.grad is not None
    print("Output shape:", y.shape)
    print("Grad x shape:", x.grad.shape)
    print("Grad w shape:", w.grad.shape)
    print("All done.")

if __name__ == "__main__":
    main()
