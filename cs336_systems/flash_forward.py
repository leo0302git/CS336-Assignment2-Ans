import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
import torch
from torch import Tensor
from einops import rearrange
from typing import cast
from jaxtyping import Bool

import math

# class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Q, K, V, is_causal=False):
#         assert isinstance(Q, torch.Tensor)
#         assert isinstance(K, torch.Tensor)
#         assert isinstance(V, torch.Tensor)
#         #print(f'Q: {Q.shape}, K: {K.shape}, V: {V.shape},')
#         dtype = Q.dtype
#         device = Q.device
#         #print('device: ', Q.device)
#         B_dims, Nq, d = Q.shape[:-2], Q.shape[-2], Q.shape[-1]
#         Nk = K.shape[-2]
#         Bq, Bk = 8, 16
#         Tq, Tk = triton.cdiv(Nq, Bq), triton.cdiv(Nk, Bk)
#         O = torch.zeros((*B_dims,Nq,d),dtype=dtype,device=device)
#         L = torch.zeros((*B_dims,Nq),dtype=dtype,device=device)
#         for i in range(Tq):
#             Qi_upper_bound = (i+1) * Bq if (i+1) * Bq < Nq else Nq
#             Qi = Q[...,i * Bq : Qi_upper_bound,:] # 取(...,Bq,d)
#             # print(f'Qi: {Qi.shape} 非边界情况下应该是(...,Bq,d)') 
#             Oi = O[...,i * Bq : Qi_upper_bound,:]
#             actual_Bq = Qi.shape[-2]
#             li = torch.zeros((*B_dims,actual_Bq,),dtype=dtype, device=device)
#             mi = torch.full((*B_dims,actual_Bq,),-torch.inf,dtype=dtype, device=device)
            
#             for j in range(Tk):
#                 Kj_upper_bound = (j+1) * Bk if (j+1) * Bk < Nk else Nk
#                 Kj = K[...,j * Bk: Kj_upper_bound, :] # (...,Bk,d)
#                 Vj = V[...,j * Bk: Kj_upper_bound, :] # (...,Bk,d)
#                 Kj_T = rearrange(Kj, '... Bk d -> ... d Bk')
#                 Sij = Qi @ (Kj_T) / math.sqrt(d) # (...,Bq,Bk)
#                 Sij_rowmax = torch.max(Sij,dim=-1,keepdim=False).values # (...,Bq,)
#                 # assert isinstance(Sij_rowmax, Tensor), f'Sij_rowmax not a Tensor. type: {type(Sij_rowmax)}'
#                 # assert not torch.isnan(Sij_rowmax).any().item(), 'Sij_rowmax has Nan'
#                 mi_new = torch.maximum(mi,Sij_rowmax) # (...,Bq,)
#                 Pij = torch.exp(Sij - mi_new[...,None]) # 将mi扩充为 (...,Bq,Bk)，以便广播, 得到 (...,Bq,Bk)
#                 Pij_rowsum = torch.sum(Pij,dim=-1) # (...,Bq)
#                 li = torch.mul((mi-mi_new).exp(), li) + Pij_rowsum # (...,Bq), broadcast li
#                 # assert not torch.isnan(li).any().item(), 'li has Nan'
#                 mi_diaged = torch.diag_embed((mi-mi_new).exp()) # (...,Bq,Bq)
#                 # assert not torch.isnan(mi_diaged).any().item(), 'mi_diag has Nan'
#                 # assert not torch.isnan(Pij).any().item(), 'Pij has Nan'
#                 # assert not torch.isnan(Vj).any().item(), 'Vj has Nan'
#                 # assert not torch.isnan(Oi).any().item(), f'In i:{i}, j:{j} iteration, before computing, Oi has Nan.'
#                 #print(' before computing: ', Oi)
#                 #print(f'Kj_T: {Kj_T.shape}, Vj: {Vj.shape}, Sij: {Sij.shape}, Sij_rowmax: {Sij_rowmax.shape}, mi: {mi.shape}, mi_new: {mi_new.shape}, Pij: {Pij.shape}, Pij_rowsum: {Pij_rowsum.shape}, li: {li.shape}, mi_diaged: {mi_diaged.shape}, Oi: {Oi.shape}')
#                 #print('mi_diaged: ', mi_diaged, 'Pij: ', Pij, 'Vj: ', Vj)
#                 Oi = mi_diaged @ Oi + Pij @ Vj  # (...,Bq,d)
#                # print(' after computing: ', Oi)
#                 # assert not torch.isnan(Oi).any().item(), f'In i:{i}, j:{j} iteration, Oi has Nan.'
#                 #if i == 0 and j == 3:
#                     #print('Qi: ', Qi)
#                     #print('Kj: ', Kj)
#                     #print('Oi: ', Oi)
#                     #print(f'Kj_T: {Kj_T.shape}, Vj: {Vj.shape}, Sij: {Sij.shape}, Sij_rowmax: {Sij_rowmax.shape}, mi: {mi.shape}, mi_new: {mi_new.shape}, Pij: {Pij.shape}, Pij_rowsum: {Pij_rowsum.shape}, li: {li.shape}, mi_diaged: {mi_diaged.shape}, Oi: {Oi.shape}')
#                 mi = mi_new
#             Oi = torch.diag_embed(li).inverse() @ Oi
#             O[...,i * Bq : Qi_upper_bound,:] = Oi
#             #print('outside Oi: ', Oi)
#             L[...,i * Bq: Qi_upper_bound] = mi + li.log() # (...,Bq)
#         ctx.save_for_backward(L) 
#         # print('O: ', O)
#         # print('L: ', L)
#         return O
#     @staticmethod
#     def backward(ctx):
#         raise NotImplementedError
# def flash_bwd_pytorch(Q, K, V, O, dO, L):
#     assert isinstance(Q, Tensor)
#     assert isinstance(K, Tensor)
#     assert isinstance(V, Tensor)
#     assert isinstance(O, Tensor)
#     assert isinstance(L, Tensor)
#     d = Q.shape[-1]
#     scale = math.sqrt(d)
#     D = torch.sum(O * dO, dim=-1, keepdim=False) # (...,Nq)
#     K_T = rearrange(K, '... Nk d -> ... d Nk')
#     S = Q @ (K_T) / scale
#     P = (S - L[...,None]).exp()
#     P_T = rearrange(P, '... Nq Nk -> ... Nk Nq')
#     dV = P_T @ dO
#     V_T = rearrange(V, '... Nk d -> ... d Nk')
#     dP = dO @ V_T
#     dS = P * (dP - D[...,None])
#     dQ = dS @ K / scale
#     dS_T = rearrange(dS, '... Nq Nk -> ... Nk Nq')
#     dK = dS_T @ Q / scale
#     return dQ, dK, dV, None # 因为forward里面传入了is_causal，所以这里也需要传对应的梯度，不过设成None即可

def flash_bwd_pytorch(Q, K, V, O, dO, L):
    # 统一用 fp32 做 backward（避免 bf16/fp32 混用 + 更稳定）
    orig_dtype = Q.dtype
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    Of = O.float()
    dOf = dO.float()
    Lf = L.float()  # 本来就是 fp32，但写上更明确

    d = Qf.shape[-1]
    scale = math.sqrt(d)

    D = torch.sum(Of * dOf, dim=-1, keepdim=False)
    K_T = rearrange(Kf, '... Nk d -> ... d Nk')
    S = Qf @ K_T / scale
    P = (S - Lf[..., None]).exp()
    P_T = rearrange(P, '... Nq Nk -> ... Nk Nq')

    dV = P_T @ dOf
    V_T = rearrange(Vf, '... Nk d -> ... d Nk')
    dP = dOf @ V_T
    dS = P * (dP - D[..., None])
    dQ = dS @ Kf / scale
    dS_T = rearrange(dS, '... Nq Nk -> ... Nk Nq')
    dK = dS_T @ Qf / scale

    # cast 回输入 dtype（和 autograd 期望一致）
    return dQ.to(orig_dtype), dK.to(orig_dtype), dV.to(orig_dtype), None

_flash_backward_compiled = None
class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert isinstance(Q, torch.Tensor)
        assert isinstance(K, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        dtype = Q.dtype
        device = Q.device
        B_dims, Nq, d = Q.shape[:-2], Q.shape[-2], Q.shape[-1]
        Nk = K.shape[-2]
        Bq, Bk = 8, 16
        Tq, Tk = math.ceil(Nq / Bq), math.ceil(Nk / Bk)
        # Tq, Tk = triton.cdiv(Nq, Bq), triton.cdiv(Nk, Bk)
        O = torch.zeros((*B_dims,Nq,d),dtype=dtype,device=device)
        L = torch.zeros((*B_dims,Nq),dtype=dtype,device=device)
        for i in range(Tq):
            Qi_upper_bound = (i+1) * Bq if (i+1) * Bq < Nq else Nq
            Qi = Q[...,i * Bq : Qi_upper_bound,:] # 取(...,Bq,d)
            Oi = O[...,i * Bq : Qi_upper_bound,:]
            actual_Bq = Qi.shape[-2]
            li = torch.zeros((*B_dims,actual_Bq,),dtype=dtype, device=device)
            mi = torch.full((*B_dims,actual_Bq,),-torch.inf,dtype=dtype, device=device)
            
            for j in range(Tk):
                Kj_upper_bound = (j+1) * Bk if (j+1) * Bk < Nk else Nk
                Kj = K[...,j * Bk: Kj_upper_bound, :] # (...,Bk,d)
                Vj = V[...,j * Bk: Kj_upper_bound, :] # (...,Bk,d)
                Kj_T = rearrange(Kj, '... Bk d -> ... d Bk')
                Sij = Qi @ (Kj_T) / math.sqrt(d) # (...,Bq,Bk)
                Sij_rowmax = torch.max(Sij,dim=-1,keepdim=False).values # (...,Bq,)
                mi_new = torch.maximum(mi,Sij_rowmax) # (...,Bq,)
                Pij = torch.exp(Sij - mi_new[...,None]) # 将mi扩充为 (...,Bq,Bk)，以便广播, 得到 (...,Bq,Bk)
                Pij_rowsum = torch.sum(Pij,dim=-1) # (...,Bq)
                li = torch.mul((mi-mi_new).exp(), li) + Pij_rowsum # (...,Bq), broadcast li
                mi_diaged = torch.diag_embed((mi-mi_new).exp()) # (...,Bq,Bq)
                Oi = mi_diaged @ Oi + Pij @ Vj  # (...,Bq,d)
                mi = mi_new
            Oi = torch.diag_embed(li).inverse() @ Oi
            # Oi = 1 / torch.diag_embed(li) @ Oi
            O[...,i * Bq : Qi_upper_bound,:] = Oi
            L[...,i * Bq: Qi_upper_bound] = mi + li.log() # (...,Bq)
        ctx.save_for_backward(Q, K, V, O, L) 
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        global _flash_backward_compiled # 设成全局的，只编译一次
        if _flash_backward_compiled is None:
            _flash_backward_compiled = torch.compile(flash_bwd_pytorch,backend='aot_eager') # 设置aot_eager可以在CPU情景下仅测试test_flash_backward_pytorch，而不测试test_flash_backward_triton
        return _flash_backward_compiled(Q, K, V, O, dO, L)

   
class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        Nk = K.shape[-2]

        O = torch.empty_like(Q)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # grid的第一维是token tile的数量，第二维是Batch
        grid = (
            triton.cdiv(Nq, Q_TILE_SIZE),
            B,
        )

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal = is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        global _flash_backward_compiled # 设成全局的，只编译一次
        if _flash_backward_compiled is None:
            _flash_backward_compiled = torch.compile(flash_bwd_pytorch,backend='aot_eager') # 设置aot_eager可以在CPU情景下仅测试test_flash_backward_pytorch，而不测试test_flash_backward_triton
        return _flash_backward_compiled(Q, K, V, O, dO, L)
        
    


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd, # 如果是contiguous的，则stride_qb = N*D stride_qq = D, stride_qd = 1
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False
):
    # ------------------------------------------------------------
    # Program indices
    # ------------------------------------------------------------
    query_tile_index = tl.program_id(0)   # which query block
    batch_index = tl.program_id(1)        # which batch

    # ------------------------------------------------------------
    # Construct block pointers
    # ------------------------------------------------------------
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D), # 这定义了Q_block_ptr所组成的张量的大小。描述的是：在当前 batch 内，Q 这块“逻辑矩阵”的完整大小。Triton 需要知道全局 shape，才能做boundary check
        strides=(stride_qq, stride_qd), # 定义了整块的逻辑矩阵之间的stride
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 指明当前 program（负责某一个batch中的��一个query block） 要从全局矩阵的哪个“坐标”开始取 tile。
        block_shape=(Q_TILE_SIZE, D), # 一个 program instance 在一次访存/计算中操作的数据块大小。
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_start = query_tile_index * Q_TILE_SIZE # 在这个kernel instance里不变了
    # ------------------------------------------------------------
    # Load Q tile
    # ------------------------------------------------------------
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q = Q * tl.full((), scale, Q.dtype)

    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        S = tl.dot(Q, tl.trans(K), acc=tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), tl.float32))  # 或者不用 acc，用：S = tl.dot(Q, tl.trans(K)).to(tl.float32)
        if is_causal:
            k_start = key_tile_index * K_TILE_SIZE
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)            # (Bq,)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)            # (Bk,)

            # causal mask: disallow attending to future keys
            causal = k_idx[None, :] > q_idx[:, None]               # (Bq, Bk)

            # out-of-bounds mask 超出边界的统一置为False
            q_valid = q_idx < N_QUERIES
            k_valid = k_idx < N_KEYS
            valid = q_valid[:, None] & k_valid[None, :]            # (Bq, Bk)

            mask = causal | (~valid) # valid是False的地方表示越界，取反之后为True，表示要被掩盖的部分

            S = tl.where(mask, -1e6, S) # mask为True的地方用-1e6代替，False的地方用S原本的值

        m_ij = tl.maximum(m_i, tl.max(S, axis=1))
        P = tl.exp(S - m_ij[:, None])            # fp32
        l_ij = tl.sum(P, axis=1)                 # fp32
        alpha = tl.exp(m_i - m_ij)               # fp32

        l_i = l_i * alpha + l_ij

        P = P.to(V.dtype)                        # cast P to V dtype
        PV = tl.dot(P, V).to(tl.float32)         # accumulate in fp32
        O_i = O_i * alpha[:, None] + PV

        m_i = m_ij
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    O_out = O_i.to(O_block_ptr.type.element_ty)  # cast before store
    tl.store(O_block_ptr, O_out, boundary_check=(0, 1))
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))



# def flash_attention_forward(Q, K, V, is_causal = False):
#     """
#     Q, K, V: (B, N, D)
#     """
#     B, Nq, D = Q.shape
#     Nk = K.shape[-2]

#     O = torch.empty_like(Q)
#     L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

#     Q_TILE_SIZE = 16
#     K_TILE_SIZE = 16

#     grid = (
#         triton.cdiv(Nq, Q_TILE_SIZE),
#         B,
#     )

#     flash_fwd_kernel[grid](
#         Q, K, V,
#         O, L,
#         Q.stride(0), Q.stride(1), Q.stride(2),
#         K.stride(0), K.stride(1), K.stride(2),
#         V.stride(0), V.stride(1), V.stride(2),
#         O.stride(0), O.stride(1), O.stride(2),
#         L.stride(0), L.stride(1),
#         Nq, Nk,
#         1.0 / math.sqrt(D),
#         D=D,
#         Q_TILE_SIZE=Q_TILE_SIZE,
#         K_TILE_SIZE=K_TILE_SIZE,
#         is_causal = is_causal
#     )
#     return O, L

# def reference_attention(Q, K, V):
#     """
#     Standard PyTorch attention for comparison
#     """
#     d = Q.shape[-1]
#     S = Q @ K.transpose(-1, -2) / math.sqrt(d)
#     P = torch.softmax(S, dim=-1)
#     O = P @ V
#     L = torch.logsumexp(S, dim=-1)
#     return O, L


def test_flash_attention():
    torch.manual_seed(0)
    device = "cuda"

    B = 2
    N = 32
    D = 64

    Q = torch.randn(B, N, D, device=device, dtype=torch.float16)
    K = torch.randn(B, N, D, device=device, dtype=torch.float16)
    V = torch.randn(B, N, D, device=device, dtype=torch.float16)

    # FlashAttention
    O_flash, L_flash = flash_attention_forward(Q, K, V)

    # Reference
    O_ref, L_ref = reference_attention(Q, K, V)

    # Check
    print("Output shape:", O_flash.shape)
    print("Max abs diff O:", (O_flash - O_ref).abs().max().item())
    print("Max abs diff L:", (L_flash - L_ref).abs().max().item())

    assert torch.allclose(O_flash, O_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(L_flash, L_ref, atol=1e-2, rtol=1e-2)

    print("✅ FlashAttention forward kernel correct!")


if __name__ == "__main__":
    test_flash_attention()
