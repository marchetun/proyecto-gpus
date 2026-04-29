import torch
import triton
import triton.language as tl
#Sacada de -> https://openai.com/index/triton/
@triton.jit
def matmul_kernel(
    A, B, C, Bias, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
    ACTIVATION: tl.constexpr
):
    # IDs del programa
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    #Definición de bloques y punteros
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    #Empezamos a hacer la multiplicación por bloques
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)
        # Avanzamos punteros
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    # Suma de Bias y ReLU
    bias_ptr = Bias + rn
    bias = tl.load(bias_ptr, mask=rn < N)
    acc += bias[None, :]
    if ACTIVATION == "relu":
        acc = tl.maximum(0.0, acc)
    elif ACTIVATION == "none":
        pass # No hacemos nada

    # Guardado final
    C_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc, mask=mask)

def triton_matmul(x, w, b, activation = "none"):
    M, K = x.shape
    K_w, N = w.shape # Asumiendo que W ya está en forma [K, N]
    
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Definimos el grid 2D
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']), 
        triton.cdiv(N, meta['BLOCK_N'])
    )
    
    matmul_kernel[grid](
        x, w, y, b,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        ACTIVATION = activation
    )
    return y