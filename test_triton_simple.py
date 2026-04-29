import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_simple(
    A, B, C, Bias, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
    ACTIVATION: tl.constexpr
):
    # Sin GROUP_M - versión simple
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    # Suma de Bias
    bias_ptr = Bias + rn
    bias = tl.load(bias_ptr, mask=rn < N)
    acc += bias[None, :]
    
    if ACTIVATION == "relu":
        acc = tl.maximum(0.0, acc)

    # Guardado final
    C_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc, mask=mask)

def triton_matmul_simple(x, w, b, activation="none"):
    M, K = x.shape
    K_w, N = w.shape
    
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']), 
        triton.cdiv(N, meta['BLOCK_N'])
    )
    
    matmul_kernel_simple[grid](
        x, w, y, b,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        ACTIVATION=activation
    )
    return y

def test_matmul_with_bias_activation():
    device = torch.device("cuda")
    
    torch.manual_seed(42)
    x = torch.randn(1, 784, device=device, dtype=torch.float32)
    w = torch.randn(784, 128, device=device, dtype=torch.float32)
    b = torch.randn(128, device=device, dtype=torch.float32)
    
    # PyTorch
    linear = torch.nn.Linear(784, 128, device=device, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(w.t())
        linear.bias.copy_(b)
    
    pytorch_result = linear(x)
    pytorch_relu = torch.nn.functional.relu(pytorch_result)
    
    print(f"PyTorch output (before ReLU): min={pytorch_result.min():.6f}, max={pytorch_result.max():.6f}")
    print(f"PyTorch output (after ReLU): min={pytorch_relu.min():.6f}, max={pytorch_relu.max():.6f}\n")
    
    # Triton simple
    triton_result = triton_matmul_simple(x, w, b, activation="relu")
    
    print(f"Triton SIMPLE output: min={triton_result.min():.6f}, max={triton_result.max():.6f}\n")
    
    # Comparar
    diff = (pytorch_relu - triton_result).abs()
    print(f"Max difference: {diff.max():.6e}")
    print(f"Mean difference: {diff.mean():.6e}")

if __name__ == "__main__":
    test_matmul_with_bias_activation()
