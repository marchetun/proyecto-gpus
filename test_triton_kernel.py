import torch
import triton
import triton.language as tl

@triton.jit
def simple_matmul_test(
    A, B, C, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Versión simplificada sin GROUP_M para debugging
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

    C_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc, mask=mask)

def test_matmul():
    device = torch.device("cuda")
    
    # Test case: 32x32 Y 32x32 = 32x32 (grandes suficientes para Triton, creo)
    torch.manual_seed(42)
    A = torch.randn(32, 32, device=device, dtype=torch.float32)
    B = torch.randn(32, 32, device=device, dtype=torch.float32)
    
    # PyTorch
    C_pytorch = A @ B
    print(f"PyTorch result shape: {C_pytorch.shape}")
    print(f"PyTorch[0,0]: {C_pytorch[0,0]:.6f}\n")
    
    # Triton
    C_triton = torch.empty((32, 32), device=device, dtype=torch.float32)
    
    M, K = A.shape
    K_b, N = B.shape
    
    grid = (1, 1)
    
    simple_matmul_test[grid](
        A, B, C_triton,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_triton.stride(0), C_triton.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    
    print(f"Triton result shape: {C_triton.shape}")
    print(f"Triton[0,0]: {C_triton[0,0]:.6f}")
    
    diff = (C_pytorch - C_triton).abs()
    max_diff = diff.max().item()
    print(f"\nMax Difference: {max_diff:.6e}")
    print(f"Mean Difference: {diff.mean().item():.6e}\n")

if __name__ == "__main__":
    test_matmul()
