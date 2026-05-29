import torch
from kernels.matmulMod import triton_matmul

def test_matmul_with_bias_activation():
    device = torch.device("cuda")
    
    # Setup similar a MNIST: 1x784 @ 784x128 = 1x128 + bias (128,)
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
    
    # Triton
    triton_result = triton_matmul(x, w, b, activation="relu")
    
    print(f"Triton output: min={triton_result.min():.6f}, max={triton_result.max():.6f}\n")
    
    # Comparar
    diff = (pytorch_relu - triton_result).abs()
    print(f"Max difference: {diff.max():.6e}")
    print(f"Mean difference: {diff.mean():.6e}")
    
    # Verificar pesos en PyTorch
    print(f"\nPyTorch weight shape: {linear.weight.shape}")
    print(f"Input weight to Triton shape: {w.shape}")
    
    # Verificar bias
    print(f"Bias min: {b.min():.6f}, max: {b.max():.6f}")
    print(f"PyTorch bias min: {linear.bias.min():.6f}, max: {linear.bias.max():.6f}")

if __name__ == "__main__":
    test_matmul_with_bias_activation()
