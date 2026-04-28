import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(#El compilador lo expande a nivel de hilo
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)#pid equivale a grupo de hilos
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(0.0, x) # Usamos 0.0 para asegurar float32
    tl.store(y_ptr + offsets, output, mask=mask)

def triton_relu(x: torch.Tensor):
    # 1. Preparar el tensor de salida en la misma GPU que x
    y = torch.empty_like(x)
    n_elements = x.numel()
    
    # 2. Definir el Grid
    # Queremos suficientes programas para cubrir todos los elementos
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)#división hacia arriba
    
    # 3. Lanzar el kernel
    relu_kernel[grid](
        x, y, n_elements, 
        BLOCK_SIZE=1024
    )
    return y