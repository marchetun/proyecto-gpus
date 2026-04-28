import torch
import triton
import triton.language as tl

def check_env():
    print("--- Verificación de Entorno ---")
    print(f"Versión de PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        # Prueba simple de tensor en GPU
        x = torch.tensor([1.0, 2.0]).to("cuda")
        print("Operación básica en GPU: Exitosa")
    else:
        print("ERROR: No se detectó GPU. Triton requiere una GPU NVIDIA.")

    try:
        print(f"Versión de Triton: {triton.__version__}")
    except:
        print("ERROR: Triton no está correctamente instalado.")

if __name__ == "__main__":
    check_env()