import torch
import triton
import triton.language as tl

def check_env():
    print("--- Verificación de Entorno ---")
    print(f"Versión de PyTorch: {torch.__version__}")
    
    print(f"Versión de Triton: {triton.__version__}")
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Nombre de GPU: {torch.cuda.get_device_name(0)}")
    print("-----------------------------")
if __name__ == "__main__":
    check_env()