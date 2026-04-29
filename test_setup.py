import torch
import triton
import triton.language as tl

def check_env():
    print("--- Verificación de Entorno ---")
    print(f"Versión de PyTorch: {torch.__version__}")
    
    # Tomamos una imagen de prueba (ej. un '7')
    

    # Pasamos la imagen por tu modelo Triton
    prediccion = modelo_custom(imagen_test)

# Obtenemos el número con mayor probabilidad
clase_predicha = torch.argmax(prediccion, dim=1)
print(f"El modelo Triton dice que el número es: {clase_predicha.item()}")

if __name__ == "__main__":
    check_env()