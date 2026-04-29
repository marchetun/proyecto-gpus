import torch
import numpy as np
import time
from MNISTModelTriton import MNISTModelTriton
from models.pytorch_mlp import VanillaMLP

def run_test():
    device = 'cuda'
    
    # 1. Cargar la imagen .npy
    try:
        # Cargamos una imagen de prueba (ej. la primera del dataset de test)
        raw_data = np.load('data/test_images.npy') 
        sample_img = torch.from_numpy(raw_data[0]).float().to(device)
        sample_img = sample_img.view(1, 28 * 28) / 255.0 # Normalizar y aplanar
    except:
        print("No se encontró data/test_images.npy, usando ruido aleatorio para el test.")
        sample_img = torch.randn(1, 784, device=device)

    # 2. Inicializar Modelos
    # Modelo Triton (con tus kernels fusionados)
    model_triton = MNISTModelTriton().to(device).eval()
    
    # Modelo Vanilla (para comparar resultados y tiempos)
    model_vanilla = VanillaMLP().to(device).eval()
    # Cargamos los mismos pesos al vanilla para que la comparación sea justa
    model_vanilla.load_state_dict(torch.load('mnist_model_weights.pth'))

    # 3. Validar Precisión (Parity Check)
    with torch.no_grad():
        out_vanilla = model_vanilla(sample_img)
        out_triton = model_triton(sample_img)
        
        diff = torch.abs(out_vanilla - out_triton).max()
        print(f"--- Validación de Resultados ---")
        print(f"Diferencia máxima entre PyTorch y Triton: {diff:.6e}")
        
        pred = torch.argmax(out_triton, dim=1).item()
        print(f"Predicción del modelo Triton: {pred}")

    # 4. Benchmark de Velocidad
    # Calentamiento (Warm-up) para que la GPU alcance su velocidad máxima
    for _ in range(100):
        model_vanilla(sample_img)
        model_triton(sample_img)

    # Medir PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        model_vanilla(sample_img)
    torch.cuda.synchronize()
    time_vanilla = (time.time() - start)

    # Medir Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        model_triton(sample_img)
    torch.cuda.synchronize()
    time_triton = (time.time() - start)

    print(f"\n--- Benchmark (1000 ejecuciones) ---")
    print(f"Tiempo PyTorch: {time_vanilla:.4f}s")
    print(f"Tiempo Triton (Fusionado): {time_triton:.4f}s")
    print(f"Mejora de velocidad: {((time_vanilla/time_triton)-1)*100:.2f}%")

if __name__ == "__main__":
    run_test()