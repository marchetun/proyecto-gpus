import cv2
import torch
import numpy as np
from models.MNISTModelTriton import MNISTModelTriton

def webcam_inference():
    device = torch.device("cuda")
    
    # 1. Inicializar modelo y cargar pesos (Exactamente como en tu validate.py)
    model = MNISTModelTriton().to(device)
    checkpoint = torch.load('mnist_model_weights.pth')

    with torch.no_grad():
        # .t() para la forma [In, Out] y .contiguous() para el layout de Triton[cite: 5]
        model.fc1.weight.copy_(checkpoint['fc1.weight'].t().contiguous())
        model.fc1.bias.copy_(checkpoint['fc1.bias'].contiguous())
        model.fc2.weight.copy_(checkpoint['fc2.weight'].t().contiguous())
        model.fc2.bias.copy_(checkpoint['fc2.bias'].contiguous())
    
    model.eval()

    # 2. Configurar captura de video
    cap = cv2.VideoCapture(0)
    print("Iniciando webcam... Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Preprocesamiento directo (Siguiendo tu lógica de validate.py)
        # Convertimos a gris y escalamos a 28x28
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Invertimos colores: MNIST es fondo negro (0) y trazo blanco (1-255)
        # Si no inviertes, el modelo no reconocerá números negros sobre papel blanco
        gray = cv2.bitwise_not(gray) 
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Convertir a tensor, mover a GPU y cambiar a Float32
        # IMPORTANTE: No normalizamos aquí (/ 255.0) según tu instrucción
        images = torch.from_numpy(resized).to(device).to(torch.float32)
        
        # Aplanar y asegurar contigüidad (Layout para el kernel de Triton)
        images = images.view(-1, 784).contiguous()

        # 4. Inferencia con kernels fusionados de Triton[cite: 3, 5]
        with torch.no_grad():
            # Opcional: imprimir para debug como en tu script
            # print(f"Shape: {images.shape}, Dtype: {images.dtype}, Contiguous: {images.is_contiguous()}")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            prediction = predicted.item()

        # 5. Visualización
        cv2.putText(frame, f"Prediccion: {prediction}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow('Triton MNIST Real-Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_inference()