import cv2
import torch
import numpy as np
from models.MNISTModelTriton import MNISTModelTriton

def webcam_inference():
    device = torch.device("cuda")
    
    # Inicializar modelo y cargar pesos
    model = MNISTModelTriton().to(device)
    checkpoint = torch.load('mnist_model_weights.pth')

    with torch.no_grad():
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
        if not ret: break

        # Definimos la regionde interes
        h, w, _ = frame.shape
        box_size = 300  # Tamaño del cuadrado central
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Dibujar la caja azul en la pantalla para saber donde apuntar
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Recortar solo lo que está dentro de la caja
        roi = frame[y1:y2, x1:x2]

        # Preprocesamiento Extremo (Para igualar a MNIST)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Desenfocar ligeramente para quitar el ruido del papel
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Convierte a Blanco y Negro puro, e invierte los colores a la vez
        # el otro hace que el papel blanco sea negro y la tinta negra sea blanca
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Redimensionar al tamaño final de MNIST
        resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

        # Mostramos la imagen de 28x28 ampliada para ver que pilla el kernel
        cv2.imshow('Ojos de la IA (28x28 Ampliado)', cv2.resize(resized, (200, 200)))

        # Preparar el Tensor
        images = torch.from_numpy(resized).to(device).to(torch.float32)
        images = (images / 255.0).view(-1, 784).contiguous()

        # Inferencia con Triton
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            prediction = predicted.item()

        # Mostrar resultado en la pantalla principal
        cv2.putText(frame, f"Prediccion: {prediction}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow('Triton MNIST Real-Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_inference()