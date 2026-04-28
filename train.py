import torch
import torch.nn as nn
import torch.optim as optim
from models.pytorch_mlp import VanillaMLP
from utils.data_loader import MNISTNumpyDataset
from torch.utils.data import DataLoader

def train_model():
    # 1. Configuración de hardware
    # Triton requiere CUDA, así que forzamos la comprobación
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("ADVERTENCIA: No se detectó GPU. Triton no funcionará más adelante.")
    print(f"Ejecutando en: {device}")

    # 2. Hiperparámetros
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-3

    # 3. Carga de datos (.npy de Kaggle)
    # Asumiendo que bajaste los archivos a una carpeta llamada 'data'
    try:
        train_ds = MNISTNumpyDataset(
            images_path='data/train_images.npy', 
            labels_path='data/train_labels.npy'
        )
        test_ds = MNISTNumpyDataset(
            images_path='data/test_images.npy', 
            labels_path='data/test_labels.npy'
        )
    except FileNotFoundError:
        print("Error: No se encontraron los archivos .npy en la carpeta /data")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 4. Inicializar modelo, pérdida y optimizador
    model = VanillaMLP().to(device)
    criterion = nn.CrossEntropyLoss() # Maneja internamente el Softmax
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Bucle Principal de Entrenamiento
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Mover tensores a la memoria de la GPU
            images, labels = images.to(device), labels.to(device)

            # Limpiar gradientes
            optimizer.zero_grad()
            
            # Forward pass: Aquí es donde más tarde entrará Triton
            outputs = model(images)
            
            # Calcular cuánto se equivocó la red
            loss = criterion(outputs, labels)
            
            # Backward pass: Cálculo de gradientes
            loss.backward()
            
            # Actualizar los pesos (Matriz W y Vector b)
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validar al final de cada época
        accuracy = validate(model, test_loader, device)
        print(f"Epoch {epoch+1} Finalizada - Loss Promedio: {total_loss/len(train_loader):.4f} - Accuracy: {accuracy:.2f}%")

    # Guardar los pesos para usarlos después con Triton
    torch.save(model.state_dict(), "mnist_model_weights.pth")
    print("Pesos guardados en mnist_model_weights.pth")

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    train_model()