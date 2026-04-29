import torch
from torch.utils.data import DataLoader
from models.pytorch_mlp import VanillaMLP
from utils.data_loader import MNISTNumpyDataset

def validate_pytorch():
    device = torch.device("cuda")
    
    # Cargar dataset
    test_ds = MNISTNumpyDataset(
        images_path='data/test_images.npy', 
        labels_path='data/test_labels.npy'
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Cargar modelo
    model = VanillaMLP().to(device)
    checkpoint = torch.load('mnist_model_weights.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    # Validar
    correct = 0
    total = 0
    print(f"Validando modelo PyTorch en {len(test_ds)} imágenes...")

    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.to(device).to(torch.float32).view(-1, 784)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n--- Resultado Final (PyTorch) ---")
    print(f"Imágenes procesadas: {total}")
    print(f"Precisión (Accuracy): {accuracy:.2f}%")

if __name__ == "__main__":
    validate_pytorch()
