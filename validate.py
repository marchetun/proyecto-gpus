import torch
from torch.utils.data import DataLoader
from models.MNISTModelTriton import MNISTModelTriton
from utils.data_loader import MNISTNumpyDataset # Usamos el que ya tienes

def validate_all_images():
    device = torch.device("cuda")
    
    # Cargar el dataset completo de test
    test_ds = MNISTNumpyDataset(
        images_path='data/test_images.npy', 
        labels_path='data/test_labels.npy'
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

   # Inicializamos el modelo
    model = MNISTModelTriton().to(device)
    checkpoint = torch.load('mnist_model_weights.pth')

    with torch.no_grad():
        model.fc1.weight.copy_(checkpoint['fc1.weight'].t().contiguous())
        model.fc1.bias.copy_(checkpoint['fc1.bias'].contiguous())
        model.fc2.weight.copy_(checkpoint['fc2.weight'].t().contiguous())
        model.fc2.bias.copy_(checkpoint['fc2.bias'].contiguous())
    
    model.eval()

    # Bucle de Inferencia sobre todo el dataset
    correct = 0
    total = 0
    print(f"Iniciando inferencia en {len(test_ds)} imágenes...")

    with torch.no_grad():
        
        for images, labels in test_loader:
            
            labels = labels.to(device)
            # Los datos ya vienen normalizados del data_loader (entre 0 y 1)
            images = images.to(device).to(torch.float32).view(-1, 784).contiguous()
            
            print(f"Shape: {images.shape}, Dtype: {images.dtype}, Contiguous: {images.is_contiguous()}")
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n--- Resultado Final ---")
    print(f"Imágenes procesadas: {total}")
    print(f"Precisión (Accuracy): {accuracy:.2f}%")

if __name__ == "__main__":
    validate_all_images()