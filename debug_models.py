import torch
from models.MNISTModelTriton import MNISTModelTriton
from models.pytorch_mlp import VanillaMLP
from utils.data_loader import MNISTNumpyDataset

def debug_models():
    device = torch.device("cuda")
    
    # Cargar un batch pequeño para debugging
    test_ds = MNISTNumpyDataset(
        images_path='data/test_images.npy', 
        labels_path='data/test_labels.npy'
    )
    
    # Tomar solo 1 imagen
    images, label = test_ds[0]
    images = images.unsqueeze(0).to(device).to(torch.float32).view(1, 784)
    label = torch.tensor([label]).to(device)
    
    print(f"Imagen shape: {images.shape}, min: {images.min():.4f}, max: {images.max():.4f}")
    print(f"Label: {label.item()}\n")
    
    # Modelo PyTorch
    pytorch_model = VanillaMLP().to(device)
    checkpoint = torch.load('mnist_model_weights.pth')
    pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    
    # Modelo Triton
    triton_model = MNISTModelTriton().to(device)
    with torch.no_grad():
        triton_model.fc1.weight.copy_(checkpoint['fc1.weight'].t().contiguous())
        triton_model.fc1.bias.copy_(checkpoint['fc1.bias'].contiguous())
        triton_model.fc2.weight.copy_(checkpoint['fc2.weight'].t().contiguous())
        triton_model.fc2.bias.copy_(checkpoint['fc2.bias'].contiguous())
    triton_model.eval()
    
    # Forward pass
    with torch.no_grad():
        # PyTorch
        pytorch_fc1 = pytorch_model.fc1(images)
        pytorch_relu = pytorch_model.relu(pytorch_fc1)
        pytorch_output = pytorch_model.fc2(pytorch_relu)
        pytorch_pred = torch.argmax(pytorch_output, dim=1).item()
        
        # Triton
        triton_fc1 = triton_model.fc1(images)
        triton_fc2 = triton_model.fc2(triton_fc1)
        triton_pred = torch.argmax(triton_fc2, dim=1).item()
        
        print(f"PyTorch FC1 shape: {pytorch_fc1.shape}, min: {pytorch_fc1.min():.4f}, max: {pytorch_fc1.max():.4f}")
        print(f"PyTorch ReLU shape: {pytorch_relu.shape}, min: {pytorch_relu.min():.4f}, max: {pytorch_relu.max():.4f}")
        print(f"PyTorch Output: {pytorch_output}")
        print(f"PyTorch Prediction: {pytorch_pred}\n")
        
        print(f"Triton FC1 shape: {triton_fc1.shape}, min: {triton_fc1.min():.4f}, max: {triton_fc1.max():.4f}")
        print(f"Triton FC2 shape: {triton_fc2.shape}")
        print(f"Triton Output: {triton_fc2}")
        print(f"Triton Prediction: {triton_pred}\n")
        
        # Comparar pesos
        print(f"FC1 Weight norm PyTorch: {pytorch_model.fc1.weight.norm():.4f}")
        print(f"FC1 Weight norm Triton: {triton_model.fc1.weight.norm():.4f}")
        print(f"FC1 Weight diff: {(pytorch_model.fc1.weight - triton_model.fc1.weight.t()).norm():.4e}\n")
        
        print(f"FC1 Bias norm PyTorch: {pytorch_model.fc1.bias.norm():.4f}")
        print(f"FC1 Bias norm Triton: {triton_model.fc1.bias.norm():.4f}")
        print(f"FC1 Bias diff: {(pytorch_model.fc1.bias - triton_model.fc1.bias).norm():.4e}\n")
        
        # Diferencia en FC1 computación
        diff_fc1 = (pytorch_fc1 - triton_fc1).abs().max()
        print(f"Diferencia máxima FC1: {diff_fc1:.6e}")
        
        diff_relu = (pytorch_relu - triton_fc1).abs().max()
        print(f"Diferencia: ReLU PyTorch vs FC1 Triton (después de ReLU aplicado): {diff_relu:.6e}\n")
        
        # Diferencia en output
        diff_output = (pytorch_output - triton_fc2).abs().max()
        print(f"Diferencia máxima Output: {diff_output:.6e}")

if __name__ == "__main__":
    debug_models()
