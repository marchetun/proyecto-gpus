import torch
import torch.nn as nn

class VanillaMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(VanillaMLP, self).__init__()
        # Definición de capas (inicialización de pesos y sesgos)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x llega como [batch_size, 1, 28, 28] desde el DataLoader
        # Aplanamos la imagen: de (28,28) a (784,)
        x = x.view(x.size(0), -1) 
        
        # Primera capa: x * W1 + b1
        x = self.fc1(x)
        
        # Activación: max(0, x)
        x = self.relu(x)
        
        # Segunda capa: x * W2 + b2
        x = self.fc2(x)
        
        return x