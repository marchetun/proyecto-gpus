import torch
from .TritonLinear import TritonLinear

# Definir la arquitectura usando Triton
class MNISTModelTriton(torch.nn.Module):
    def __init__(self):
        super(MNISTModelTriton, self).__init__()
        # 784 entradas -> 128 neuronas (oculta) -> 10 salidas (dígitos)
        self.fc1 = TritonLinear(784, 128, activation = "relu")
        self.fc2 = TritonLinear(128, 10, activation = "none")

    def forward(self, x):
        # Aplanamos la imagen de 28x28 a 784
        x = x.view(-1, 28 * 28)
        x = self.fc1(x) # Aquí ya se aplica Matmul + Bias + ReLU (funde tu kernel)
        x = self.fc2(x) # Segunda capa
        return x

