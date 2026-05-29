import torch
import torch.nn as nn
from kernels.matmulMod import triton_matmul 
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation = "relu"):
        super(TritonLinear, self).__init__()
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicializamos pesos y bias como parámetros normales de PyTorch
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return triton_matmul(x, self.weight, self.bias, activation=self.activation)