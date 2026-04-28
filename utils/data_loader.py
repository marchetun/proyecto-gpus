import torch
import numpy as np
from torch.utils.data import Dataset

class MNISTNumpyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        # Cargamos los archivos .npy
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Si la imagen está aplanada (784), la volvemos a (28, 28) 
        # para que las transformaciones estándar funcionen
        if image.shape[0] == 784:
            image = image.reshape(28, 28)

        # Convertir a float32 y normalizar manualmente si no hay transform
        image = image.astype(np.float32) / 255.0
        
        # Convertir a Tensor de PyTorch [1, 28, 28]
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label