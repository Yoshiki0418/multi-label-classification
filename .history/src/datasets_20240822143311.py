import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.num_classes = 2

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if 'regular' in img_path:
            label = 0
        else:
            label = 1

        if self.transform:
            image = self.transform(image)

        return image, label