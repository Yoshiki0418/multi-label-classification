import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchmetrics
import pandas as pd

from src.utils import set_seed
from src.datasets import ImageDataset, ImageDataset_test
from src.model import *



@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------  
    image_path = "multi-label-datasets/dog/1.jpeg"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 画像のサイズを調整 
        transforms.ToTensor(),  # 画像をテンソルに変換
    ])

    test_set = ImageDataset_test(image_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=1, num_workers=1
    )

    # ------------------
    #       Model
    # ------------------
    model = VGG16().to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    for images in tqdm(test_loader, desc="Test"):
        outputs = model(images.to(args.device))
        y = torch.sigmoid(outputs)
        print(y)
        

    


if __name__ == "__main__":
    run()