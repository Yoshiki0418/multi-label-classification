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
    image_path = "multi-label-datasets/dog_cat/30.jpeg"

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
    # クラス名をリストで定義
    classes = ['犬', '猫']

    for images in tqdm(test_loader, desc="Test"):
        
        outputs = model(images.to(args.device))
        print(outputs)
        y = torch.sigmoid(outputs)
        print(y)
        
        # yの値が0.5以上かどうかをチェックし、対応するクラス名を表示
        predicted_labels = [classes[idx] for idx, value in enumerate(y[0]) if value >= 0.65]
        
        # 結果の表示
        if not predicted_labels:
            print("判定結果: どのクラスも検出されませんでした。")
        else:
            print("判定結果:", "と".join(predicted_labels), "が検出されました。")
            
            

      


if __name__ == "__main__":
    run()