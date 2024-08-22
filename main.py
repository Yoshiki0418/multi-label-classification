import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchmetrics import F1Score

from src.utils import set_seed
from src.datasets import ImageDataset
from src.model import VGG16

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir= hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="multi-label-classification")
    
    #------------------------------
    #        DataLoader
    #------------------------------
    # ディレクトリ構造に基づいてデータを準備する
    data_dir = "./multi-label-datasets"
    categories = ["cat", "dog", "dog_cat"]

    # データを格納するリスト
    file_paths = []
    labels = []

    # 各カテゴリに対してデータを収集
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        for file_name in os.listdir(category_dir):
            if file_name.endswith(".jpeg"):
                file_paths.append(os.path.join(category_dir, file_name))
                labels.append(category)

    # トレーニングセットと検証セットに分割 (例: 80% トレーニング, 20% 検証)
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 画像のサイズを調整 
        transforms.ToTensor(),  # 画像をテンソルに変換
    ])

    train_set = ImageDataset(train_files, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    val_set = ImageDataset(val_files, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False)

    #----------------------------
    #         model
    #----------------------------
    model = VGG16().to(args.device)

    #----------------------------
    #        Optimizer
    #----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #----------------------------
    #      Start traning
    #----------------------------
    # F1 Scoreの設定
    f1_score = F1Score(num_classes=train_set.num_classes, average='macro', mdmc_average='samplewise', task='multilabel').to(args.device)

    # 損失関数の定義
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)

            loss = criterion(y, y_pred)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 勾配クリッピング
            optimizer.step()

            # F1 Scoreの更新
            y_pred_sig = torch.sigmoid(y_pred) > 0.5
            f1_score.update(y_pred_sig, y.int())
        # F1 Scoreの計算
        train_f1 = f1_score.compute()
        f1_score.reset()
        print(f"Epoch {epoch+1}/{args.epochs} | train F1: {train_f1:.3f}")

        model.eval()
        for X, y in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            loss = criterion(y, y_pred)

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

if __name__ == "__main__":
    run()
