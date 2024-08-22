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
from torchmetrics import Precision, Recall
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
import onnxruntime as ort

from .src.utils import set_seed
from .src.datasets import 

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

    #----------------------------
    #         model
    #----------------------------
    model = 

    #----------------------------
    #        Optimizer
    #----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #----------------------------
    #      Start traning
    #----------------------------