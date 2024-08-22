import torch
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms

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
        
        # ファイル名からラベルを決定
        if 'dog_cat' in img_path:
            label = torch.tensor([1, 1], dtype=torch.float32)
        elif 'dog' in img_path:
            label = torch.tensor([1, 0], dtype=torch.float32)
        elif 'cat' in img_path:
            label = torch.tensor([0, 1], dtype=torch.float32)
        else:
            label = torch.tensor([0, 0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

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

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 画像のサイズを調整 
    transforms.ToTensor(),  # 画像をテンソルに変換
])

train_set = ImageDataset(train_files, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
val_set = ImageDataset(val_files, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False)

for data, label in enumerate(train_loader):
    print(data)