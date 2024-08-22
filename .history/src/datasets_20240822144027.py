import torch
from PIL import Image
import torch.utils.data

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
            # 犬でも猫でもない場合のデフォルトラベルを設定
            label = torch.tensor([0, 0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
