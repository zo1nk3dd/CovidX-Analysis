import pytorch_lightning as pl
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

CLASS_LABELS = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_SIZES = [3616, 10192, 1345]

class CovidXDataset(Dataset):
    def __init__(self):
        self.img_dir = 'D:\Datasets\COVIDX\Data'
        self.class_metadata = {class_idx: pd.read_excel(f'{self.img_dir}\{label}.metadata.xlsx') for class_idx, label in enumerate(CLASS_LABELS)}
        self.transform = transforms.ConvertImageDtype(torch.float32)

    def __len__(self):
        return sum(CLASS_SIZES)

    def __getitem__(self, idx):
        class_idx, image_idx = self.location_from_idx(idx)
        img_path = f"{self.img_dir}\{CLASS_LABELS[class_idx]}\{CLASS_LABELS[class_idx]}-{image_idx}.png"
        image = read_image(img_path, ImageReadMode.GRAY)
        image = self.transform(image)
        # Unused domain information
        # domain = self.class_metadata[class_idx].iloc(image_idx, 3)
        return image, class_idx
    
    def location_from_idx(self, idx: int):
        class_idx = 0
        image_idx = idx
        while class_idx < len(CLASS_LABELS):
            if image_idx < CLASS_SIZES[class_idx]:
                return class_idx, image_idx + 1
            else:
                image_idx -= CLASS_SIZES[class_idx]
                class_idx += 1
        return class_idx, image_idx + 1
        


class CovidXDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        data = CovidXDataset()
        self.train, self.val, self.test = random_split(data, [0.6, 0.2, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


        


