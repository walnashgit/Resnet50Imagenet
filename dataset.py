#### DATASET


import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch

from config import CONFIG

scale = 1.1
IMAGE_SIZE = 224


class ImageNetDataset:
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.dataset = []
        self.train = train

        annot_file = True
        if CONFIG.get("data_annotation_file", {}):
            if train and CONFIG["data_annotation_file"]["train"]:
                df = pd.read_csv(CONFIG["data_annotation_file"]["train"])
                self.dataset = [(row[0], row[1]) for _, row in df.iterrows()]
            elif not train and CONFIG["data_annotation_file"]["val"]:
                df = pd.read_csv(CONFIG["data_annotation_file"]["val"])
                self.dataset = [(row[0], row[1]) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_dir = CONFIG["root_dir"] + "/ILSVRC/Data/CLS-LOC/%s/" % ("train" if self.train else "val")
        image, label = self.dataset[idx]
        image = Image.open(image_dir + image).convert("RGB")
        # image = np.array(image)  # Convert PIL Image to NumPy array
        if self.transform:
            image = self.transform(image)  # Albumentations transform
        # label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        return image, label


train_transformation = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=256, antialias=True),
    transforms.CenterCrop(224),
    # Normalize the pixel values (in R, G, and B channels)
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = CONFIG["batch_size"]
        self.num_workers = CONFIG["num_workers"]
        self.augment_prob = CONFIG["augment_prob"]

    def setup(self, stage: str = None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = ImageNetDataset(
                train=True,
                transform=train_transformation
            )
            self.val_dataset = ImageNetDataset(
                train=False,
                transform=val_transformation  # No augmentations for validation
            )
        if stage == "test":
            self.test_dataset = ImageNetDataset(
                train=False,
                transform=val_transformation  # No augmentations for testing
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=CONFIG["pin_memory"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=CONFIG["pin_memory"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=CONFIG["pin_memory"],
        )