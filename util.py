train_dir = "/data/ILSVRC/Data/CLS-LOC/train"
annotation_file = "/data/working/metadata_train.csv"  # Output CSV file name
class_index_file = "/data/working/class_map.csv"
val_dir = "/data/ILSVRC/Data/CLS-LOC/val"
val_annotation_file = "/data/working/metadata_val.csv"
val_mapping_file = "/data/LOC_val_solution.csv"


from torch.utils.data import Dataset
from PIL import Image

class CustomDataSetImagenet(Dataset):
    def __init__(self, annotation_file, transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')
import os
import time

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("Libraries imported - ready to use PyTorch", torch.__version__)

def show_image(image, label):
    print('plotting image')
    image = image.permute(1, 2, 0)
    plt.title(f'Label: {label}')
    # plt.imshow(image.squeeze())
    plt.imshow(image.squeeze())
    plt.show()

# device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# resume training options
resume_training = True


class Params:
    def __init__(self):
        self.batch_size = 464 #448 #384 #128 #192
        self.name = "resnet_152_sgd1"
        self.workers = 28
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


# training_folder_name = "/kaggle/input/imagenet-train-subset-100k/imagenet_subtrain"
# val_folder_name = "/kaggle/input/imagenet-train-subset-100k/imagenet_subtrain"

# training_folder_name = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
# val_folder_name = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"

params = Params()
def get_data_loaders():
    train_transformation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # train_dataset = torchvision.datasets.ImageFolder(
    #     root=training_folder_name,
    #     transform=train_transformation
    # )
    # annotation_file = "/kaggle/working/metadata.csv"
    train_dataset = CustomDataSetImagenet(
        annotation_file=annotation_file,
        transform=train_transformation
    )
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        # sampler=train_sampler,
        shuffle=True,
        num_workers=params.workers,
        pin_memory=True,
    )

    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=256, antialias=True),
        transforms.CenterCrop(224),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # val_dataset = torchvision.datasets.ImageFolder(
    #     root=val_folder_name,
    #     transform=val_transformation
    # )

    val_dataset = CustomDataSetImagenet(
        annotation_file=val_annotation_file,
        transform=val_transformation
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        num_workers=params.workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader


# train_loader, val_loader = get_data_loaders()
print("loader created")