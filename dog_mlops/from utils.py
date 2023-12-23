import os
import subprocess
import random
from pathlib import Path
from dvc.repo import Repo
import pickle
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from torch.utils.data import Dataset
from dataclass import DogDataset
from torchvision import datasets, transforms
from utils import load_data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm



train_val_files = sorted(list(Path("/home/max/MLOps/data/train").rglob("*.jpeg")))
print(f"Найдено файлов: {len(train_val_files)}")



train_val_labels = [path.parent.name for path in train_val_files]
            
train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

train_dataset = DogDataset(train_files, mode="train")

val_dataset = DogDataset(val_files, mode="val")


print(f"Length of dataset: {len(train_dataset)}")

for i in range(5):
    data, label = train_dataset[i]
    print(f"Data sample {i}: {data}, Label: {label}")

import matplotlib.pyplot as plt


def show_image(image, label):
    plt.imshow(image.permute(1, 2, 0))  # Переставляем каналы для корректного отображения
    plt.title(label)
    plt.show()




dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for batch, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch}:")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    show_image(data, labels)
    # Визуализируем данные из батча, если это изображения
    break  # Убрать break для проверки нескольких батчей
