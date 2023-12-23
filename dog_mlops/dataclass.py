import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from torch.utils.data import Dataset

from torchvision import datasets, transforms
from utils import load_data

# Constants
DATA_MODES = ["train", "val", "test"]
RESCALE_SIZE = 224


# Define a DogDataset 


class DogDataset(Dataset):
    def __init__(self, files, mode):
        self.files = sorted(files)
        self.mode = mode
        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()

        if self.mode not in DATA_MODES:
            raise NameError(
                f"{self.mode} is not correct; correct modes: {DATA_MODES}"
            )

        if self.mode != "test":
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open("label_encoder.pkl", "wb") as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype="float32")

        if self.mode == "train":
            x = train_transform(x)
        else:
            x = val_transform(x)

        if self.mode == "test":
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
            

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


class DogDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataloader_num_workers, val_size):
        super().__init__()
        self.save_hyperparameters()
        self.val_size = val_size
        self.dataloader_num_workers = dataloader_num_workers
        self.batch_size = batch_size
    
    def prepare_data(self):
        self.TRAIN_DIR, self.TEST_DIR = load_data()


    
    def setup(self, stage=None):
        print("setaup")
        if stage == "fit":
            print("setaup__train")
            print(self.TRAIN_DIR,self.TEST_DIR)
            train_val_files = sorted(list(Path("/home/max/MLOps/data/train").rglob("*.jpeg")))
            print(f"Найдено файлов: {len(train_val_files)}")

            if len(train_val_files) == 0:
                raise RuntimeError("Не найдено файлов для обработки. Проверьте путь к данным.")

            train_val_labels = [path.parent.name for path in train_val_files]
            
            train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

            print(len(train_files))
            self.train_dataset = DogDataset(train_files, mode="train")
            print(len(self.train_dataset))
            self.val_dataset = DogDataset(val_files, mode="val")
            

        if stage == "test" or stage == "predict":
            print("setaup__teest")
            test_files = sorted(list(Path(self.TEST_DIR).rglob("*.jpeg")))
            self.test_dataset = DogDataset(test_files, mode="test")


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        print("train")
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.dataloader_num_workers, shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        print("val")
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.dataloader_num_workers, shuffle=True
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        print("test")
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=True
        )

