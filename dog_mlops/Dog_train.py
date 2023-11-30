import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from utils import DEVICE, predict, train


# Constants
DATA_MODES = ["train", "val", "test"]
RESCALE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 5
MODEL_DIR = Path("dog_model")

# Create train and validation datasets
TRAIN_DIR = Path("data/train")


# Define a DogDataset class
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


def set_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Check GPU availability
    set_seed(64)

    train_val_files = sorted(list(TRAIN_DIR.rglob("*.jpeg")))

    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, stratify=train_val_labels
    )

    train_dataset = DogDataset(train_files, mode="train")
    val_dataset = DogDataset(val_files, mode="val")

    n_classes = len(np.unique(train_val_labels))

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.classifier = nn.Linear(in_features=1792, out_features=n_classes)

    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)

    model = model.to(DEVICE)

    # history =
    train(
        train_dataset,
        val_dataset,
        model,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    torch.save(model, MODEL_DIR)

    idxs = list(map(int, np.random.uniform(0, 1000, 20)))
    imgs = [val_dataset[id][0].unsqueeze(0) for id in idxs]

    probs_ims = predict(model, imgs)

    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    y_pred = np.argmax(probs_ims, -1)

    actual_labels = [val_dataset[id][1] for id in idxs]
    img_actual_labels = [
        val_dataset.label_encoder.inverse_transform([al])[0]
        for al in actual_labels
    ]

    preds_class = [label_encoder.classes_[i] for i in y_pred]

    print(
        "f1_score -",
        f1_score(img_actual_labels, preds_class, average="weighted"),
    )


if __name__ == "__main__":
    main()
    print("Training complete.")
