import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights, resnet18

from DataClass import DogDataset
from utils import DEVICE, predict, set_seed, train


# Create train and validation datasets
# TRAIN_DIR = Path("data/train")


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Check GPU availability
    set_seed(64)

    train_dir = Path(cfg.data.train_dir)

    train_val_files = sorted(list(train_dir.rglob("*.jpeg")))

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
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
    )

    model_dir = Path(cfg.training.model_dir)
    torch.save(model, model_dir)

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
