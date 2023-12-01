import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from Dog_train import DogDataset
from utils import predict


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def infer(cfg: DictConfig):
    test_dir = Path(cfg.data.test_dir)
    model_dir = Path(cfg.training.model_dir)
    csv_dir = Path(cfg.csv.csv_dir)
    model = torch.load(model_dir)

    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    test_files = sorted(list(test_dir.rglob("*.jpeg")))
    test_dataset = DogDataset(test_files, mode="test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
    probs = predict(model, test_loader)

    preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
    test_filenames = [path.name for path in test_dataset.files]

    df = pd.DataFrame()
    df["Name"] = test_filenames
    df["Expected"] = preds
    df.to_csv(csv_dir, index=False)
    df.head()


if __name__ == "__main__":
    infer()
    print("csv created")
