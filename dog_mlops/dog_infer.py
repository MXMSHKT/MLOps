from pathlib import Path

# import numpy as np
# import pandas as pd
import pytorch_lightning as pl

from dog_mlops.dataclass import DogDataModule
from dog_mlops.model import DogModel


def infer(cfg):
    csv_dir = Path(cfg.csv.csv_dir)
    pl.seed_everything(64)
    model = DogModel(cfg)

    model = DogModel.load_from_checkpoint(checkpoint_path="dog_model.ckpt")
    model.eval()
    # label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    dm = DogDataModule(cfg)
    dm.setup(stage="predict")
    test_loader = dm.test_dataloader()
    # test_dataset = dm.test_dataset

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
    )

    predicts = trainer.test(model, test_loader)
    # print(len(test_loader))
    # print(len(test_dataset))
    # print(test_dataset[0])  # Inspect the first sample
    print(predicts)
    print(csv_dir)

    # preds = label_encoder.inverse_transform(np.argmax(predicts, axis=1))
    # test_filenames = [path.name for path in test_dataset.files]

    # print(type(predicts))
    print("csv created")
