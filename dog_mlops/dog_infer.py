from pathlib import Path

import hydra
import pytorch_lightning as pl

from dog_mlops.dataclass import DogDataModule, DogModel


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def infer(cfg):
    # Getting best checkpoint name
    best_model_name = (
        Path(cfg.artifacts.checkpoint.dirpath) / cfg.loggers.experiment_name / "best.txt"
    )

    with open(best_model_name, "r") as f:
        best_checkpoint_name = f.readline()

    # Getting a torch-model
    model = DogModel.load_from_checkpoint(best_checkpoint_name)

    dm = DogDataModule(cfg)
    dm.setup(stage="predict")
    test_loader = dm.test_dataloader()

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
    )

    trainer.test(model, test_loader)


if __name__ == "__main__":
    infer()
    print("csv created")
