import pytorch_lightning as pl

from dog_mlops.dataclass import DogDataModule
from dog_mlops.model import DogModel


def infer(cfg):
    model = DogModel.load_from_checkpoint(checkpoint_path=cfg.model.save_model_name)

    dm = DogDataModule(cfg)
    dm.setup(stage="predict")
    test_loader = dm.test_dataloader()

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
    )

    trainer.predict(model, test_loader)
