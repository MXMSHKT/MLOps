import os
from typing import Any

import lightning.pytorch as pl
import omegaconf
import torch
import transformers
import torchvision

class DogModel(pl.LightningModule):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.backbone = transformers.AutoModel.from_pretrained(
            conf["model"]["name"],
            return_dict=True,
            output_hidden_states=True,
        )
        
        self.fc = torch.nn.Linear(2048, conf["data"]["n_classes"])
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        return self.backbone(data)
    


    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        data, target = batch
        y_pred = self(data)

        loss = self.loss_fn(y_pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        
        predictions = torch.argmax(y_pred, dim=1)
        correct = torch.sum(predictions == target).float() 
        accuracy = correct / data.shape[0]
        
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy}
    
    
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        data, target = batch
        y_pred = self(data)

        loss = self.loss_fn(y_pred, target)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        predictions = torch.argmax(y_pred, dim=1)
        correct = torch.sum(predictions == target).float() 
        accuracy = correct / data.shape[0]
        
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_accuracy": accuracy}
    

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass


    def configure_optimizers(self) -> Any:
        
        param_optimizer = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': self.conf["train"]["weight_decay"],
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}]


        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.conf["train"]["learning_rate"]
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.conf["train"]["step_size"], 
            gamma=self.conf["train"]["gamma"])
        
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

        
    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
