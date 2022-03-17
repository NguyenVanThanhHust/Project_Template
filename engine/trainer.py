import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

class LitUnet(pl.LightningModule):
    def __init__(self, model, loss, optim):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim = optim

    def forward(self, x):
        mask = self.model(x)
        return mask

    def training_step(self, batch, batch_idx):
        data, target = batch 
        pred = self.model(data)
        loss = self.loss(pred, target)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch 
        pred = self.model(data)
        loss = self.loss(pred, target)
        self.log('valid_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, target = batch 
        pred = self.model(data)
        loss = self.loss(pred, target)
        self.log('test_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        return self.optim


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
    ):

    unet = LitUnet(model, loss_fn, optimizer, )
    
    # ------------
    # training
    # ------------
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    trainer.fit(unet, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=val_loader)
    print(result)