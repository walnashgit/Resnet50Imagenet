import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR
import time
from torch_lr_finder import LRFinder


# Define the PyTorch Lightning module
class ResNet50LightningModule(pl.LightningModule):
    def __init__(self, lr_dataloader, lr_finder=True, num_classes=1000):
        super(ResNet50LightningModule, self).__init__()
        self.with_lr_finder = lr_finder
        self.lr_dataloader = lr_dataloader
        self.save_hyperparameters()
        self.model = models.resnet50(pretrained=False, num_classes=num_classes)
        # self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.epoch_start_time = None

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        # prediction = output.argmax(dim=1)
        # acc = self.accuracy(prediction, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", acc*100, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        epoch_duration = time.time() - self.epoch_start_time
        self.log('train_epoch_time', epoch_duration, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # Run validation only every 10 epochs and on the last epoch
        # if (self.current_epoch + 1) % 10 == 0 or (self.current_epoch + 1) == self.trainer.max_epochs:
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target).item()
        # prediction = output.argmax(dim=1)
        # acc = self.accuracy(prediction, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", acc*100, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            self.log('val_epoch_time', epoch_duration, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.with_lr_finder:
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2)
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.learning_rate, momentum=0.9, weight_decay=1e-4)
            self.find_lr(optimizer)
        else:
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2)
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.learning_rate, momentum=0.9, weight_decay=1e-4)
            self.max_lr = self.hparams.learning_rate

        print("max_lr used: ", self.max_lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            # steps_per_epoch=len(self.lr_dataloader),
            steps_per_epoch=len(self.train_dataloader()),
            # self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            epochs=self.trainer.max_epochs,
            # pct_start=0.3,
            # anneal_strategy='linear',
            div_factor=10.0,
            final_div_factor=100.0,
            three_phase=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            },
        }

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def find_lr(self, optimizer):
        lr_finder = LRFinder(self, optimizer, criterion=self.criterion)
        lr_finder.range_test(self.lr_dataloader, end_lr=10, num_iter=500)
        self.max_lr = lr_finder.plot()[-1]  # Plot the loss vs learning rate
        print(f"Suggested learning rate: {self.max_lr}")
        lr_finder.reset()
