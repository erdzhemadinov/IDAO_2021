import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

device = "cpu"

class Print(nn.Module):
    """Debugging only"""

    def forward(self, x):
        print(x.size())
        return x


class Clamp(nn.Module):
    """Clamp energy output"""

    def forward(self, x):

        x = torch.clamp(x, min=0, max=30)
        return x


class SimpleConv(pl.LightningModule):
    def __init__(self, mode: ["classification", "regression"] = "classification"):
        super().__init__()
        self.mode = mode

        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 32, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),

                    nn.Conv2d(32, 32, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),

                    nn.Conv2d(32, 64, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),


                    nn.Flatten(),
                )

        self.drop_out = nn.Dropout(0.15)

        self.fc1 = nn.Linear(1664, 500)
        self.fc2 = nn.Linear(500, 2)  # for classification
        self.fc3 = nn.Linear(500, 1)  # for regression


        self.stem = nn.Sequential(
            self.layer1, self.drop_out, self.fc1,
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem, self.fc3)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def training_epoch_end(self, outs):
        # log epoch metric
        if self.mode == "classification":
            self.log("train_acc_epoch", self.train_acc.compute())
        else:
            pass

    def validation_step(self, batch, batch_idx):
        x_target, class_target, reg_target, _ = batch
        #print("here")
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())

            reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)#ReduceLROnPlateau(optimizer, ...)
        if self.mode == "classification":
            return {
               'optimizer': optimizer,
               'lr_scheduler': scheduler,
               'monitor': 'classification_loss'
           }
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'regression_loss'
            }


    def forward(self, x):
        if self.mode == "classification":
            class_pred = self.classification(x.float().to("cpu"))
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x.float().to("cpu"))
            return {"energy": reg_pred}
