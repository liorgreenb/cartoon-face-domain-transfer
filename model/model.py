import itertools
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data.image_dataset_loader import ImageDataset
from model.discriminator import Discriminator
from model.generator import GeneratorResNet


class Model(pl.LightningModule):

    def __init__(self, hyperparams):
        super(Model, self).__init__()

        self.hyperparams = hyperparams

        self.num_res_blocks = hyperparams.num_res_blocks
        self.input_shape = hyperparams.input_shape
        self.learning_rate = hyperparams.learning_rate
        self.B1 = hyperparams.B1
        self.B2 = hyperparams.B1
        self.n_epochs = hyperparams.n_epochs
        self.start_epoch = hyperparams.start_epoch
        self.epoch_decay = hyperparams.epoch_decay
        self.batch_size = hyperparams.batch_size
        self.lambda_cycle_loss = hyperparams.lambda_cycle_loss
        self.lambda_identity_loss = hyperparams.lambda_identity_loss

        self.G_AB = GeneratorResNet(self.input_shape, self.num_res_blocks)
        self.G_BA = GeneratorResNet(self.input_shape, self.num_res_blocks)
        self.D_A = Discriminator(self.input_shape)
        self.D_B = Discriminator(self.input_shape)

        # Adversarial ground truths
        self.valid = torch.ones((self.batch_size, *self.D_A.output_shape))
        self.fake = torch.zeros((self.batch_size, *self.D_A.output_shape))

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.fake_A = None
        self.fake_B = None
        self.recov_A = None
        self.recov_B = None

    def forward(self, real_A, real_B):
        return self.G_AB(real_A), self.G_BA(real_B)

    def training_step(self, batch, batch_index, optimizer_index=0):
        loss = None
        loss_type = None

        # Set model input
        real_A = batch["A"]
        real_B = batch["B"]

        # -------------------------------
        #  Train Generators (G_AB, G_BA)
        # -------------------------------
        if optimizer_index == 0:
            # Identity loss
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            self.fake_B, self.fake_A = self.forward(real_A, real_B)
            loss_GAN_AB = self.criterion_GAN(self.D_B(self.fake_B), self.valid)
            loss_GAN_BA = self.criterion_GAN(self.D_A(self.fake_A), self.valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            self.recov_B, self.recov_A = self.forward(self.fake_A, self.fake_B)

            # Cycle loss
            loss_cycle_A = self.criterion_cycle(self.recov_A, real_A)
            loss_cycle_B = self.criterion_cycle(self.recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss = loss_GAN + self.lambda_cycle_loss * loss_cycle + self.lambda_identity_loss * loss_identity

            loss_type = 'G'

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        elif optimizer_index == 1:

            # Real loss
            loss_real = self.criterion_GAN(self.D_A(real_A), self.valid)
            # Fake loss
            loss_fake = self.criterion_GAN(self.D_A(self.fake_A), self.fake)
            # Total loss
            loss = (loss_real + loss_fake) / 2
            loss_type = "D_A"

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        elif optimizer_index == 2:
            # Real loss
            loss_real = self.criterion_GAN(self.D_B(real_B), self.valid)
            # Fake loss
            loss_fake = self.criterion_GAN(self.D_B(self.fake_B), self.fake)
            # Total loss
            loss = (loss_real + loss_fake) / 2
            loss_type = "D_B"

        tqdm_dict = {f"{loss_type}_loss": loss}

        return {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, batch, batch_nb):
        loss_data = self.training_step(batch, batch_nb)

        return {
            'val_loss': loss_data['loss'],
            'progress_bar': loss_data['progress_bar'],
            'log': loss_data['log']
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}
    #
    # def test_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.learning_rate,
            betas=(self.B1, self.B2)
        )
        optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.learning_rate, betas=(self.B1, self.B2))
        optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.learning_rate, betas=(self.B1, self.B2))

        # Learning rate update schedulers
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(self.n_epochs, self.start_epoch, self.epoch_decay).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(self.n_epochs, self.start_epoch, self.epoch_decay).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(self.n_epochs, self.start_epoch, self.epoch_decay).step
        )

        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]

    @pl.data_loader
    def train_dataloader(self):
        return self.create_data_loader()

    @pl.data_loader
    def val_dataloader(self):
        return self.create_data_loader()

    @pl.data_loader
    def test_dataloader(self):
        return self.create_data_loader()

    def create_data_loader(self):
        return DataLoader(
            ImageDataset(),
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=multiprocessing.cpu_count(),
        )
