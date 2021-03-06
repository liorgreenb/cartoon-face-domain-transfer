import itertools
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.utils import make_grid

from constants import MODE_TRAIN, MODE_VAL, MODE_TEST
from data.image_dataset_loader import ImageDataset
from model.discriminator import Discriminator
from model.generator import GeneratorResNet

class LambdaLRSteper:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class Model(pl.LightningModule):

    def __init__(self, hparams, device, G_AB, G_BA):
        super(Model, self).__init__()

        self.hparams = hparams
        self.device = device

        self.input_shape = hparams.input_shape
        self.learning_rate = hparams.learning_rate
        self.B1 = hparams.b1
        self.B2 = hparams.b1
        self.n_epochs = hparams.n_epochs
        self.start_epoch = hparams.start_epoch
        self.epoch_decay = hparams.epoch_decay
        self.batch_size = hparams.batch_size
        self.lambda_cycle_loss = hparams.lambda_cycle_loss
        self.lambda_identity_loss = hparams.lambda_identity_loss

        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = Discriminator(self.input_shape)
        self.D_B = Discriminator(self.input_shape)

        # Adversarial ground truths
        self.valid = torch.ones((self.batch_size, *self.D_A.output_shape)).to(device)
        self.fake = torch.zeros((self.batch_size, *self.D_A.output_shape)).to(device)

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
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)

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
            loss_fake = self.criterion_GAN(self.D_A(self.fake_A.detach()), self.fake)
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
            loss_fake = self.criterion_GAN(self.D_B(self.fake_B.detach()), self.fake)
            # Total loss
            loss = (loss_real + loss_fake) / 2
            loss_type = "D_B"

        tqdm_dict = {f"{loss_type}_loss": loss}

        return OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

    def validation_step(self, batch, batch_nb):
        if batch_nb == 0:
            self.sample_network_images(batch)

        loss_data = self.training_step(batch, batch_nb)

        return {
            'val_loss': loss_data['loss'],
            'progress_bar': loss_data['progress_bar'],
            'log': loss_data['log']
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

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
            optimizer_G, lr_lambda=LambdaLRSteper(self.n_epochs, self.start_epoch, self.epoch_decay).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLRSteper(self.n_epochs, self.start_epoch, self.epoch_decay).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLRSteper(self.n_epochs, self.start_epoch, self.epoch_decay).step
        )

        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]

    @pl.data_loader
    def train_dataloader(self):
        return self.create_data_loader(MODE_TRAIN)

    @pl.data_loader
    def val_dataloader(self):
        return self.create_data_loader(MODE_VAL)

    @pl.data_loader
    def test_dataloader(self):
        return self.create_data_loader(MODE_TEST)

    def create_data_loader(self, mode):
        return DataLoader(
            ImageDataset(mode),
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=multiprocessing.cpu_count(),
        )

    def sample_network_images(self, batch):
        """Saves a generated sample from the test set"""
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)
        fake_B, fake_A = self.forward(real_A, real_B)

        # Arrange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)

        # Arrange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        self.logger.experiment.add_image(f'sample_images_{self.current_epoch}', image_grid, 0)
