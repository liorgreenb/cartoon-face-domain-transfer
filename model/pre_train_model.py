import itertools
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.utils import make_grid

from constants import MODE_TRAIN, MODE_VAL, MODE_TEST
from data.image_dataset_loader import ImageDataset


class PreTrainModel(pl.LightningModule):

    def __init__(self, hparams, device, g_ab, g_ba):
        super(PreTrainModel, self).__init__()

        self.hparams = hparams
        self.device = device

        self.learning_rate = hparams.learning_rate
        self.B1 = hparams.b1
        self.B2 = hparams.b1
        self.batch_size = hparams.batch_size

        self.g_ab = g_ab
        self.g_ba = g_ba

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()

    def forward(self, real_a, real_b):
        return self.g_ba(real_a), self.g_ab(real_b)

    def training_step(self, batch, batch_index):
        # Set model input
        real_a = batch["A"].to(self.device)
        real_b = batch["B"].to(self.device)

        # -------------------------------
        #  Pri Train Generators (g_ab, g_ba)
        # -------------------------------
        # Identity loss
        id_a, id_b = self.forward(real_a, real_b)
        loss_id_a = self.criterion_identity(id_a, real_a)
        loss_id_b = self.criterion_identity(id_b, real_b)

        loss = (loss_id_a + loss_id_b) / 2

        tqdm_dict = {"G_loss": loss}

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

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.g_ab.parameters(), self.g_ba.parameters()), lr=self.learning_rate,
            betas=(self.B1, self.B2)
        )
        return optimizer_G

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
        real_a = batch["A"].to(self.device)
        real_b = batch["B"].to(self.device)
        id_a, id_b = self.forward(real_a, real_b)

        # Arrange images along x-axis
        real_a = make_grid(real_a, nrow=5, normalize=True)
        real_b = make_grid(real_b, nrow=5, normalize=True)
        id_a = make_grid(id_a, nrow=5, normalize=True)
        id_b = make_grid(id_b, nrow=5, normalize=True)

        # Arrange images along y-axis
        image_grid = torch.cat((real_a, id_b, real_b, id_b), 1)
        self.logger.experiment.add_image(f'pre_train_sample_images_{self.current_epoch}', image_grid, 0)
