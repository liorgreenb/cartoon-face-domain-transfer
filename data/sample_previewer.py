from glob import glob
import matplotlib.pyplot as plt
import torch
from PIL import Image
import random

from torchvision.utils import make_grid, save_image


def preview_sample_images(glob_path):
    cartoon_set = glob(glob_path)
    preview_samples = random.sample(cartoon_set, 10)

    fig = plt.figure(figsize=(20, 5))
    columns = 5
    rows = 2
    for (sample_index, sample_path) in enumerate(preview_samples):
        img = Image.open(sample_path)
        fig.add_subplot(rows, columns, sample_index + 1)
        plt.imshow(img)
    plt.show()


def sample_network_images(batches_done, imgs, G_AB, G_BA):
    """Saves a generated sample from the test set"""
    G_AB.eval()
    G_BA.eval()
    real_A = torch.tensor(imgs["A"])
    fake_B = G_AB(real_A)
    real_B = torch.tensor(imgs["B"])
    fake_A = G_BA(real_B)

    # Arrange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    # Arrange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"images/{batches_done}.png", normalize=False)
