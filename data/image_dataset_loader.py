import multiprocessing

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader

from constants import DOMAIN_A_READY_FILES, DOMAIN_B_READY_FILES

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


class ImageDataset(Dataset):
    def __init__(self, mode=None):
        self.transform = transforms.Compose(transforms_)
        self.files_a = glob(DOMAIN_A_READY_FILES)
        self.files_b = glob(DOMAIN_B_READY_FILES)

    def __getitem__(self, index):
        image_a = Image.open(self.files_a[index % len(self.files_a)])
        image_b = Image.open(self.files_b[index % len(self.files_b)])
        item_a = self.transform(image_a)
        item_b = self.transform(image_b)

        return {"A": item_a, "B": item_b}

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))