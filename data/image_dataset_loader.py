import multiprocessing

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader

from constants import DOMAIN_A_READY_FILES, DOMAIN_B_READY_FILES, get_domain_files_by_mode, DOMAIN_A_READY, \
    DOMAIN_A_EXT, DOMAIN_B_READY, DOMAIN_B_EXT

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


class ImageDataset(Dataset):
    def __init__(self, mode):
        self.transform = transforms.Compose(transforms_)
        self.files_a = glob(get_domain_files_by_mode(DOMAIN_A_READY, mode, DOMAIN_A_EXT))
        self.files_b = glob(get_domain_files_by_mode(DOMAIN_B_READY, mode, DOMAIN_B_EXT))

    def __getitem__(self, index):
        image_a = Image.open(self.files_a[index % len(self.files_a)])
        image_b = Image.open(self.files_b[index % len(self.files_b)])
        item_a = self.transform(image_a)
        item_b = self.transform(image_b)

        return {"A": item_a, "B": item_b}

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))