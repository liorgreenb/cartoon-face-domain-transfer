import multiprocessing
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image
from tqdm import tqdm

from constants import MODEL_INPUT_SIZE


def preprocess_image(image_path, domain_folder, domain_ready_folder, pbar):
    img = Image.open(image_path)
    processed_image = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    processed_image = processed_image.convert('RGB')
    processed_image.save(image_path.replace(domain_folder, domain_ready_folder))
    pbar.update()


def preprocess_dataset(dataset_glob, domain_folder, domain_ready_folder, dataset_size):
    dataset = glob(dataset_glob)[:dataset_size]
    with tqdm(total=len(dataset)) as pbar:
        pool = ThreadPool(multiprocessing.cpu_count())
        pool.map(lambda image_path: preprocess_image(image_path, domain_folder, domain_ready_folder, pbar), dataset)