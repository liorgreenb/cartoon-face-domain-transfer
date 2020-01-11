import multiprocessing
from copy import copy
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

from PIL import Image
from tqdm import tqdm

from constants import MODEL_INPUT_SIZE, DOMAIN_A_FILES, DOMAIN_B_FILES, DOMAIN_A_FOLDER, DOMAIN_A_READY, \
    DOMAIN_B_READY, DOMAIN_B_FOLDER


def preprocess(model_input_size):
    dataset_a = glob(DOMAIN_A_FILES)
    dataset_b = glob(DOMAIN_B_FILES)
    dataset_size = min(len(dataset_a), len(dataset_b))
    preprocess_dataset(dataset_a, DOMAIN_A_FOLDER, DOMAIN_A_READY, dataset_size, model_input_size)
    preprocess_dataset(dataset_b, DOMAIN_B_FOLDER, DOMAIN_B_READY, dataset_size, model_input_size)


def preprocess_image(image_path, domain_folder, domain_ready_folder, model_input_size, pbar):
    img = Image.open(image_path)
    processed_image = img.resize((model_input_size, model_input_size))
    processed_image = processed_image.convert('RGB')
    processed_image.save(image_path.replace(domain_folder, domain_ready_folder))
    pbar.update()


def preprocess_dataset(dataset_glob, domain_folder, domain_ready_folder, dataset_size, model_input_size):
    shuffled_data = copy(dataset_glob)
    np.random.shuffle(shuffled_data)
    dataset = shuffled_data[:dataset_size]
    with tqdm(total=len(dataset)) as pbar:
        pool = ThreadPool(multiprocessing.cpu_count())
        pool.map(
            lambda image_path: preprocess_image(image_path, domain_folder, domain_ready_folder, model_input_size, pbar),
            dataset)
