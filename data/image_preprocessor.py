import multiprocessing
from copy import copy
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

from PIL import Image
from tqdm import tqdm

from constants import MODEL_INPUT_SIZE, DOMAIN_A_FILES, DOMAIN_B_FILES, DOMAIN_A_FOLDER, DOMAIN_A_READY, \
    DOMAIN_B_READY, DOMAIN_B_FOLDER, MODE_TRAIN, MODE_VAL, MODE_TEST, get_domain_folder_by_mode


def preprocess(model_input_size, train_ratio=0.6, validation_ratio=0.2):
    assert train_ratio + validation_ratio <= 1
    test_ratio = 1 - train_ratio - validation_ratio
    dataset_a = glob(DOMAIN_A_FILES)
    dataset_b = glob(DOMAIN_B_FILES)
    dataset_size = min(len(dataset_a), len(dataset_b))
    mode_list = [MODE_TRAIN] * (int(train_ratio * dataset_size)) + [MODE_VAL] * (int(validation_ratio * dataset_size)) + \
                [MODE_TEST] * (int(test_ratio * dataset_size))
    preprocess_dataset(dataset_a, DOMAIN_A_FOLDER, DOMAIN_A_READY, dataset_size, model_input_size, mode_list)
    preprocess_dataset(dataset_b, DOMAIN_B_FOLDER, DOMAIN_B_READY, dataset_size, model_input_size, mode_list)


def preprocess_image(ziped_value, domain_folder, domain_ready_folder, model_input_size, pbar):
    image_path, mode = ziped_value
    img = Image.open(image_path)
    processed_image = img.resize((model_input_size, model_input_size))
    processed_image = processed_image.convert('RGB')
    processed_image.save(image_path.replace(domain_folder, get_domain_folder_by_mode(domain_ready_folder, mode)))
    pbar.update()


def preprocess_dataset(dataset_glob, domain_folder, domain_ready_folder, dataset_size, model_input_size, mode_list):
    shuffled_data = copy(dataset_glob)
    np.random.shuffle(shuffled_data)
    dataset = shuffled_data[:dataset_size]
    ziped_data = zip(dataset, mode_list)
    with tqdm(total=len(dataset)) as pbar:
        pool = ThreadPool(multiprocessing.cpu_count())
        pool.map(
            lambda ziped_value: preprocess_image(ziped_value, domain_folder, domain_ready_folder, model_input_size,
                                                 pbar), ziped_data)
