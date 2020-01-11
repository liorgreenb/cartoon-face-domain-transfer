def get_domain_folder_by_mode(domain_ready_folder, mode):
    return f"{domain_ready_folder}/{mode}"


def get_domain_files_by_mode(domain_ready_folder, mode, ext):
    return f"{get_domain_folder_by_mode(domain_ready_folder, mode)}/*.{ext}"


MODE_TRAIN = "train"
MODE_VAL = "validation"
MODE_TEST = "test"

READY_PREFIX = "ready_"

DOMAIN_B_BASE_FOLDER = "celeba"

DOMAIN_A_FOLDER = "cartoonset10k"
DOMAIN_B_FOLDER = f"{DOMAIN_B_BASE_FOLDER}/img_align_celeba"

DOMAIN_A_FILES = f"{DOMAIN_A_FOLDER}/*.png"
DOMAIN_A_READY = f"{READY_PREFIX}{DOMAIN_A_FOLDER}"
DOMAIN_A_READY_FILES = f"{DOMAIN_A_READY}/*.png"
DOMAIN_A_EXT = 'png'

DOMAIN_B_FILES = f"{DOMAIN_B_FOLDER}/*.jpg"
DOMAIN_B_READY = f"{READY_PREFIX}{DOMAIN_B_BASE_FOLDER}"
DOMAIN_B_READY_FILES = f"{DOMAIN_B_READY}/*.jpg"
DOMAIN_B_EXT = 'jpg'

MODEL_INPUT_SIZE = 32  # TODO change to 128

CELEB_A_URL = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
CELEB_A_HASH = '00d2c5bc6d35e252742224ab0c1e8fcb'
