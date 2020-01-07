READY_PREFIX = "ready_"

DOMAIN_A_FOLDER = "cartoonset10k"
DOMAIN_B_FOLDER = "celeba"

DOMAIN_A_FILES = f"{DOMAIN_A_FOLDER}/*.png"
DOMAIN_A_READY = f"{READY_PREFIX}{DOMAIN_A_FOLDER}"
DOMAIN_A_READY_FILES = f"{DOMAIN_A_READY}/*.png"

DOMAIN_B_FILES = f"{DOMAIN_B_FOLDER}/img_align_celeba/*.jpg"
DOMAIN_B_READY = f"{READY_PREFIX}{DOMAIN_B_FOLDER}"
DOMAIN_B_READY_FILES = f"{DOMAIN_B_READY}/*.jpg"

MODEL_INPUT_SIZE = 128

CELEB_A_URL = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
CELEB_A_HASH = '00d2c5bc6d35e252742224ab0c1e8fcb'