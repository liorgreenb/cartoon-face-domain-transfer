from constants import DOMAIN_A_READY, DOMAIN_B_READY
from data.image_preprocessor import preprocess
from services import clear_folder


def prepare_data_ready_folders():
    clear_folder(DOMAIN_A_READY)
    clear_folder(DOMAIN_B_READY)


def main():
    prepare_data_ready_folders()
    preprocess()


if __name__ == '__main__':
    main()
