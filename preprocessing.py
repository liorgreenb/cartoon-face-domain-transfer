import argparse

from constants import DOMAIN_A_READY, DOMAIN_B_READY, MODEL_INPUT_SIZE, MODE_TRAIN, MODE_TEST, MODE_VAL, \
    get_domain_folder_by_mode
from data.image_preprocessor import preprocess
from services import clear_folder


def prepare_data_ready_folders():
    for domain_ready_folder in [DOMAIN_A_READY, DOMAIN_B_READY]:
        for mode in [MODE_TRAIN, MODE_VAL, MODE_TEST]:
            clear_folder(get_domain_folder_by_mode(domain_ready_folder, mode))


def main(args):
    model_input_size = args.model_input_size
    prepare_data_ready_folders()
    preprocess(model_input_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for the preprocessing step')
    parser.add_argument('--model_input_size', default=MODEL_INPUT_SIZE, type=int, help="")
    args = parser.parse_args()
    main(args)
