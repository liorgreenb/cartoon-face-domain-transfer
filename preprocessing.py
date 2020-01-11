import argparse

from constants import DOMAIN_A_READY, DOMAIN_B_READY, MODEL_INPUT_SIZE
from data.image_preprocessor import preprocess
from services import clear_folder


def prepare_data_ready_folders():
    clear_folder(DOMAIN_A_READY)
    clear_folder(DOMAIN_B_READY)


def main(args):
    model_input_size = args.model_input_size
    prepare_data_ready_folders()
    preprocess(model_input_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for the preprocessing step')
    parser.add_argument('--model_input_size', default=MODEL_INPUT_SIZE, type=int, help="")
    args = parser.parse_args()
    main(args)
