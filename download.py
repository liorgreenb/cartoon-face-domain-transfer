from constants import DOMAIN_A_FOLDER, DOMAIN_B_FOLDER
from data.datasets_downloader import download_celeba, download_cartoonset10k
from services import create_folder_if_not_exist


def prepare_data_folders():
    create_folder_if_not_exist(DOMAIN_A_FOLDER)
    create_folder_if_not_exist(DOMAIN_B_FOLDER)


def main():
    prepare_data_folders()
    download_celeba(DOMAIN_B_FOLDER)
    download_cartoonset10k()


if __name__ == '__main__':
    main()
