import os
import shutil
import zipfile
from tqdm import tqdm
import hashlib
from urllib.request import urlretrieve

from constants import DOMAIN_A_FOLDER, DOMAIN_B_FOLDER, CELEB_A_URL, CELEB_A_HASH


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def prepare_data_folders():
    create_folder_if_not_exist(DOMAIN_A_FOLDER)
    create_folder_if_not_exist(DOMAIN_B_FOLDER)


def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)


def download_celeba(data_path):
    """
    Download and extract database
    :param data_path:
    """
    database_name = 'celeba'
    extract_path = data_path
    save_path = f'{database_name}.zip'

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=f'Downloading {database_name}') as pbar:
            urlretrieve(
                CELEB_A_URL,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == CELEB_A_HASH, \
        f'{save_path} file is corrupted.  Remove the file and try again.'

    try:
        _unzip(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if __name__ == '__main__':
    prepare_data_folders()
    download_celeba(DOMAIN_B_FOLDER)
