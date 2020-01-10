import os
import shutil
import zipfile


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def clear_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    create_folder_if_not_exist(path)


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
