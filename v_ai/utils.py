# v_ai/utils.py

import os


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
