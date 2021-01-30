import os
import random

from voxforge import CONFIG
from voxforge.helper_functions import flac_to_mfcc


def build():
    DATA_PATH = CONFIG['dataset_2']['data_path']
    speakers = os.listdir(DATA_PATH)
    data_to_test_against = []
    for speaker in speakers:
        files = os.listdir(os.path.join(DATA_PATH, speaker))
        data_to_test_against.append(
            (flac_to_mfcc(
                file_path=os.path.join(os.path.join(DATA_PATH, speaker), files[0])
            ), int(speaker))
        )
    random.shuffle(data_to_test_against)
    return data_to_test_against
