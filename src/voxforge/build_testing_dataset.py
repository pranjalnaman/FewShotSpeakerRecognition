import os
import random

from voxforge import CONFIG
from voxforge.helper_functions import flac_to_mfcc


def build():
    DATA_PATH = CONFIG['dataset_2']['data_path']
    speakers = os.listdir(DATA_PATH)
    test_data = []
    for speaker in speakers:
        files = os.listdir(os.path.join(DATA_PATH, speaker))
        for file in files[5:10]:
            test_data.append(
                (flac_to_mfcc(
                    file_path=os.path.join(os.path.join(DATA_PATH, speaker), file)
                ), int(speaker))
            )
    random.shuffle(test_data)
    return test_data