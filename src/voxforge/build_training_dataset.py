import os
import pickle
from collections import OrderedDict

from voxforge import CONFIG
from voxforge.helper_functions import flac_to_mfcc

DATA_PATH = CONFIG['dataset_2']['data_path']
FILES_PATH = CONFIG['dataset_2']['files_path']


def load_data_to_dictionary(data_path: str) -> OrderedDict:
    """

    Args:
        data_path:

    Returns:

    """
    data_dict = OrderedDict()
    speakers = os.listdir(data_path)

    for speaker in speakers:
        print(speaker)  # Just to keep track of progress
        data_dict[speaker] = []
        filenames = os.listdir(os.path.join(data_path, speaker))
        for file in filenames[:5]:
            mfcc = flac_to_mfcc(file_path=os.path.join(os.path.join(data_path, speaker), file))
            data_dict[speaker].append(mfcc)
    return data_dict


def load_data() -> OrderedDict:
    """

    Returns:

    """
    if 'data_dictionary.pickle' in os.listdir(FILES_PATH):
        with open(os.path.join(FILES_PATH, 'data_dictionary.pickle'), 'rb') as f:
            data_dictionary = pickle.load(f)
    else:
        data_dictionary = load_data_to_dictionary(data_path=DATA_PATH)
        with open(os.path.join(FILES_PATH, 'data_dictionary.pickle'), 'wb') as f:
            pickle.dump(data_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data_dictionary


def flatten_dictionary_to_list(data_dict: OrderedDict) -> list:
    """

    Args:
        data_dict:

    Returns:

    """
    audio_list = []
    speaker_name_list = []
    for key in data_dict.keys():
        audio_list.extend(data_dict[key])
        speaker_name_list.extend([key] * len(data_dict[key]))

    return audio_list, speaker_name_list


def build_paired_dataset(audio_list: list, speaker_name_list: list) -> list:
    """

    Args:
        audio_list:
        speaker_name_list:

    Returns:

    """
    X, y = [], []
    for i in range(len(audio_list)):
        for j in range(i, len(audio_list)):
            entry = (audio_list[i], audio_list[j])
            X.append(entry)
            y.append(1 if speaker_name_list[i] == speaker_name_list[j] else 0)
    return X, y


def get_data() -> tuple:
    """

    Returns:

    """
    data_dict = load_data()
    audio_list, speaker_name_list = flatten_dictionary_to_list(data_dict=data_dict)
    X, y = build_paired_dataset(audio_list=audio_list, speaker_name_list=speaker_name_list)
    return X, y
