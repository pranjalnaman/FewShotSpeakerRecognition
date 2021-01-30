import librosa
import numpy as np


def flac_to_mfcc(file_path, max_pad_len=196):
    """

    Args:
        file_path:
        max_pad_len:

    Returns:

    """
    wave, sample_rate = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sample_rate)
    mfcc = mfcc[:, :max_pad_len]
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfcc = np.expand_dims(mfcc, 2)
    return mfcc


def check_if_files_are_from_same_speaker(filename_1: str, filename_2: str) -> bool:
    """

    Args:
        filename_1:
        filename_2:

    Returns:

    """
    speaker_1, speaker_2 = filename_1.split('-')[0], filename_2.split('-')[0]
    return speaker_1 == speaker_2


def split_X_into_left_and_right(X: list) -> tuple:
    """

    Args:
        X:

    Returns:

    """
    X_left = []
    X_right = []
    for entry in X:
        X_left.append(entry[0])
        X_right.append(entry[1])
    return np.array(X_left), np.array(X_right)
