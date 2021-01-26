import librosa
import numpy as np
import tensorflow as tf


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
    return mfcc


def extract_mfcc(file_path):
    """

    Args:
        file_path:

    Returns:

    """
    file_name = bytes.decode(file_path.numpy())
    mfcc = tf.convert_to_tensor(flac_to_mfcc(file_name))
    mfcc = tf.expand_dims(mfcc, 2)
    return mfcc
