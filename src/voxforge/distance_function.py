from keras import backend as K


def distance(encoded_l, encoded_r):
    """

    Args:
        encoded_l:
        encoded_r:

    Returns:

    """
    return K.abs(encoded_l - encoded_r)
