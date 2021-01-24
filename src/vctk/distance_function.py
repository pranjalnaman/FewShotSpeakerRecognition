from keras import backend as K


def distance(encoded_l, encoded_r):
    return K.abs(encoded_l - encoded_r)
