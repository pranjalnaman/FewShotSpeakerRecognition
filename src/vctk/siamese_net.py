from keras import Model
from keras.layers import Dense, Input
from few_shot_speaker_recognition.src.vctk.distance_function import distance
from few_shot_speaker_recognition.src.vctk.conv_net import get_conv_model


def get_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    conv_model = get_conv_model(input_shape=input_shape, output_shape=8192)

    encoded_l = conv_model(left_input)
    encoded_r = conv_model(right_input)

    distance_vector = distance(encoded_l=encoded_l, encoded_r=encoded_r)
    prediction = Dense(1, activation='sigmoid')(distance_vector)

    return Model(inputs=[left_input, right_input], outputs=prediction)
