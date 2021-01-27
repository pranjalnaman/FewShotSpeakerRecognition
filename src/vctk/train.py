import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from few_shot_speaker_recognition.src.vctk.build_dataset import get_data
from few_shot_speaker_recognition.src.vctk.helper_functions import split_X_into_left_and_right
from few_shot_speaker_recognition.src.vctk.siamese_net import get_siamese_model

siamese_model = get_siamese_model(input_shape=(20, 196, 1))

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

X_train_left, X_train_right = split_X_into_left_and_right(X=X_train)
X_test_left, X_test_right = split_X_into_left_and_right(X=X_test)

optimizer = Adam(lr=0.00006)
siamese_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
siamese_model.fit(x=[X_train_left, X_train_right], y=np.array(y_train), batch_size=64)

siamese_model.save('model.h5')
#y_pred = siamese_model.predict_classes(X_test)
score = siamese_model.evaluate(x=[X_test_left, X_test_right], y=np.array(y_test), verbose=1)
print(score)
