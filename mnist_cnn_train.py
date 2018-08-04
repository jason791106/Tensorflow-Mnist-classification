import h5py
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 15
INPUT_SHAPE = (28, 28, 1)

with h5py.File('./data/mnist_train.h5') as hf:
    X, Y = hf['imgs'][:], hf['labels'][:]
print("Loaded images from mnist_train.h5")


def cnn_model(input_shape=None,NUM_CLASSES = None):

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    fc1 = Flatten()(pool2)
    fc1 = Dense(128, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)
    out = Dense(NUM_CLASSES, activation='softmax', name='fc1000')(fc1)

    model = Model(input=inputs, output=out)

    return model


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

model = cnn_model(input_shape=INPUT_SHAPE, NUM_CLASSES=num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

json_string = model.to_json()
open('./models/mnist_classifier_architecture.json', 'w').write(json_string)

checkpointer = ModelCheckpoint(filepath='./models/mnist_classifier_best_weights.h5', verbose=1, monitor='loss',
                               mode='auto', save_best_only=True)

model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, Y_val),
              shuffle=True,
              verbose=1,
              callbacks=[checkpointer])

model.save_weights('./models/mnist_classifier_last_weights.h5', overwrite=True)
