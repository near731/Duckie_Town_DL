#Imnports

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import load_model


# Setting hyperparameters:

conv_size_1 = 32
conv_size_2 = 64
conv_size_3 = 128
conv_size_4 = 64
conv_size_5 = 16
dense_size_1 = 256
dense_size_2 = 128
dense_size_3 = 32

def conv_model():
  # Create model
  model = Sequential()
  # Adding Layers

  model.add(Conv2D(
    conv_size_1,
    kernel_size=(3, 3),
    activation='relu',
    strides = (1, 1),
    input_shape=(40,80,3) ))

  model.add(BatchNormalization())

  model.add(Dropout( 0.2 ))

  model.add(Conv2D(
    conv_size_2,
    kernel_size=(5, 5),
    activation='relu',
    strides = (2, 2),
    input_shape=(40,80,3) )) 
  model.add(BatchNormalization())

  model.add(Dropout( 0.2 ))
  
  model.add(Conv2D(
    conv_size_3,
    kernel_size=(3, 3),
    strides = (1, 1),
    activation='relu'))

  model.add(BatchNormalization())

  model.add(Dropout( 0.3 ))

  model.add(Flatten())

  model.add(Dense(dense_size_1, activation='relu'))

  model.add(Dense(dense_size_2, activation='relu'))

  model.add(Dense(dense_size_3, activation='relu'))

  # Output Layer

  model.add(Dense(5, activation='softmax')) 
  
  model.compile(optimizer=keras.optimizers.Adam(),               
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

