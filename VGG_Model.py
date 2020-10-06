import os, time, datetime
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
#from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input


from keras import models
from keras.models import Model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU

import Functions as Fn

#Define VGG model training function
def trainVGG(x_train, y_train, x_valid, y_valid, x_test, y_test):
  # Define the parameters for instanitaing VGG16 model. 
  IMG_WIDTH = 48
  IMG_HEIGHT = 48
  IMG_DEPTH = 3
  BATCH_SIZE = 16
  print('Parameters is defined')

  # Preprocessing the input 
  x_train = preprocess_input(x_train)
  x_valid = preprocess_input(x_valid)
  x_test  = preprocess_input(x_test)
  print('Input is preprocessed')

  #Conv base
  conv_base = VGG16(weights='imagenet',\
                    include_top=False, \
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

  print('Conv Base is created')
  conv_base.summary()
  
  
  #Model features
  train_features = conv_base.predict(np.array(x_train), batch_size=BATCH_SIZE, verbose=1)
  val_features = conv_base.predict(np.array(x_valid), batch_size=BATCH_SIZE, verbose=1)
  test_features = conv_base.predict(np.array(x_test), batch_size=BATCH_SIZE, verbose=1)
  
  print('Featurs are created')
  print("train_features shape:", train_features.shape)
  print("test_features shape:", test_features.shape)
  print("val_features shape:", val_features.shape)
  
  train_features_flat = np.reshape(train_features, (train_features.shape[0], 1*1*512))
  val_features_flat = np.reshape(val_features, (val_features.shape[0], 1*1*512))
  test_features_flat = np.reshape(test_features, (test_features.shape[0], 1*1*512))
  print('Features are flattened')
  
  # 7.0 Define the densely connected classifier followed by leakyrelu layer and finally dense layer for the number of classes
  NB_TRAIN_SAMPLES = train_features_flat.shape[0]
  NB_VALIDATION_SAMPLES = val_features_flat.shape[0]
  NB_EPOCHS = 100

  model = models.Sequential()
  model.add(layers.Dense(512, activation='relu', input_dim=(1*1*512)))
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dense(no_of_class, activation='softmax'))
  
  print('Model layers are created')
  
  # Compile the model.
  model.compile(
  loss='categorical_crossentropy',
  optimizer=optimizers.Adam(),
  metrics=['acc'])
  
  print('Model is compiled')
  
  #Define callbacks
  reduce_learning = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=2,
    min_lr=0)

  early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto')

  callbacks = [reduce_learning, early_stopping]
  
  print('Call backs are defined')
  
  print('\n\n>>>Model Trainig starts now')
  
  # Train the the model
  history = model.fit(
    train_features_flat,
    y_train,
    epochs=NB_EPOCHS,
    validation_data=(val_features_flat, y_valid),
    callbacks=callbacks)
    
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  
  print(f'Model training accuracy: {acc}')
  print(f'Model validation accuracy: {val_loss}')
  print(f'Model training loss: {acc}')
  print(f'Model validation loss: {val_loss}')
  
  return acc, val_acc, loss, val_loss, test_features_flat
  
  
  
  
