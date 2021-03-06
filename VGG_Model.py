import os, time, datetime
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
import numpy as np

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input
from keras import models
from keras.models import Model
from keras import layers
from keras import optimizers
#from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU



#Define VGG model training function
def trainModel(DS, x_train, y_train, x_valid, y_valid, x_test, y_test):
  '''Build and Train (pretrained) VGG16 Model'''
  #Prepare Dataset for training
  x_train, y_train, x_valid, y_valid, x_test, y_test = prepareDataset(x_train, y_train, x_valid, y_valid, x_test, y_test)
  
  # Define the parameters for VGG16 model. 
  IMG_HEIGHT = 48
  IMG_WIDTH = 48
  IMG_DEPTH = 3
  BATCH_SIZE = 16
  NB_EPOCHS = 100
  no_of_class = noClass(DS)
  print('Parameters are defined')

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
  
  from keras import callbacks
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
  
  print('')
  print(f'Model training accuracy: {acc[-1]}')
  print(f'Model validation accuracy: {val_loss[-1]}')
  print(f'Model training loss: {acc[-1]}')
  print(f'Model validation loss: {val_loss[-1]}')
  
  test_acc, test_precision, test_recall, test_fscore, hamming_loss = modelPerformance(model, test_features_flat, y_test)
  
  results = {'train':[acc[-1], val_acc[-1], loss[-1], val_loss[-1]],\
             'test':[test_acc, test_precision, test_recall, test_fscore, hamming_loss]}
  
  return model, history, results
  
def noClass(DS):
  '''Define number of original classes in DS'''
  if (DS.lower() == 'mnist') or (DS.lower() == 'fashionmnist'):
    no_class = 10
    
  return no_class
  
def prepareDataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
  '''>>> VGG FUNCTION <<<
      Preparing X & Y Data'''

  print('Preparing X Data')
  X_tr = prepareXData(X_train)
  X_v = prepareXData(X_valid)
  X_te = prepareXData(X_test)

  print('Preparing Y Data')
  Y_tr = to_categorical(Y_train)
  Y_v = to_categorical(Y_valid)
  Y_te = to_categorical(Y_test)
  
  print('Data is ready')
  
  return X_tr, Y_tr, X_v, Y_v, X_te, Y_te


def prepareXData(DS):
  '''>>> VGG FUNCTION <<<
      Preparing X Data'''
      
  DS = np.dstack([DS] * 3)
  DS = DS.reshape(-1, 28,28,3)
  DS = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in DS])
  DS = DS / 255.
  DS = DS.astype('float32')
  
  return DS
  
def modelPerformance(model, features, labels):
  '''Measure Model Performance'''
  #Predict
  y_pred = model.predict(features)
  
  test_acc = accuracy_score(labels, y_pred.round(), normalize=True, sample_weight=None)  
  test_precision, test_recall, test_fscore, support = precision_recall_fscore_support(labels, y_pred.round(), average='weighted')
  hamming = hamming_loss(labels, y_pred.round())
  
  print('*********************************************')
  print(f'Model testing accuracy: {test_acc}')
  print(f'Model testing precision: {test_precision}')
  print(f'Model testing recall: {test_recall}')
  print(f'Model testing hamming loss: {hamming}')
  
  return test_acc, test_precision, test_recall, test_fscore, hamming