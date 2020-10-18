from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
import numpy as np

import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical
from keras import optimizers
import tensorflow as tf
from tensorflow import keras

#Define ResNet50 model training function
def trainModel(DS, x_train, y_train, x_valid, y_valid, x_test, y_test):
  '''Build and Train (pretrained) ResNet50 Model'''
  #Prepare Dataset for training
  x_train, y_train, x_valid, y_valid, x_test, y_test = prepareDataset(x_train, y_train, x_valid, y_valid, x_test, y_test)
  
  # Define the parameters for ResNet50 model.
  NB_EPOCHS = 20
  no_of_class = noClass(DS)
  print('Parameters are defined')
  
  #Base Model
  base_model = ResNet50(include_top=False,
                        pooling='avg',
                        weights=resnet_path))
                           
  print('Base Model is created')
  base_model.summary()
  
  # Adding Dense Layers
  # Global Average Pooling layer (better way to "flatten")
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # Adding a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # Addinf a softmax layer -- no of class
  predictions = Dense(no_of_class, activation='softmax')(x)
  
  # Create new model
  model = Model(inputs=base_model.input, outputs=predictions)
  
  # Conv & pooling layers are not trainable
  for layer in base_model.layers:
    layer.trainable = False
  
  print('Model is created')
  
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
    x_train,
    y_train,
    epochs=NB_EPOCHS,
    validation_data=(x_valid, y_valid),
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
  
  test_acc, test_precision, test_recall, test_fscore, hamming_loss = modelPerformance(model, x_test, y_test)
  
  results = {'train':[acc[-1], val_acc[-1], loss[-1], val_loss[-1]],\
             'test':[test_acc, test_precision, test_recall, test_fscore, hamming_loss]}
  
  return model, history, results
  
  
def noClass(DS):
  '''Define number of original classes in DS'''
  if (DS.lower() == 'mnist') or (DS.lower() == 'fashionmnist'):
    no_class = 10
    
  return no_class

def prepareDataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
  '''>>> InceptionV3 FUNCTION <<<
      Preparing X & Y Data'''

  print('Preparing X Data')
  X_tr = X_train.astype(np.float) / 255.0
  X_v = X_valid.astype(np.float) / 255.0
  X_te = X_test.astype(np.float) / 255.0

  print('Preparing Y Data')
  Y_tr = to_categorical(Y_train)
  Y_v = to_categorical(Y_valid)
  Y_te = to_categorical(Y_test)
  
  print('Data is ready')
  
  return X_tr, Y_tr, X_v, Y_v, X_te, Y_te
  
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