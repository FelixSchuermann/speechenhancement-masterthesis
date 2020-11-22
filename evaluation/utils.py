"""
Created on Fri Feb 28 16:35:50 2020

@author: Felix Schürmann
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
import os
import librosa
import numpy as np
import joblib
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import normalize, to_categorical
from keras.layers import BatchNormalization
from spnn import *
from keras.models import model_from_json
from nets import *
from keras.models import Sequential
from scipy import signal
from pesq import pesq

import datetime

class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


class batchend(tf.keras.callbacks.Callback):


  def on_train_batch_end(self, batch, logs=None):
    #print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    print(self.model.layers[-1].output)
    #global predtensor
    predtensor=self.model.layers[-1].output
    print(predtensor)
    print(predtensor.get_values())
    #print(predtensor.numpy())
    #self.model.get_layer("inpred")=predtensor
    #self.model.layers[1].set_weights(predtensor)
    #aua.assign(tf.ones([257,16,1]))
    #print(aua)

import numpy as np

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=10):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))



def downsampler(xtrain,ytrain,xtest,ytest,mask,mask_test):

    numnunzero=100

    alldata=[mask,mask_test]
    q=1
    i=0
    columnstodelete=[]
    for elem in tqdm(range(xtrain.shape[1])):
        #print(np.count_nonzero(mask[:,elem]))
        if np.count_nonzero(mask[:,elem])<=numnunzero:
            columnstodelete.append(i)
            #print(columnstodelete)
        i=i+1
    if q==1:
        xtrain=np.delete(xtrain, columnstodelete, 1)
        ytrain=np.delete(ytrain, columnstodelete, 1)
        mask=np.delete(mask, columnstodelete, 1)
    else:
        xtest=np.delete(xtrain, columnstodelete, 1)
        ytest=np.delete(ytrain, columnstodelete, 1)
        mask_test=np.delete(mask_test, columnstodelete, 1)
    q=2
    i=0
    columnstodelete=[]
    for elem in range(xtest.shape[1]):

       #print(np.count_nonzero(mask_test[:,elem]))
       if np.count_nonzero(mask_test[:,elem])<=numnunzero:
            columnstodelete.append(i)
            #print(columnstodelete)
       i=i+1
    xtest=np.delete(xtest, columnstodelete, 1)
    ytest=np.delete(ytest, columnstodelete, 1)
    mask_test=np.delete(mask_test, columnstodelete, 1)

    return xtrain,ytrain,xtest,ytest,mask,mask_test


def pesq_on_batch(y_denoised,ytest,test_phase,sr=16000):

    pesqvalue=1
    try:


        y_denoised = np.squeeze(y_denoised,axis=3)
        y_denoised = np.squeeze(y_denoised,axis=0)



        y_denoised = librosa.db_to_amplitude(y_denoised)
        ytest = librosa.db_to_amplitude(ytest)

        denoised = y_denoised*test_phase
        original = ytest*test_phase

        denoised = librosa.istft(denoised)
        original = librosa.istft(original)

        #print(denoised)
        #print(original)

        denoised = librosa.util.normalize(denoised)
        original = librosa.util.normalize(original)

        #pesqvalue=pesq(sr, original, denoised, 'wb')


        pesqvalue=pesq(sr, original, denoised, 'wb')
        #print(pesqvalue)
    except:
        print("pesq didnt work")
        presqvalue=1

    return pesqvalue


def source_to_distortion(batch_predicted,target_gt):

    batch_predicted = librosa.db_to_amplitude(batch_predicted)
    target_gt = librosa.db_to_amplitude(target_gt)

    distortion = (batch_predicted-target_gt)**2

    return 10*np.log10(np.divide(target_gt, distortion, out=(np.ones_like(Noisy))*50, where=distortion!=0))
