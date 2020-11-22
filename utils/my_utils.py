#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:35:50 2020

@author: ubuntu
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
from pystoi import stoi
import utils.pmsqe as pmsqe
#q=3

# t = np.linspace(0, 1, 100)

# # xi = list(range(len(t)))
# stepfunc = enhance2(t)
# # plt.plot(xi, stepfunc, linestyle='-', color='b', label='Square')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # #plt.xticks(xi, stepfunc)s
# # plt.title('compare')
# # plt.legend()
# # plt.show()

# plt.figure()
# plt.plot(np.linspace(0,1,100),stepfunc, 'b')
# #plt.xticks(np.arange(0, 100, step=0.1))
# plt.show()

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


        pmsqe.init_constants(Fs=sr, Pow_factor=pmsqe.perceptual_constants.Pow_correc_factor_Hann, apply_SLL_equalization=True,
                           apply_bark_equalization=True, apply_on_degraded=True, apply_degraded_gain_correction=True)
        
        
        #pesqvalue=pesq(sr, original, denoised, 'wb')
        pesqvalue=per_frame_PMSQE(original,denoised)
        #print(pesqvalue)
    except:
        print("pesq didnt work")
        presqvalue=1

    return pesqvalue


# def stoi_on_batch(y_denoised,ytest,test_phase,sr=16000):
#
#     stoivalue=0
#     try:
#
#
#         y_denoised = np.squeeze(y_denoised,axis=3)
#         y_denoised = np.squeeze(y_denoised,axis=0)
#
#
#
#         y_denoised = librosa.db_to_amplitude(y_denoised)
#         ytest = librosa.db_to_amplitude(ytest)
#
#         denoised = y_denoised*test_phase
#         original = ytest*test_phase
#
#         denoised = librosa.istft(denoised)
#         original = librosa.istft(original)
#
#         #print(denoised)
#         #print(original)
#
#         denoised = librosa.util.normalize(denoised)
#         original = librosa.util.normalize(original)
#
#         #pesqvalue=pesq(sr, original, denoised, 'wb')
#
#
#         stoi=stoi(sr, original, denoised, 'wb')
#         #print(pesqvalue)
#     except:
#         print("stoi didnt work")
#         stoivalue=0
#
#     return stoivalue

def stoi_on_batch(y_denoised,ytest,test_phase,sr=16000):

    stoivalue=0

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


    stoivalue=stoi( original, denoised,sr, 'wb')
    #print(pesqvalue)

    #print("stoi didnt work")
    #stoivalue=0

    return stoivalue


def source_to_distortion(batch_predicted,target_gt):

    batch_predicted = librosa.db_to_amplitude(batch_predicted)
    target_gt = librosa.db_to_amplitude(target_gt)

    distortion = (batch_predicted-target_gt)**2

    return 10*np.log10(np.divide(target_gt, distortion, out=(np.ones_like(Noisy))*50, where=distortion!=0))

def histplot(data,xlab="SNR",ylab="Häufigkeit",binwidth=5,flatten=True):
    data = data.flatten()
    plt.hist(data, bins=range(min(data.astype(np.int)), max(data.astype(np.int)) + binwidth, binwidth),density=True)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()