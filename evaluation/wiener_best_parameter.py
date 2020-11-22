# -*- coding: utf-8 -*-
"""
experimental script for finding best wiener filter parameters for used apriori NN

"""

import numpy as np
import datetime as dt
import os
import librosa
import numpy as np
import joblib
import sklearn
from sklearn import preprocessing
from pesq import pesq
from pystoi import stoi
import statistics
from spnn import *
from nets import *
from tqdm import tqdm



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#import tensorflow as tf

from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
from keras.utils import normalize, to_categorical
from keras.layers import BatchNormalization
from spnn import *
from keras.models import model_from_json
from nets import *
from keras.models import Sequential
from scipy import signal

import my_utils as myut

ibm=False

WIN_LEN=32
#model = baseline_resnet(WIN_LEN)

import sys
import scipy
import csv

INIT=False
FULL2D=False
ONEAR=False
PLOTONLY=True

if INIT==True:
    """
    Model and Data Loading. See train_infer.py for documentation
    """

    ceil_bins=joblib.load("ceil_bins2.pkl")
    h=2
    BATCH_LEN=10000
    q=11
    WIN_LEN=32

    NET_TYPE="cnn_oned_60"

    net_regression=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
    regression=net_regression.return_model()
    print(regression.summary())
    regression.load_weights("cnn_oned_60011aprioSNR.h5")

    _, _, X_test_minmax, _, mask, mask_test,PX_train,PX_test,_,_, _,_,_,_,xte,yte,xta,_ = dataLoader(q,ibm)

    n,ph = inputs2(X_test_minmax,PX_test,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
    n=np.array(n)

    n = np.expand_dims(n,axis=3)
    reg=regression.predict(n,verbose=1)
    print(reg)
    reg = reverse_aprioMask(reg)
    regm = np.subtract(reg,30)

    myut.histplot(regm)

    regm_lin=librosa.db_to_power(regm)
    wienergain = WienerGain(regm_lin,alpha=0.5,beta=2,parametric=True)

    out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    out = librosa.db_to_power(out)
    wienergain= np.transpose(wienergain)
    out=out*wienergain
    out= librosa.power_to_db(out)
    noisy_out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    original = yte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]

    mask_test=np.transpose(mask_test)
    originalSNR = mask_test[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    myut.histplot(originalSNR)

    print(out)

    ph=np.array(ph)
    ph= np.transpose(ph)

    out= librosa.db_to_amplitude(out)
    noisy_out= librosa.db_to_amplitude(noisy_out)
    original = librosa.db_to_amplitude(original)
    out = out*ph
    noisy_out = noisy_out*ph
    original = original*ph


    spec_n = noisy_out


    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(spec_n), y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/noisy_res'+str(h)+'.png', dpi=fig.dpi)

    out = librosa.istft(out)
    #out = np.clip(out, a_min = -1, a_max = 1)
    noisy_out=librosa.istft(noisy_out)
    original = librosa.istft(original)

    out = librosa.util.normalize(out)
    noisy_out = librosa.util.normalize(noisy_out)
    original = librosa.util.normalize(original)


    betalist = [0.1,0.2,0.3,0.4,0.5,1,1.5,2,5,10]
    alphalist= [0.1,0.2,0.3,0.4,0.5,1,1.5,2,5,10]

    pesqbeta=[]
    pesqalpha=[]

    PLOTONLY=True

if ONEAR==True:
    """
    Loop through combination of alpha and beta and do evaluation with objevtive measures.
    """
    results=[]
    for valA in tqdm(alphalist):
        for valB in tqdm(betalist):
            wienergain = WienerGain(regm_lin,alpha=valA,beta=valB,parametric=True)
            out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
            out = librosa.db_to_power(out)
            wienergain= np.transpose(wienergain)
            out=out*wienergain
            out= librosa.power_to_db(out)
            out= librosa.db_to_amplitude(out)
            out = out*ph
            out = librosa.istft(out)


            print(pesq(16000,original, out, 'wb'))
            results.append(pesq(16000,original, out, 'wb'))

        print(results)
if PLOTONLY==True:
    import numpy
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    # Set up grid and test data
    nx, ny = 10,10
    x = range(nx)
    y = range(ny)

    data = numpy.random.random((nx, ny))
    print(data.shape)
    dmg_mat = np.load("wienerbestparameters.npy")

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X.T, Y.T, dmg_mat)
    plt.show()
    hf = plt.figure()

    plt.show()
