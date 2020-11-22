# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:59:14 2020

@author: Felix
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf

#from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import normalize, to_categorical
from keras.layers import BatchNormalization
from tqdm import tqdm

import librosa
import librosa.display
from scipy import signal
import time

from librosa.core import istft
from nets import *

FIRSTPLOTS=False
PARAMETRIC=True

def wienergain(xi):
    return np.divide(xi, np.add(xi, 1.0))

def WienerGain(xi,alpha=1,beta=1,parametric=False):
    if parametric==True:
        gain = np.power(np.divide(xi,(alpha+xi)),beta)
    else:
        gain = np.divide(xi,(1+xi))
    
    return gain

def srwf(xi):
	"""
	Computes the square-root Wiener filter (WF) gain function.
	Argument/s:
		xi - a priori SNR.
	Returns:
		SRWF gain function.
	"""
	return np.sqrt(wienergain(xi)) # SRWF gain function.


if FIRSTPLOTS==True:
    t = np.linspace(-60,60,120)
    
    m_ibm = 1/(1+np.exp(-0.1*t))
    
    gb = wienergain(t)
    gb[0:60] *= 0
    
    fig, ax = plt.subplots()
    ax.set_xlim(-60,60)
    ax.plot( t ,m_ibm,'b--')
    
    ax.grid(True)
    ax.set_ylabel("Maske")
    ax.set_xlabel("a priori SNR")
    #plt.legend()
    fig.savefig('aprioSNRmaske.png', dpi=fig.dpi)
    
    #gain 
    xi=t
    gain= wienergain(t)
    fig, ax = plt.subplots()
    ax.set_xlim(-20,30)
    ax.plot(t ,gb,'r--', alpha=0.4)
    ax.plot(t ,gain,'b--')


    ax.grid(True)
    ax.set_ylabel("Gain")
    ax.set_xlabel("SNR")
    #plt.legend()
    fig.savefig('aprioSNRmaske.png', dpi=fig.dpi)


if PARAMETRIC==True:
    tdb= np.linspace(-60,80,140)
    tlin = librosa.db_to_power(tdb)
    # gain1 = WienerGain(tlin)
    # gain2 = WienerGain(tlin,beta=0.5,parametric=True)
    # gain3 = WienerGain(tlin,beta=1.5,parametric=True)
    # gain4 = WienerGain(tlin,beta=2,parametric=True)
    # #gain5 = WienerGain(tlin,beta=2.5,parametric=True)
    # gain5 = WienerGain(tlin,beta=5,parametric=True)
    
    gain1 = WienerGain(tlin)
    gain2 = WienerGain(tlin,alpha=0.5,parametric=True)
    #gain3 = WienerGain(tlin,alpha=1.5,parametric=True)
    gain4 = WienerGain(tlin,alpha=2,parametric=True)
    #gain5 = WienerGain(tlin,beta=2.5,parametric=True)
    gain5 = WienerGain(tlin,alpha=5,parametric=True)
    gain6 = WienerGain(tlin,alpha=0.1,parametric=True)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-20,30)
    
    # ax.plot(tdb ,gain2,'r--', alpha=0.9,label="beta=0.5")
    # ax.plot(tdb ,gain1,'b--', alpha=0.9,label="beta=1")
 
    # ax.plot(tdb ,gain3,'y--', alpha=0.9,label="beta=1.5")
    # ax.plot(tdb ,gain4,'g--', alpha=0.9,label="beta=2")
    # ax.plot(tdb ,gain5,'c--', alpha=0.9,label="beta=5")
    
    ax.plot(tdb ,gain6,'m--', alpha=0.9,label="alpha=0.1")
    ax.plot(tdb ,gain2,'r--', alpha=0.9,label="alpha=0.5")
    ax.plot(tdb ,gain1,'b--', alpha=0.9,label="alpha=1")

    #ax.plot(tdb ,gain3,'y--', alpha=0.9,label="alpha=1.5")
    ax.plot(tdb ,gain4,'g--', alpha=0.9,label="alpha=2")
    ax.plot(tdb ,gain5,'c--', alpha=0.9,label="alpha=5")
    
    
    
    ax.grid(True)
    ax.set_ylabel("Gain")
    ax.set_xlabel("SNR")
    plt.legend()
    plt.show()