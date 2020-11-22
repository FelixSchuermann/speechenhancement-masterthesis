"""
Script for plotting evaluation metrics from .csv files

@author: Felix Schürmann, Masters Thesis on Deep Learning methods for speech enhancement.
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

import sys
import scipy
import csv

import numpy as np

subdirs =['Sirene','Auto','Flugzeug','PartyBabble','Straße','Waschmaschine'] ## noise subdirs
db=[20,15,10,5,0,-5] ## all SNRs
csv = []
NET_TYPE="cnn_oned_60"

#read in .csv files with evaluation metrics:
for h in range(0,6):
    csv.append(np.genfromtxt ('metrics'+str(NET_TYPE)+subdirs[h]+'aprioSNR_mean-30.csv', delimiter=","))


#which metrics shall be plotted:

PESQANDSTOI=1
SNR15=0
POSTGAIN=0
LDSPLOT=0


# convert all metrics into one 3D-Array
csv = np.array(csv)



if PESQANDSTOI==1:
    """
    Plots STOI and PESQ Metrics for all noise types at all SNRs   
    """

    
    pesqS = csv[0,:,7]
    pesqA= csv[1,:,7]
    pesqF = csv[2,:,7]
    pesqP= csv[3,:,7]
    pesqStr = csv[4,:,7]
    pesqWa= csv[5,:,7]


    fig, ax = plt.subplots()
    ax.set_xlim(20,-5)
    ax.plot(db ,pesqS,'b--', label='Sirene', marker="x")
    plt.plot(db,pesqA,'r--', label="Auto",marker="x")
    plt.plot(db,pesqF,'y--', label="Flugzeug",marker="x")
    plt.plot(db,pesqP,'g--', label="PartyBabble",marker="x")
    plt.plot(db,pesqStr,'c--', label ="Straße", marker="x")
    plt.plot(db,pesqWa,'m--', label="Waschmaschine",marker="x")
    ax.grid(True)
    ax.set_ylabel("PESQ")
    ax.set_xlabel("db SNR")
    plt.legend()
    fig.savefig(str(NET_TYPE)+'pesq_all', dpi=fig.dpi)
    
    
    pesqS = csv[0,:,8]
    pesqA= csv[1,:,8]
    pesqF = csv[2,:,8]
    pesqP= csv[3,:,8]
    pesqStr = csv[4,:,8]
    pesqWa= csv[5,:,8]

    print(db)
    print(pesqS)

    fig, ax = plt.subplots()
    ax.set_xlim(20,-5)
    ax.plot( db ,pesqS,'b--', label='Sirene', marker="x")
    plt.plot(db,pesqA,'r--', label="Auto",marker="x")
    plt.plot(db,pesqF,'y--', label="Flugzeug",marker="x")
    plt.plot(db,pesqP,'g--', label="PartyBabble",marker="x")
    plt.plot(db,pesqStr,'c--', label ="Straße", marker="x")
    plt.plot(db,pesqWa,'m--', label="Waschmaschine",marker="x")
    ax.grid(True)
    ax.set_ylabel("STOI")
    ax.set_xlabel("db SNR")
    plt.legend()
    fig.savefig(str(NET_TYPE)+'stoi_all', dpi=fig.dpi)


if POSTGAIN==1:

    """
    Post-GAIN PESQ Plots for in car noise at tuned parameter y
    
    """
    
    ex = [1,1.5,2,2.5,5]
    # load post gain metrics file
    pesqgain= np.load('pesqgain.npy')
    
    pesqgain= np.transpose(pesqgain)
    fig, ax = plt.subplots()
    ax.set_xlim(20,-5)
    ax.set_ylim(-0.3,0.6)
    ax.plot(db ,pesqgain[0],'b--', label="\u03B3=1", marker="x")
    plt.plot(db,pesqgain[1],'r--', label="\u03B3=1.5",marker="x")
    plt.plot(db,pesqgain[2],'y--', label="\u03B3=2",marker="x")
    plt.plot(db,pesqgain[3],'g--', label="\u03B3=2.5",marker="x")
    plt.plot(db,pesqgain[4],'c--', label ="\u03B3=5", marker="x")
    ax.grid(True)
    ax.set_ylabel("PESQ")
    ax.set_xlabel("db SNR")
    plt.legend()
    fig.savefig('incar_pesqgain.png', dpi=fig.dpi)
    


if SNR15==1:
    
    """
    All metrics at 15db SNR
    
    """
    
    pesq15 = csv[:,1,7]
    stoi15= csv[:,1,8]
    
    from matplotlib.ticker import FormatStrFormatter

    degrees=45
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[4,3])
    plt.bar(subdirs,pesq15,width=0.8,align='center')
    plt.xticks(rotation=degrees)
    plt.tight_layout()
    fig.savefig('pesq15.png', dpi=fig.dpi)
    plt.show()
    
    
    fig = plt.figure(figsize=[4,3])

    plt.bar(subdirs,stoi15,width=0.8,align='center')
    plt.xticks(rotation=degrees)
    plt.tight_layout()
    fig.savefig('stoi15.png', dpi=fig.dpi)
    plt.show()
    
    sdr = csv[:,:,9]
    
    fig = plt.figure(figsize=[4,3])
    ax = fig.add_axes([0,0,1,1])
    im=ax.imshow(sdr)
    subdirs.insert(0,"1")
    db.insert(0,"1")
    ax.set_yticklabels(subdirs)
    ax.set_xticklabels(db)
    cb=fig.colorbar(im, ax=ax)
    plt.xlabel("db SNR")
    cb.set_label("SDR")
    plt.tight_layout()
    fig.savefig('sdr15.png', dpi=fig.dpi)
    plt.show()
 
if LDSPLOT==1:
    
    """
    Compare spectral densities of different noise types:
    
    """
    
    yplane, sr = librosa.load('D:/database/de/noise/airplane_landing16_c1.wav', sr=16000)
    ycar, sr = librosa.load('D:/database/de/noise/in-car16_c1.wav', sr=16000)
    yvoice, sr = librosa.load('D:/database/de/cleanwav/clean-1-common_voice_de_17284738.mp3.wav', sr=16000)
    
    
    from scipy import signal
    
    freqscar, psdcar = signal.welch(ycar)
    freqsplane, psdplane = signal.welch(yplane)
    freqsvoice, psdvoice = signal.welch(yvoice)
    
    plt.figure(figsize=(5, 4))
    plt.semilogx(freqscar, psdcar)
    #plt.title('Spektrale Leistungsdichte Auto')
    plt.xlabel('Frequenz')
    plt.ylabel('Energie')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(5, 4))
    plt.semilogx(freqsplane, psdplane)
    #plt.title('Spektrale Leistungsdichte Flugzeug')
    plt.xlabel('Frequenz')
    plt.ylabel('Energie')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(5, 4))
    plt.semilogx(freqsvoice, psdvoice)
    #plt.title('Spektrale Leistungsdichte Sprache')
    plt.xlabel('Frequenz')
    plt.ylabel('Energie')
    plt.tight_layout()
    plt.show()
    
    WAV_DIR='D:/database/de/cleanwav/'
    import random
    psdlist=[]
    for j in range(0,1000):
        print(j)
        path = WAV_DIR
        rnd_file = random.choice(os.listdir(WAV_DIR))
        yvoice, sr = librosa.load(WAV_DIR+rnd_file, sr=16000)
        freqsvoice, psdvoice = signal.welch(yvoice)
        psdlist.append(psdvoice)
    
    sums = sum(psdlist)
    sumstot = sums/1000
    
    plt.figure(figsize=(5, 4))
    plt.semilogx(freqsvoice, sumstot)
    #plt.title('Spektrale Leistungsdichte Sprache')
    plt.xlabel('Frequenz')
    plt.ylabel('Energie')
    plt.tight_layout()
    plt.show()
