# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:00:20 2020

@author: Felix Schürmann
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
#from pystoi import stoi
import statistics
from spnn import *
from nets import *

import sys
import scipy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import basename
from tqdm import tqdm
path = os.getcwd()

db_list=[0,5,15]
db=10

#exit()

def histplot(data,xlab="SNR",ylab="Häufigkeit",binwidth=5,flatten=True):
    data = data.flatten()
    plt.hist(data, bins=range(min(data.astype(np.int)), max(data.astype(np.int)) + binwidth, binwidth),density=True)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #plt.show()

def globalvariance(xin):
    gvmean=[]
    print (xin.shape)
    for i in tqdm(range(xin.shape[0])):
        g=xin[i,:].tolist()
        g = statistics.mean(g)
        gvmean.append(g)
    gvmean=np.array(gvmean)

    hssum=[]
    for d in tqdm(range(xin.shape[0])):
        hs=[]
        for j in range(xin.shape[1]):
            h= np.power(xin[d,j]-gvmean[d],2)
            hs.append(h)
        hssum.append(hs)
    hssum=np.array(hssum)

    gv=np.mean(hssum,axis=1)
    return gv

def globalvariance_independent(xin):
    m = np.mean(xin)
    allv = np.power(xin-m,2)
    allmean = np.mean(allv)
    return allmean

q=3
WIN_LEN=32

NOISE_TYPE="Sirene"
NET_TYPE="cnn_oned_60"
net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
model=net1.return_model()
print(model.summary())
model.load_weights("60cnn_oned_32win_adam2350.h5")



plotlist_real=[]
plotlist_inf=[]
varlist=[]
for db in db_list:

    print(db)
    print("Datensaetze werden geladen: ")
    X=joblib.load(path+"\\"+str(NOISE_TYPE)+'\\noisyspec_mix_db_'+str(db)+'.pkl')
    print("Noisy Data loaded! 50 % done")
    y=joblib.load(path+"\\"+str(NOISE_TYPE)+'\\cleanspec_mix_db_'+str(db)+'.pkl')
    p=joblib.load(path+"\\"+str(NOISE_TYPE)+'\\noisyphase_mix_db_'+str(db)+'.pkl')
    cp=joblib.load(path+"\\"+str(NOISE_TYPE)+'\\cleanphase_mix_db_'+str(db)+'.pkl')
    print("Test Dataset loaded!")

    xta= np.hstack(X)
    yta= np.hstack(y)
    p= np.hstack(p)
    cp= np.hstack(cp)


    mask = IRM2(xta,yta)
    mask = np.clip(mask,0,1)
    mask= np.array(mask)
    print(mask.shape)
    gv= globalvariance(mask)
    gv_i = globalvariance_independent(mask)
    mask = np.array(mask)
    mask = np.clip(mask,0,1)
    mask_part = mask[:,0:10000]
    mask_part_flat = mask_part.flatten()

    plotlist_real.append(mask_part_flat)


    num_bins = 40
    n, bins, patches = plt.hist(mask_part_flat, num_bins, density=True, facecolor='cornflowerblue', alpha=1)

    plt.xlabel('Gain Parameter')
    plt.ylabel('Häufigkeit')

    plt.title(r'Histogramm für Rauschtyp '+str(basename(path))+' bei '+str(db)+'db SNR')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

    scaler = preprocessing.StandardScaler().fit(xta[0:1000])
    xta_minmax = scaler.fit_transform(xta)
    print("data scaling! using StandardScaler")
    n,ph = inputs2(xta_minmax,p,0,10000,WIN_LEN)
    n= np.expand_dims(n,3)
    n=np.array(n)
    reg = model.predict(n)

    gvreg=globalvariance(np.transpose(reg))
    gv_i_reg=globalvariance_independent(reg)

    flatreg = reg.flatten()
    plotlist_inf.append(flatreg)
    varlist.append([gv_i,gv_i_reg])

    fig=plt.figure(figsize=(10,4),dpi=200)
    plt.plot(gv, 'b-', label="GV Gain GT")
    plt.plot(gvreg, 'r--', label="GV Gain inferiert")
    plt.legend()
    plt.show()



num_bins=20

colors = ['#E69F00', '#56B4E9']
names= ["IRM", "IRM inferiert"]

fig=plt.figure(figsize=(10,4),dpi=200)
plt.subplot(131)
plt.title('0dB')
plt.hist([plotlist_real[0], plotlist_inf[0]], bins = int(1/0.05), density=True,
         color = colors, label=names)
plt.legend()
plt.subplot(132)
plt.title('5dB')
plt.hist([plotlist_real[1], plotlist_inf[1]], bins = int(1/0.05), density=True,
         color = colors, label=names)
plt.subplot(133)
plt.title('15dB')
plt.hist([plotlist_real[2], plotlist_inf[2]], bins = int(1/0.05), density=True,
         color = colors, label=names)

fig.savefig(str(NET_TYPE)+' Histogramme_par'+str(NOISE_TYPE)+".png")
