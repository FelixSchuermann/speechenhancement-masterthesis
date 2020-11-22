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
from pystoi import stoi
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
import csv


path = os.getcwd()
global cn
cn=0

db_list=[15]
#db_list=[20,10,0]



def SDR(original,predicted):
    """
    This function calculated the source-to-distortion ratio on a batchwise stft basis.
    Herefore SDR on all tf-units is found and the mean is put out.
    """
    original = librosa.db_to_amplitude(original)
    predicted = librosa.db_to_amplitude(predicted)

    distortion = predicted-original

    # power spectrum:
    original = original**2
    distortion = distortion**2

    sdr=np.divide(original,distortion)
    # Fixing NAN Values:
    sdr= np.nan_to_num(sdr,nan=60.0,posinf=60.0,neginf=60.0)
    sdr = 10*np.log10(sdr)
    return np.mean(sdr)


def pesq_from_fft(noisy,phase_noisy,clean,phase_clean,out=False):
     """
     Calculate PESQ Metric on stft batch
     """
     phase_noisy=np.array(phase_noisy)
     noisy=librosa.db_to_amplitude(noisy)

     noisy=noisy*phase_noisy
     noisy = librosa.istft(noisy)

     clean=np.array(clean)
     phase_clean=np.array(phase_clean)
     clean=librosa.db_to_amplitude(clean)
     clean=clean*phase_clean
     clean = librosa.istft(clean)

     if out==True:
         global cn

         scipy.io.wavfile.write(path+'\\gvepre\\predictGVE'+str(cn)+'.wav',16000,noisy)
         cn=cn+1

     sr =16000

     pesqvalue=pesq(sr, clean, noisy, 'wb')
     #print(pesqvalue)
     return pesqvalue

def stoi_from_fft(noisy,phase_noisy,clean,phase_clean):
     """
     Calculate STOI Metric on stft batch
     """

     phase_noisy=np.array(phase_noisy)
     noisy=librosa.db_to_amplitude(noisy)

     noisy=noisy*phase_noisy
     noisy = librosa.istft(noisy)

     clean=np.array(clean)
     phase_clean=np.array(phase_clean)
     clean=librosa.db_to_amplitude(clean)
     clean=clean*phase_clean
     clean = librosa.istft(clean)

     sr =16000

     stoivalue=stoi(noisy, clean,sr, 'wb')
     #print(pesqvalue)
     return stoivalue



def globalvariance(xin):
    """
    function for dimension-depended global variance
    """
    gvmean=[]
    print (xin.shape)
    for i in tqdm(range(xin.shape[0])):
        g=xin[i,:].tolist()
        g = statistics.mean(g)
        #print(g)
        gvmean.append(g)
    gvmean=np.array(gvmean)
    vk=statistics.mean(gvmean)
    print(statistics.mean(gvmean))

    hssum=[]
    for d in tqdm(range(xin.shape[0])):
        hs=[]
        for j in range(xin.shape[1]):
            h= np.power(xin[d,j]-gvmean[d],2)
            #print(h)
            hs.append(h)
        hssum.append(hs)
    hssum=np.array(hssum)
    print(hssum.shape)
    hssum= np.sum(hssum,axis=1)
    print(hssum.shape)

    gv=hssum/xin.shape[1]
    return gv,vk

def globalvariance_independent(xin):
    m = np.mean(xin)
    allv = np.power(xin-m,2)
    allmean = np.mean(allv)
    return allmean,m


def mask_variance_scaling(xin,mean,gvref,gvest):
    m = xin-mean
    beta= np.sqrt(gvref/gvest)
    #beta= np.sqrt(gvref/gvest)*2
    print('BETA='+str(np.sqrt(gvref/gvest)))
    m = m*np.sqrt(gvref/gvest)
    m= m+mean
    m = np.clip(m,0,1)

    return m,beta


q=3
WIN_LEN=32

NET_TYPE="cnn_oned_60"
net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
model=net1.return_model()
print(model.summary())
model.load_weights("60cnn_oned_32win_adam2350.h5")


NLIST=['Sirene','Auto','Flugzeug','PartyBabble','Straße','Waschmaschine']

plotlist_real=[]
plotlist_inf=[]
varlist=[]
for NOISE_TYPE in NLIST:
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
        gv, _= globalvariance(mask)
        gv_i,_ = globalvariance_independent(mask)

        #xin=mask


        #exit()
        mask = np.array(mask)

        mask = np.clip(mask,0,1)
        mask_part = mask[:,0:10000]
        mask_part_flat = mask_part.flatten()

        plotlist_real.append(mask_part_flat)


        num_bins = 10
        # the histogram of the data
        n, bins, patches = plt.hist(mask_part_flat, num_bins, density=1, facecolor='royalblue', alpha=1)

        #y = mlab.normpdf(bins, mu, sigma)
        #plt.plot(bins, y, 'r--')
        plt.xlabel('Gain Parameter')
        plt.ylabel('Häufigkeit normiert')
        plt.title(r'Histogramm für Rauschtyp '+str(basename(path))+' bei '+str(db)+'db SNR')

        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.show()



        scaler = preprocessing.StandardScaler().fit(xta[0:1000])
        xta_minmax = scaler.fit_transform(xta)
        print("data scaling! using StandardScaler")
        n,ph = inputs2(xta_minmax,p,0,40000,WIN_LEN)
        n=np.array(n)
        #print(n)
        oraclemean = np.mean(mask)

        n = np.expand_dims(n,axis=3)
        reg = model.predict(n,verbose=1)
        reg = np.transpose(reg)
        reg = np.power(reg,1.5)


        gvreg,_=globalvariance(np.transpose(reg))
        gv_i_reg,_=globalvariance_independent(reg)
        regvar,beta = mask_variance_scaling(reg,oraclemean,gv_i,gv_i_reg)
        reg = np.divide(1, reg, out=np.ones_like(reg), where=reg!=0)
        #reg=reg*1.2

        infered= xta[:,0:40000]*reg
        original = yta[:,0:40000]

        infered = np.clip(infered,-80,0)
        original = np.clip(original,-80,0)

        inferedGV, ik= globalvariance(infered)
        originalGV, ik2= globalvariance(original)
        inferedGVI,mii = globalvariance_independent(infered)
        originalGVI,mio = globalvariance_independent(original)

        inferedGVI= np.full((257),inferedGVI)
        originalGVI= np.full((257),originalGVI)
        gvn, ik3 = globalvariance(xta)
        #gvn = np.sqrt(gvn)


        norm= np.abs(ik3)
        fig=plt.figure(figsize=(10,4),dpi=200)
        plt.plot(gvn, 'g--', label="GV gestörtes Signal",alpha=0.5)
        plt.plot(inferedGV, 'b-', label="GV Features inferiert")
        plt.plot(originalGV, 'r-', label="GV Features GT")
        plt.plot(np.arange(0,257,1),inferedGVI, 'b--', label="GVI Features inferiert")
        plt.plot(np.arange(0,257,1),originalGVI, 'r--', label="GVI Features GT")
        plt.xlabel("FFT Bins")
        #plt.ylabel("$\sigma$")
        plt.legend()
        fig.savefig('GVEplot'+str(NOISE_TYPE)+str(db)+'.png', dpi=fig.dpi)
        plt.show()




        flatreg = reg.flatten()
        plotlist_inf.append(flatreg)
        varlist.append([gv_i,gv_i_reg])


        regvar = np.divide(1, regvar, out=np.ones_like(regvar), where=regvar!=0)
        #regvar = np.clip(regvar,0,1)


        varinc=[]
        penoisyl=[]
        peregl=[]
        peregvarl=[]
        stoiinc=[]
        sdrn=[]
        sdrreg=[]
        sdrregvar=[]
        sdrinc=[]

        BATCH_LEN=800

        for h in range(0,30):

            """PESQ"""
            pe_noisy = pesq_from_fft(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])
            pe_reg = pesq_from_fft(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*reg[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])
            pe_regvar = pesq_from_fft(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*regvar[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN],out=True)

            """STOI"""
            st_noisy = stoi_from_fft(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])
            st_reg = stoi_from_fft(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*reg[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])
            st_regvar = stoi_from_fft(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*regvar[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])

            """SDR"""
            sdr_noisy = SDR(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN],yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])
            sdr_reg = SDR(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*reg[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])
            sdr_regvar = SDR(xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*regvar[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN], yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN])


            sdrn.append(sdr_noisy)
            sdrreg.append(sdr_reg)
            sdrregvar.append(sdr_regvar)

            penoisyl.append(pe_noisy)
            peregl.append(pe_reg)
            peregvarl.append(pe_regvar)


            print(pe_noisy,pe_reg,pe_regvar)

            varinc.append(pe_regvar-pe_reg)
            stoiinc.append(st_noisy-st_reg)
            sdrinc.append(sdr_regvar-sdr_reg)

        pnmean= statistics.mean(penoisyl)
        pimean= statistics.mean(peregl)
        pegvmean= statistics.mean(peregvarl)
        stmean= statistics.mean(stoiinc)
        sdrmean= statistics.mean(sdrinc)
        print(NOISE_TYPE)
        print("PESQ AT:"+str(db))
        print("without GVE: " + str(pimean))
        print("with GVE: " + str(pegvmean))
        print("SDR change: " +str(sdrmean))

        """Output metrics in CSV file"""
        with open('GVE'+str(NET_TYPE)+str(NOISE_TYPE)+'PG.csv','a',newline='') as f:
                thewriter = csv.writer(f)
                thewriter.writerow([str(db),str(np.around(pnmean,decimals=2)),str(np.around(pimean,decimals=2)),str(np.around(pegvmean,decimals=2)),str(np.around(stmean,decimals=2)),str(np.around(beta,decimals=2)),str(np.around(sdrmean,decimals=2))])

        del X,y, p, cp, mask,mask_part,gvreg,reg
