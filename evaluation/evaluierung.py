"""
Created on Sat Apr  4 17:06:21 2020

Evaluation:
PESQ,STOI and SDR


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


#db=[15,10,5,0]
db=[20,15,10,5,0,-5]
WIN_LEN=32

def SDR(original,predicted):



    original = librosa.db_to_amplitude(original)
    predicted = librosa.db_to_amplitude(predicted)

    distortion = predicted-original

    original = original**2
    distortion = distortion**2

    sdr=np.divide(original,distortion)

    sdr= np.nan_to_num(sdr,nan=60.0,posinf=60.0,neginf=60.0)
    #print(sdr)
    sdr = 10*np.log10(sdr)

    return np.mean(sdr)



cwd = os.getcwd()

subdirs =['Sirene','Auto','Flugzeug','PartyBabble','Straße','Waschmaschine']
#subdirs =['Auto','Flugzeug','PartyBabble','Straße','Waschmaschine']

# model= tf.keras.models.load_model("resnet_40blocks_adam2360.h5")
# NET_TYPE="resnet_baseline64subset"

NET_TYPE="densenet"
net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
model=net1.return_model()

model.load_weights("densenet013.h5")
#print(model.summary())


#model= tf.keras.models.load_model("resnet_version2_40blocks_32WIN_adam2350.h5")
#print(model.summary())
#NET_TYPE="resnet_v2"


#ceil_bins=joblib.load("ceil_bins.pkl")
cleanpesqlist=[]

for folder in subdirs:

    if os.path.exists('metrics'+str(NET_TYPE)+str(folder)+'.csv'):
        os.remove('metrics'+str(NET_TYPE)+str(folder)+'.csv')

    for snr in db:
        print("Rauschtyp:" +str(folder) +"bei " +str(snr) +" SNR")
        print(os.path.join(cwd, folder, "noisyspec_mix_db_"+str(snr)+".pkl"))
        print("Datensaetze werden geladen: ")
        #X=joblib.load('noisyspec_mix_db_'+str(snr)+'.pkl')
        X=joblib.load(os.path.join(cwd, folder, 'noisyspec_mix_db_'+str(snr)+'.pkl'))
        print("Noisy Data loaded! 50 % done")

        #y=joblib.load('cleanspec_mix_db_'+str(snr)+'.pkl')
        y=joblib.load(os.path.join(cwd, folder, 'cleanspec_mix_db_'+str(snr)+'.pkl'))
        print()
        p=joblib.load(os.path.join(cwd, folder, 'noisyphase_mix_db_'+str(snr)+'.pkl'))
        cp=joblib.load(os.path.join(cwd, folder, 'cleanphase_mix_db_'+str(snr)+'.pkl'))
        #p=joblib.load('noisyphase_mix_db_'+str(snr)+'.pkl')
        #cp=joblib.load('cleanphase_mix_db_'+str(snr)+'.pkl')
        print("Test Dataset loaded!")

        xta= np.hstack(X)
        yta= np.hstack(y)
        p= np.hstack(p)
        cp= np.hstack(cp)


        scaler = preprocessing.StandardScaler().fit(xta[0:1000])
        xta_minmax = scaler.fit_transform(xta)
        print("data scaling! using StandardScaler")

        #xta_minmax = xta


        sr= 16000
        BATCH_LEN=800
        pesqlist=[]
        cleanpesqlist=[]
        stoilist=[]
        cleanstoilist=[]
        sdrlist=[]
        cleansdrlist=[]
        for j in range(1,30):
             noisy=xta[:,j*BATCH_LEN:j*BATCH_LEN+BATCH_LEN]
             noisy=np.array(noisy)
             noisy_to_sdr=noisy
             phase_noisy=p[:,j*BATCH_LEN:j*BATCH_LEN+BATCH_LEN]
             phase_noisy=np.array(phase_noisy)
             noisy=librosa.db_to_amplitude(noisy)
             noisy=noisy*phase_noisy
             noisy = librosa.istft(noisy)
             noisy = librosa.util.normalize(noisy)

             clean=yta[:,j*BATCH_LEN:j*BATCH_LEN+BATCH_LEN]

             clean=np.array(clean)
             clean_to_sdr=clean
             phase_clean=cp[:,j*BATCH_LEN:j*BATCH_LEN+BATCH_LEN]
             phase_clean=np.array(phase_clean)
             clean=librosa.db_to_amplitude(clean)
             clean=clean*phase_clean
             clean = librosa.istft(clean)
             clean = librosa.util.normalize(clean)


             try:
                 pesqvalue=pesq(sr, clean, noisy, 'wb')
                 print(np.around(pesqvalue,decimals=2))
                 stoivalue=stoi(clean,noisy,sr,extended=False)
                 print(np.around(stoivalue,decimals=2))
                 sdrvalue=SDR(clean_to_sdr,noisy_to_sdr)
                 #sdrvalue=SDR(clean,noisy)
                 #sdrvalue=np.ceil(sdrvalue).astype(np.int())
                 print(np.around(sdrvalue,decimals=2))
             except:
                 pesqvalue=1
             #sdrvalue=SDR(clean_to_sdr,noisy_to_sdr)
             pesqlist.append(np.around(pesqvalue,decimals=2))
             stoilist.append(np.around(stoivalue,decimals=2))
             sdrlist.append(np.around(sdrvalue,decimals=2))


        x = np.around(statistics.mean(pesqlist),decimals=2)
        s= np.around(statistics.mean(stoilist),decimals=2)
        sd = np.array(sdrlist)
        sd = sd.astype(np.float64)
        sd = sd.tolist()
        sd = np.around(statistics.mean(sd),decimals=2)

        print('Mittelwert PESQ= '+str(x))
        print('Mittelwert STOI= '+str(s))
        print('Mittelwert SDR=  '+str(sd))

        for h in range(1,30):

            n,ph = inputs2(xta_minmax,p,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
            n=np.array(n)
            #print(n)
            # CNN models need channel input
            n=np.expand_dims(n,3)
            reg = model.predict(n)
            reg = np.divide(1, reg, out=np.ones_like(reg), where=reg!=0)
            #reg[reg>=1.2] *= 1.5
            #reg = np.power(reg,0.5)
            ## clipping values ## try with greater max attenuation..
            reg = np.clip(reg, a_min = 1, a_max = 100)
            out= xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*np.transpose(reg)
            out=np.array(out)
            predicted_for_sdr=out
            phase_noisy=p[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
            out=librosa.db_to_amplitude(out)
            phase_noisy=np.array(phase_noisy)
            out=out*phase_noisy

            out = librosa.istft(out)
            out= librosa.util.normalize(out)
            print("clean")

            clean=yta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
            clean=np.array(clean)
            clean_for_sdr=clean
            phase_clean=cp[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
            phase_clean=np.array(phase_clean)
            clean=librosa.db_to_amplitude(clean)
            clean=clean*phase_clean
            clean = librosa.istft(clean)
            clean = librosa.util.normalize(clean)

            cleanpesqlist.append(np.around(pesq(sr,clean, out, 'wb'),decimals=2))
            cleanstoilist.append(np.around(stoi(clean,out,sr,extended=False),decimals=2))
            cleansdrlist.append(np.around(SDR(clean_for_sdr,predicted_for_sdr),decimals=2))
            #cleansdrlist.append(SDR(clean,out))
            print(pesq(sr,clean, out, 'wb'))
            print(stoi(clean,out,sr,extended=False))
            print(SDR(clean_for_sdr,predicted_for_sdr))
            #print(SDR(clean,out))

        y = np.around(statistics.mean(cleanpesqlist),decimals=2)
        sy = np.around(statistics.mean(cleanstoilist),decimals=2)
        #sdy = statistics.mean(cleansdrlist)


        sdy = np.array(cleansdrlist)
        sdy = sdy.astype(np.float64)
        sdy = sdy.tolist()
        sdy = np.around(statistics.mean(sdy),decimals=2)
        print('Mittelwert PESQ= '+str(y))
        print('Pesq increase:' +str(y-x))

        print("STOI increase:" +str(sy-s))
        print("SDR change: "+str(sdy-sd)+ " db!")




        #f = open("PESQ_"+str(NET_TYPE)+"_"+str(folder)+"_"+str(snr)+".txt", "w")
        #f.write("Noise PESQ Mean: "+ str(x)+" Clean PESQ Mean: "+str(y)+ "PESQ Increase "+str(y-x)+ "STOI noisy mean"+str(s)+"stoiy clean mean"+str(sy)+"stoi increase"+str(sy-s)+"sdr decrease in db: "+str(sdy-sd))
        #f.close()

        with open('metrics'+str(NET_TYPE)+str(folder)+'SD-strategie.csv','a',newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow([str(snr),str(x),str(s),str(sd),str(y),str(sy),str(sdy),str(np.around((y-x),decimals=2)),str(np.around((sy-s),decimals=2)),str(np.around((sdy-sd),decimals=2))])
