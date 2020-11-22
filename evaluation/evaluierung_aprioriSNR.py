"""
Evaluation for A PRIORI estimates:
PESQ,STOI and SDR

This script loads datasets containing speech and noise mixture from subdirs as .pkl and
infers speech enhancement by a given net & weights. Afterwards PESQ, STOI and SDR are calculated
and the change in values are beeing recorded to .csv files.

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

# specify the SNR that should be loaded from files.
db=[20,15,10,5,0,-5]
# framelength of contextwindow:
WIN_LEN=32

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



cwd = os.getcwd()

# specify subdirs containing .pkl files with mixture datasets.
subdirs =['Sirene','Auto','Flugzeug','PartyBabble','Straße','Waschmaschine']


## specify which net gets loaded, aswell as a configuration of output bins, contextwindow length, optimizer and loss functions.
## herefore a class from nets.py get called which returns the net with .return_model()
NET_TYPE="cnn_oned_60"
net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
model=net1.return_model()


#load weights belonging to net type:
model.load_weights("cnn_oned_60011aprioSNR.h5")
print(model.summary())




cleanpesqlist=[]

for folder in subdirs:

    
    ## remove old metrics calculations if needed..
    if os.path.exists('metrics'+str(NET_TYPE)+str(folder)+'aprioSNR.csv'):
        os.remove('metrics'+str(NET_TYPE)+str(folder)+'aprioSNR.csv')

    ## load datasets:
    for snr in db:
        print("Noisetype:" +str(folder) +"at " +str(snr) +" SNR")
        print(os.path.join(cwd, folder, "noisyspec_mix_db_"+str(snr)+".pkl"))
        print("Datasets are beeing loaded.. ")
        X=joblib.load(os.path.join(cwd, folder, 'noisyspec_mix_db_'+str(snr)+'.pkl'))
        print("Noisy Data loaded! 50 % done")
        y=joblib.load(os.path.join(cwd, folder, 'cleanspec_mix_db_'+str(snr)+'.pkl'))
        p=joblib.load(os.path.join(cwd, folder, 'noisyphase_mix_db_'+str(snr)+'.pkl'))
        cp=joblib.load(os.path.join(cwd, folder, 'cleanphase_mix_db_'+str(snr)+'.pkl'))
        print("Evaluation Dataset loaded!")

        # create one 2D-Array os stft for processing..
        xta= np.hstack(X)
        yta= np.hstack(y)
        p= np.hstack(p)
        cp= np.hstack(cp)


        ## optional Proprocessing: if BatchNorm on Graph is used this can be commented out

        scaler = preprocessing.StandardScaler().fit(xta[0:1000])
        xta_minmax = scaler.fit_transform(xta)
        print("data scaling! using StandardScaler")

        ## OR no preprocessing:
        #xta_minmax = xta


        sr= 16000   #sampling rate
        BATCH_LEN=800 # length of batch processed at a time

        ## creating lists to store metric values:
        pesqlist=[]
        cleanpesqlist=[]
        stoilist=[]
        cleanstoilist=[]
        sdrlist=[]
        cleansdrlist=[]

        for j in range(1,30):
             # looping through the noisy 2d array, chopping it up to windows of BATCH_LEN and calculating Metrics on batch

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
                 print(np.around(sdrvalue,decimals=2))
             except:
                 pesqvalue=0

             pesqlist.append(np.around(pesqvalue,decimals=2))
             stoilist.append(np.around(stoivalue,decimals=2))
             sdrlist.append(np.around(sdrvalue,decimals=2))


        # Calculate the Mean of last evaluations:

        x = np.around(statistics.mean(pesqlist),decimals=2)
        s= np.around(statistics.mean(stoilist),decimals=2)
        sd = np.array(sdrlist)
        sd = sd.astype(np.float64)
        sd = sd.tolist()
        sd = np.around(statistics.mean(sd),decimals=2)

        print('Mean PESQ= '+str(x))
        print('Mean STOI= '+str(s))
        print('Mean SDR=  '+str(sd))


        for h in range(1,30):
            ## infers the given noisy data through the neural net and once again compares
            ## the infered data to the original undisturbed data for evaluation

            n,ph = inputs2(xta_minmax,p,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
            n=np.array(n)
            
               
            ## CNN models need channel input
            n=np.expand_dims(n,3)
            ## single output layer:
            reg= model.predict(n)
            
            reg = reverse_aprioMask(reg)
            reg = np.clip(reg,-80,80)
            regm = np.subtract(reg,30)
            #regm=reg
        
            #myut.histplot(regm)
            
            regm_lin=librosa.db_to_power(regm)
            wienergain = WienerGain(regm_lin,alpha=1,beta=1,parametric=True)
        
            out = xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
            out = librosa.db_to_power(out)
            wienergain= np.transpose(wienergain)
            out=out*wienergain
            out= librosa.power_to_db(out)
                    
            
         
            ## multi output layer (PMSQE Loss)
            #reg,st = model.predict(n)

            ## 1/reg gives regression value which can directly be multiplied in the log-domain
            #reg = np.divide(1, reg, out=np.ones_like(reg), where=reg!=0)

            ## POST-GAIN if needed:
            #reg[reg>=1.2] *= 1.5
            #reg = np.power(reg,0.5)

            ## clipping values to conquer possible outliers.
            #reg = np.clip(reg, a_min = 1, a_max = 100)

            #out= xta[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*np.transpose(reg)
            #out=np.array(out)
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


            print(pesq(sr,clean, out, 'wb'))
            print(stoi(clean,out,sr,extended=False))
            print(SDR(clean_for_sdr,predicted_for_sdr))


        y = np.around(statistics.mean(cleanpesqlist),decimals=2)
        sy = np.around(statistics.mean(cleanstoilist),decimals=2)

        sdy = np.array(cleansdrlist)
        sdy = sdy.astype(np.float64)
        sdy = sdy.tolist()
        sdy = np.around(statistics.mean(sdy),decimals=2)
        print('Mittelwert PESQ= '+str(y))
        print('Pesq increase:' +str(y-x))

        print("STOI increase:" +str(sy-s))
        print("SDR change: "+str(sdy-sd)+ " db!")

        ## Outputs .csv for every noisetyp with their achieved metric values aswell as their absolute change thrugh NN enhancement.

        with open('metrics'+str(NET_TYPE)+str(folder)+'aprioSNR_mean-30.csv','a',newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow([str(snr),str(x),str(s),str(sd),str(y),str(sy),str(sdy),str(np.around((y-x),decimals=2)),str(np.around((sy-s),decimals=2)),str(np.around((sdy-sd),decimals=2))])
