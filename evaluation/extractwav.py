"""
Script for extracting utterances from pickle file. purpose: use wav files as input for comparison
with other frameworks

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


cwd = os.getcwd()
cleanpath= cwd+"\\wavforcomp\\clean\\"
noisypath= cwd+"\\wavforcomp\\noisy\\"

#cleanpath= cwd+"/wavforcomp/clean/"
#noisypath= cwd+"/wavforcomp/noisy/"

subdirs =['Sirene','Auto','Flugzeug','PartyBabble','Straße','Waschmaschine']



for folder in subdirs:

    for snr in db:
        print("Rauschtyp:" +str(folder) +"bei " +str(snr) +" SNR")

        X=joblib.load(os.path.join(cwd, folder, 'noisyspec_mix_db_'+str(snr)+'.pkl'))

        y=joblib.load(os.path.join(cwd, folder, 'cleanspec_mix_db_'+str(snr)+'.pkl'))

        p=joblib.load(os.path.join(cwd, folder, 'noisyphase_mix_db_'+str(snr)+'.pkl'))
        cp=joblib.load(os.path.join(cwd, folder, 'cleanphase_mix_db_'+str(snr)+'.pkl'))

        xta= np.hstack(X)
        yta= np.hstack(y)
        p= np.hstack(p)
        cp= np.hstack(cp)

        sr= 16000
        BATCH_LEN=800
        pesqlist=[]
        cleanpesqlist=[]
        stoilist=[]
        cleanstoilist=[]
        sdrlist=[]
        cleansdrlist=[]
        for j in tqdm(range(1,30)):
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

             #clean = clean.astype(np.int16)
             #noisy = noisy.astype(np.int16)

             ## to match datatype int16
             clean= clean*32768
             noisy= noisy*32768

             clean = np.asarray(clean, dtype=np.int16)
             noisy = np.asarray(noisy, dtype=np.int16)

             #os.chdir(noisypath)
             scipy.io.wavfile.write(noisypath+str(folder)+str(j)+"_"+str(snr)+"_dB"+'.wav',16000,noisy)
             scipy.io.wavfile.write(cleanpath+str(folder)+str(j)+"_"+str(snr)+"_dB"+'.wav',16000,clean)
             #scipy.io.wavfile.write(str(folder)+str(snr)+"_db_"+str(j)+'_clean.wav',16000,clean)
