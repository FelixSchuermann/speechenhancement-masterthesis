"""functions for signal processing and dataloading

@author: Felix Schürmann, Masters thesis on deep learning methods for speech enhancement
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
import statistics

def dataLoader(part, ibm=False):
    """Function to load all necessary Data for training and infering"""
    global scaler

    """config dataloader:"""
    """use MULTIOUT for no data normalization + second training target as power spectrum
        CALCMEL generates mel-cepstrum components
        MINMAXSCALER is used for normalization of data from 0 to 1"""
    MULTIOUT=False
    NOISEMASK=False
    CALCMEL=False
    MINMAXSCALER=False


    print("Loading datasets: ")
    X=joblib.load('noisyspec_mix_part'+str(part)+'.pkl')
    print("Noisy Data loaded! 50 % done")
    y=joblib.load('cleanspec_mix_part'+str(part)+'.pkl')
    p=joblib.load('noisyphase_mix_part'+str(part)+'.pkl')
    print("Test Data loaded!")

    """Split Dataset into Train / Test Set (non random)"""
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


    """Create large 2D Arrays of STFTs by stacking all utterances together"""
    xte= np.hstack(X_test)
    xta= np.hstack(X_train)
    yte= np.hstack(y_test)
    yta= np.hstack(y_train)
    xval= np.hstack(X_val)
    yval= np.hstack(y_val)




    if CALCMEL==True:
        mel_xta=bark_subbands(xta)
        print('xta subband')
        mel_xte=bark_subbands(xte)
        mel_xta=np.transpose(mel_xta)
        mel_xte=np.transpose(mel_xte)
        mel_yta=bark_subbands(yta)
        mel_yte=bark_subbands(yte)
        mel_yta=np.transpose(mel_yta)
        mel_yte=np.transpose(mel_yte)
        mel_target=SNR(mel_xta,mel_yta,Bark=True)
        mel_target_test=SNR(mel_xte,mel_yte,Bark=True)
        # mel_xta=librosa.feature.melspectrogram(S=xta, n_mels=41)
        # mel_xte=librosa.feature.melspectrogram(S=xte, n_mels=41)
        # mel_yta=librosa.feature.melspectrogram(S=yta, n_mels=41)
        # mel_yte=librosa.feature.melspectrogram(S=yte, n_mels=41)
        print(mel_xte)
        scaler_mel = preprocessing.StandardScaler().fit(mel_xta[0:20000])
        mel_xta = scaler_mel.fit_transform(mel_xta)
        mel_xte = scaler_mel.fit_transform(mel_xte)
        mel_yta = scaler_mel.fit_transform(mel_yta)
        mel_yte = scaler_mel.fit_transform(mel_yte)
        print(mel_xte)
        joblib.dump(mel_xta,"mel_xta.pkl")
        joblib.dump(mel_xte,"mel_xte.pkl")
        joblib.dump(mel_yta,"mel_yta.pkl")
        joblib.dump(mel_yte,"mel_yte.pkl")
        joblib.dump(mel_target,"mel_target.pkl")
        joblib.dump(mel_target_test,"mel_target_test.pkl")
    else:
        #mel_xta= joblib.load("mel_xta.pkl")
        #mel_xte= joblib.load("mel_xte.pkl")
        #mel_yta= joblib.load("mel_yta.pkl")
        #mel_yte= joblib.load("mel_yte.pkl")
        #mel_target= joblib.load("mel_target.pkl")
        #mel_target_test= joblib.load("mel_target_test.pkl")
        mel_xta=None
        mel_xte=None
        mel_yta=None
        mel_yte=None
        mel_target=None
        mel_target_test=None



    if MINMAXSCALER==False:
        xta=np.transpose(xta)
        xte=np.transpose(xte)
        yta=np.transpose(yta)
        yte=np.transpose(yte)

        scaler = preprocessing.StandardScaler().fit(xta[0:1000])
        print("data scaling!")

        X_train_minmax = scaler.fit_transform(xta)
        print('35%')
        y_train_minmax = scaler.fit_transform(yta)
        print('75%')
        X_test_minmax = scaler.fit_transform(xte)
        print('90%')
        y_test_minmax = scaler.fit_transform(yte)

        X_val_minmax = scaler.fit_transform(xval)
        y_val_minmax = scaler.fit_transform(yval)

        X_train_minmax=np.transpose(X_train_minmax)
        X_test_minmax=np.transpose(X_test_minmax)
        y_train_minmax=np.transpose(y_train_minmax)
        y_test_minmax=np.transpose(y_test_minmax)


        #print("no prescaling active")
        # y_train_minmax = yta
        # y_test_minmax=yte
        # X_train_minmax = xta
        # X_test_minmax=xte

        # X_train_minmax=np.transpose(X_train_minmax)
        # X_test_minmax=np.transpose(X_test_minmax)
        # y_train_minmax=np.transpose(y_train_minmax)
        # y_test_minmax=np.transpose(y_test_minmax)

        xta=np.transpose(xta)
        xte=np.transpose(xte)
        yta=np.transpose(yta)
        yte=np.transpose(yte)



    else:

        xta=np.transpose(xta)
        xte=np.transpose(xte)
        yta=np.transpose(yta)
        yte=np.transpose(yte)

        #print("converting to power scale")
        #xta = librosa.db_to_power(xta)
        #xte = librosa.db_to_power(xte)
        #yta = librosa.db_to_power(yta)
        #yte = librosa.db_to_power(yte)


        scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(xta[0:1000])
        print("data scaling! using MinMaxScaler")

        X_train_minmax = scaler.fit_transform(xta)
        print('35%')
        y_train_minmax = scaler.fit_transform(yta)
        print('75%')
        X_test_minmax = scaler.fit_transform(xte)
        print('90%')
        y_test_minmax = scaler.fit_transform(yte)

        X_val_minmax = scaler.fit_transform(xval)
        y_val_minmax = scaler.fit_transform(yval)

        X_train_minmax=np.transpose(X_train_minmax)
        X_test_minmax=np.transpose(X_test_minmax)
        y_train_minmax=np.transpose(y_train_minmax)
        y_test_minmax=np.transpose(y_test_minmax)

        # y_train_minmax = yta
        # y_test_minmax=yta
        # X_train_minmax = xta
        # X_test_minmax=xte

    """Split Phase information so match Train/Test Dataset"""
    PX_train, PX_test = train_test_split(p, test_size=0.2, random_state=1)
    PX_train = np.hstack(PX_train)
    PX_test = np.hstack(PX_test)



    if ibm==True:
        print('Calculating Ideal Binary Mask')
        mask_irm = SNR(xta,yta)
        mask_irm_test=SNR(xte,yte)


        mask_irm = np.array(mask_irm)
        mask_irm_test = np.array(mask_irm_test)
        mask_irm_val = None


    else:
        #print('Calculating Mask with a priori SNR')

        #apriori..
        #mask_irm = apriori_SNR(xta,yta,mask=True)
        #mask_irm_test = apriori_SNR(xte,yte,mask=True)

        #aposteriori
        #print('Calculating Ratio Mask')
        #mask_irm = IRM2(xta,yta)
        #mask_irm_test = IRM2(xte,yte)

        print("ratio mask with ß parameter")
        mask_irm= IRM_lit(yta, xta)
        mask_irm_test= IRM_lit(yte, xte)

        # noisemask
        if NOISEMASK==True:
            print("CALCULATING IRM FOR NOISE ONLY OUTPUT")
            mask_irm = IRM2_noisemask(xta,yta)
            mask_irm_test = IRM2_noisemask(xte,yte)
            mask_irm_val = IRM2_noisemask(xval,yval)


        mask_irm = np.array(mask_irm)
        mask_irm_test = np.array(mask_irm_test)
        #mask_irm_val=np.array(mask_irm_val)
        mask_irm_val = None




    mask_irm = np.transpose(mask_irm)
    mask_irm_test = np.transpose(mask_irm_test)


    if MULTIOUT==True:
        print("Data Loader for multi out loss")
        ##with VAD..
        vada = VAD(yta)
        vade= VAD(yte)

        yta = librosa.db_to_power(yta)
        yte = librosa.db_to_power(yte)
        X_train_minmax= xta
        X_test_minmax=xte
    else:
        vada=None
        vade=None

    return X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax, mask_irm, mask_irm_test, PX_train, PX_test, X_val_minmax, mask_irm_val, mel_xta, mel_xte, mel_target, mel_target_test, xte, yte,xta,yta, vada,vade


def IBM(S, N):
    """IBM by putting 1 on larger amplitudes (not used)"""
    M = []

    for i in range(len(S)):
        m_ibm = 1 * (S[i] > N[i])
        M.append(m_ibm)

    return M

def IBM2(Clean,Noisy,mask=True):
    """IBM without log conversion"""
    M=[]
    Noisy=librosa.db_to_power(Noisy)
    Clean=librosa.db_to_power(Clean)
    N = np.subtract(Noisy,Clean)
    m_ibm= np.divide(Clean, N, out=(np.ones_like(Noisy)*-80), where=N!=0)

    if mask==True:
        m_ibm= (m_ibm >= 0).astype(int)

    return m_ibm


def SNR_to_mask(S,thres):
    """Function to create Binary Mask on Array depending on threshold"""
    thres = float(thres)
    #thres = probability threshold for S fulfilling SNR condition
    M = []

    for i in range(len(S)):
        m_ibm = 1 * (S[i] > thres)
        M.append(m_ibm)
    M=np.array(M)
    return M

def SNR(Noisy,Clean,mask=True,Bark=False):
    """Function to Calculate Signal-to-Noise Ratio, mask==True puts out IBM"""
    m_ibm=[]

    if Bark==False:
        Noisy=librosa.db_to_amplitude(Noisy)
        Clean=librosa.db_to_amplitude(Clean)

    N = np.subtract(Noisy,Clean)

    m_ibm= 20*np.log10(np.divide(Clean, N, out=np.zeros_like(Noisy), where=N!=0))

    print("masking output")
    if mask==True:
        m_ibm= (m_ibm >= 0).astype(int)
    return m_ibm


def apriori_SNR(Noisy,Clean,mask=True):
    """Function to Calculate a-priori SNR
    mask=True puts out sigmoidal mapping function"""

    m_ibm=[]


    Noisy=librosa.db_to_power(Noisy)
    Clean=librosa.db_to_power(Clean)

    N = np.subtract(Noisy,Clean)
    ##small values to avoid divide by zero
    N[N==0] += 0.000001
    Clean[Clean==0] += 0.000001

    apisnr= 20*np.log10(np.divide(Clean, N, out=np.zeros_like(Noisy), where=N!=0))


    """shifting towards zero mean"""
    apisnr= np.nan_to_num(apisnr,nan=100)
    me = np.mean(apisnr[apisnr<=50])
    print("MEAN OF A PRIORI SNR <= 80: " +str(me))
    apisnr = np.subtract(apisnr,me)

    """sigmoidal mapping function"""
    if mask==True:
        m_ibm = np.divide(1,(1+np.exp(-0.1*apisnr)))
        return m_ibm
    else:
        return apisnr

def reverse_aprioMask(mask):
    """Function to reverse the sigmoidal mapping back to SNR in dB domain"""
    return 10*np.log(mask/(1-mask))


def IRM2(N,S):
    """main function for IRM, ratio is calculated directly on dB domain"""
    M = []

    for i in range(len(S)):
        c = np.divide(N[i], S[i], out=np.ones_like(N[i]), where=S[i]!=0)
        M.append(c)

    return M

def IRM2_noisemask(N,S):
    M = []
    N = librosa.db_to_power(N)
    S = librosa.db_to_power(S)
    noise = N-S
    for i in range(len(S)):
        c = np.divide(noise[i], N[i], out=np.ones_like(N[i]), where=noise[i]!=0)
        M.append(c)

    return M




def IRM_lit(S,N):
    """IRM with parameter beta, using power spectrum"""
    M = []
    b = 0.5
    S=librosa.db_to_power(S)
    N=librosa.db_to_power(N)
    N[N==0]+=0.00000001

    for i in range(len(S)):
        c = np.divide(S[i], N[i], out=np.zeros_like(S[i]), where=N[i]!=0)
        M.append(c)

    M= np.array(M)
    M=np.power(M,b)
    return M

def test_accuracy(predicted,ground_truth):
    """function to test IBM accuracy"""
    NK=predicted.shape[0]*(predicted.shape[1])
    print(NK)
    #ACC=np.sum(predicted==ground_truth)/NK
    print(np.count_nonzero(predicted==ground_truth))
    ACC=np.count_nonzero(predicted==ground_truth)/NK
    #print(ACC)
    return ACC



"""BATCH GENERATION FUNCTIONS"""


def inputs2(x_in,y_in,s,b,win_len):
    """Generates a number of batches of contextwindows"""
    """x_in = noisy data, y_in = clean data, respectively IRM mask"""
    """s = starting frame in array, b= stopping frame in array"""
    """win_len = length of contextwindow"""

    """this function returns contextwindows as well as a one dimensional clean/mask frame vectors"""
    i=s
    c=0
    x=[]
    y=[]
    for bins in x_in[1]:
        """create_batch returns one contextwindow"""
        x.append(create_batch_2(x_in,win_len,i))
        try:
            y.append(y_in[:,i])
        except:
            pass
        i=i+1
        if i >= s+b:
            break
    return x, y


def inputs_window(x_in,y_in,s,b,win_len):
    """Same functionality as inputs2, but instead of clean/mask 1D framevector a window of
    the same length as the contextwindow is put out"""
    i=s
    c=0
    x=[]
    y=[]
    for bins in x_in[1]:
        x.append(create_batch_2(x_in,win_len,i))
        try:
            y.append(create_batch_2(y_in,win_len,i))
        except:
            pass
        i=i+1
        if i >= s+b:
            break
    return x, y


def create_batch_2(ar_in,win_len,c):
    """function which cuts out a contextwindow of ar_in"""
    """will pad context window in case counter is smaller than window"""
    win = np.zeros((ar_in.shape[0],win_len))
    i=1
    for m in range(0,win_len):
        if c<=win_len:
            win[:,win_len-1]=ar_in[:,c]
            for l in range(0,c):
                win[:,win_len-1-l]=ar_in[:,c-l]
        else:
            win[:,win_len-i]=ar_in[:,c-i]
            i=i+1
    return win


def batch_gen_standard(X_train_minmax,mask_irm,WIN_LEN,BS,LSTM=False):
    """Generator Function for Neural Net Training"""
    """indefinetly returns batches of contextwindows"""
    i = 0
    loop=True
    while loop==True:
        n=[]
        k=[]
        for o in range(0,BS):
            q, p = inputs2(X_train_minmax,mask_irm,i,1,WIN_LEN)
            n.append(q)
            k.append(p)

        n=np.array(n)
        k=np.array(k)
        n= np.squeeze(n, axis=1)
        n = np.expand_dims(n, axis=3)
        k= np.squeeze(k,axis=1)
        i=i+BS
        if i>=X_train_minmax.shape[1]-1:
            """setting back generator state if dataset is looped trough"""
            i=0
        if LSTM==False:
            yield n,k

        else:
            n = np.reshape(n,(BS,X_train_minmax.shape[0],WIN_LEN))
            yield n,k



def batch_gen_multiout(X_train_minmax,mask_irm,yta,WIN_LEN,BS,LSTM=False):
    """Generator Function which returns multiple training targets"""
    i = 0
    loop=True
    while loop==True:
        n=[]
        k=[]
        ystft=[]
        for o in range(0,BS):
            q, p = inputs2(X_train_minmax,mask_irm,i,1,WIN_LEN)
            _, yst = inputs2(X_train_minmax,yta,i,1,WIN_LEN)
            n.append(q)
            k.append(p)
            ystft.append(yst)

        n=np.array(n)
        k=np.array(k)
        ystft=np.array(ystft)
        n= np.squeeze(n, axis=1)
        n = np.expand_dims(n, axis=3)
        k= np.squeeze(k,axis=1)
        ystft= np.squeeze(ystft,axis=1)

        i=i+BS
        if i>=X_train_minmax.shape[1]-1:
            i=0
        if LSTM==False:
            yield n,[k,ystft]

        else:
            #n = np.reshape(n,(BS,WIN_LEN,X_train_minmax.shape[0]))
            n = np.reshape(n,(BS,X_train_minmax.shape[0],WIN_LEN))
            yield n,k


def batch_gen_window(X_train_minmax,mask_irm,WIN_LEN,BS,LSTM=False):
    """Generator Function for returning target arrays(windows) instead of vectors(frames)"""
    i = WIN_LEN
    loop=True
    while loop==True:
        n=[]
        k=[]
        for o in range(0,BS):
            q, p = inputs_window(X_train_minmax,mask_irm,i,1,WIN_LEN)
            n.append(q)
            k.append(p)

        n=np.array(n)
        k=np.array(k)
        n= np.squeeze(n, axis=1)
        n = np.expand_dims(n, axis=3)
        k= np.squeeze(k,axis=1)
        k = np.expand_dims(k, axis=3)

        i=i+WIN_LEN
        if i>=X_train_minmax.shape[1]-1:
            i=WIN_LEN

        if LSTM==False:
            yield n,k

        else:
            n = np.reshape(n,(BS,X_train_minmax.shape[0],WIN_LEN))
            yield n,k


"""NNET Blocks & Functions"""

def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

def res_net_block_1d(input_data, filters, conv_size, dilation):
  x = layers.Conv1D(filters, conv_size, activation='relu', padding='same',dilation_rate=dilation)(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv1D(filters, conv_size, activation=None, padding='same',dilation_rate=dilation)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

def res_net_block_2d(input_data, filters, conv_size, dilation):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same',dilation_rate=dilation)(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

def res_net_d_bn_block(input_data, filters, conv_size, dilation):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same', dilation_rate=dilation)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

def non_res_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  return x

"""Calculation of Bark Subbands / Energy"""

def bark_subbands(xin):
    ba=[]
    barks=[]
    for b in tqdm(range(0,xin.shape[1])):
        ba = bark_subbandenergy2(xin[:,b:b+1])
        barks.append(ba)

    barks=np.array(barks)
    return barks



def bark_subbandenergy(xin, sr=16000, nfft=256):

    """function for 23 bark bands"""
    # feed in array in dB

    xin = librosa.core.db_to_power(xin)

    # Bark cutoff frequencys:
    Bark= np.array([100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, \
                    1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])

    # find bandwidth in hertz
    m=[]
    barkdif=[]
    for j in range(len(Bark)):
        if j==0:
            m=100
        else:
            m=Bark[j]-Bark[j-1]
            barkdif.append(m)

    #df = frequency resolution - smpling rate divided by number of fft bins

    dF = sr/nfft
    barkdif= np.array(barkdif)

    # calculate how many fft bins correspond to bark bands
    bin_ct = barkdif/dF
    ceil_bin = np.ceil(bin_ct) #round up to integer values


    ceil_bin[-1]=50 ##take some away from last bark band because of rounding error


    # split the FFT into bark bands:
    i=0
    bks=[]
    M=[]
    for k in range(0,len(ceil_bin)):
        bks=xin[i:i+int(ceil_bin[k]),:]
        M.append(bks)
        i=i+int(ceil_bin[k])

    i=0

    # Calculate Band Energies by summing up FFT power Values
    bark_energies=[]
    sb=[]
    for m in range(0,len(M)):
        sb=M[m]
        sb=np.array(sb)
        bark_energies.append(np.sum(sb))
        #print(sb)
    bark_energies=np.array(bark_energies)


    ## divide by barkdif gives energy density..
    #return bark_energies/barkdif
    return bark_energies


def bark_subbandenergy2(xin, sr=8000, nfft=256):
    """function for 60 bands based on bark-scale"""

    # feed in array in dB

    xin = librosa.core.db_to_power(xin)
    Bark= np.array([33,66,100,133,166,200,233,266,300,333,366,400,435,470,510,550,590, 630,680,720,770,830,880,920,980,1030,1080,1140,1200,1270,1340,1410,1480,1570,1650,1720,1800,1900,2000,2100,2210,2320,2450,2560,2700,2820,2940,3060,3150,3300,3450,3700,4000,4300,4600,4950,5350,5800,6400,7100,8000])

    # find bandwidth in hertz
    m=[]
    barkdif=[]
    for j in range(len(Bark)):
        if j==0:
            m=33
        else:
            m=Bark[j]-Bark[j-1]
            barkdif.append(m)

    #df = frequency resolution - smpling rate divided by number of fft bins

    dF = sr/nfft
    barkdif= np.array(barkdif)

    # calculate how many fft bins correspond to bark bands
    bin_ct = barkdif/dF
    ceil_bin = np.ceil(bin_ct) #round up to integer values


    # split the FFT into bark bands:
    i=0
    bks=[]
    M=[]
    for k in range(0,len(ceil_bin)):
        bks=xin[i:i+int(ceil_bin[k]),:]
        M.append(bks)
        i=i+int(ceil_bin[k])

    i=0

    # Calculate Band Energies by summing up FFT power Values
    bark_energies=[]
    sb=[]
    for m in range(0,len(M)):
        sb=M[m]
        sb=np.array(sb)
        bark_energies.append(np.sum(sb))
        #print(sb)
    bark_energies=np.array(bark_energies)


    ## divide by barkdif gives energy density..
    return bark_energies/barkdif
    #return bark_energies, ceil_bin



def barkenergy_target(noisy,clean):
    """if training target are bark energy components"""
    noise=noisy-clean
    target = np.divide(clean, noise, out=(np.ones_like(clean))*30, where=noise!=0)
    return target


def subband_energy_to_fftbins(ceil_bins,xin,start,BATCH_LEN):
    """function to expand bark bands back to stft size (use case: ibm)"""
    xin = np.transpose(xin)

    # takes in the numbers of fft bins corresponding to the bark skale( ceil_bins) and quasi upsamples
    # the bark subbands to FFT size for gain function

    fft_bark_matrix=np.zeros((257,BATCH_LEN))
    bin_state=0
    i=0
    for el in ceil_bins:
        nr_of_bins=int(ceil_bins[i])
        #print(nr_of_bins)
        for k in range(0,nr_of_bins):
            fft_bark_matrix[bin_state+k,:]=xin[i,start:start+BATCH_LEN]
        bin_state=bin_state+nr_of_bins
        i=i+1
    return fft_bark_matrix



def but_filter(mel_target,order,cutoff):
    """butterworth filter approach"""
    mel_filtered=[]
    b, a = signal.butter(order, cutoff)
    for h in range(0,mel_target.shape[0]):
        bark_fr=mel_target[h,:]
        filtsig = signal.filtfilt(b, a, bark_fr)
        mel_filtered.append(filtsig)
    mel_filtered=np.array(mel_filtered)
    return mel_filtered


def globalvariance(xin):
    """dimension-depended global variance"""
    #calculate variances in feature vectors
    gvmean=[]
    #gv=[]
    print (xin.shape)
    for i in tqdm(range(xin.shape[0])):
        g=xin[i,:].tolist()
        g = statistics.mean(g)
        print(g)
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
    print(gv)
    return gv


def globalvariance_independent(xin):
    m = np.mean(xin)
    allv = np.power(xin-m,2)
    allmean = np.mean(allv)
    return allmean


def mask_variance_scaling(xin,mean,gvref,gvest):
    m = xin-mean
    m = m*np.sqrt(gvref/gvest)
    m= m+mean
    return m


def WienerGain(xi,alpha=1,beta=1,parametric=False):
    """Wiener Filter approach, returns Gain function"""
    if parametric==True:
        gain = np.power(np.divide(xi,(alpha+xi)),beta)
    else:
        gain = np.divide(xi,(1+xi))
    return gain

def VAD(y):
    """voicy activity binary mask"""
    m_ibm= (y >= -60).astype(int)
    return m_ibm
