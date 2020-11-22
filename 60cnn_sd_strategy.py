"""
Dilated-CNN-LSTM Training on SD-Strategy
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
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Permute, Conv2D,MaxPooling2D, Dropout

"""
SD Training Strategy, all necessary functions to train/test are within this file.
Processing functions are documented in spnn.py
For extensive code annotations see train_infer.py
"""

TRAIN=1
TEST=0
WIN_LEN=32

NET1=False
NET2=True

global lr0
lr0=0.001

def scheduler(epoch):
  """
  LearningRateScheduler as mentioned in thesis.
  """
  global lr0
  if epoch <= 1:
    lr0 = 0.0005
    return 0.0005
  else:
    return 0.0001 * (0.1 * (25-epoch))



load_full_set=False

if load_full_set==False:
    print("Datensaetze werden geladen: ")
    X=joblib.load('noisyspec_mix_part11.pkl')
    print("Noisy Data loaded! 50 % done")
    y=joblib.load('cleanspec_mix_part11.pkl')
    p=joblib.load('noisyphase_mix_part11.pkl')
    print("Test Dataset loaded!")

else:

    print("Datensaetze werden geladen: ")
    X=joblib.load('noisyspec.pkl')
    print("Noisy Data loaded! 50 % done")
    y=joblib.load('cleanspec.pkl')
    #p=joblib.load('noisyphase_test.pkl')
    print("Dataset loaded!")


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


xte= np.hstack(X_test)
xta= np.hstack(X_train)
yte= np.hstack(y_test)
yta= np.hstack(y_train)

xte= np.array(xte)
print(xte.shape)

scaler = preprocessing.StandardScaler().fit(xta[0:1000])
print("data scaling!")

X_train_minmax = scaler.fit_transform(xta)
print('35%')
y_train_minmax = scaler.fit_transform(yta)
print('75%')
X_test_minmax = scaler.fit_transform(xte)
print('90%')
y_test_minmax = scaler.fit_transform(yte)


if load_full_set==False:
    PX_train, PX_test = train_test_split(p, test_size=0.2, random_state=1)
    PX_train = np.hstack(PX_train)
    PX_test = np.hstack(PX_test)



def IBM(S, N):
    M = []
    for i in range(len(S)):
        m_ibm = 1 * (S[i] > N[i])
        M.append(m_ibm)
    return M

def IRM(S, N):
    M = []
    for i in range(len(S)):
        M.append(m_ibm)
    return M


def IRM2(S,N):
    M = []
    for i in range(len(S)):
        c = np.divide(S[i], N[i], out=np.ones_like(S[i]), where=N[i]!=0)
        M.append(c)
    return M

def apriori_SNR(Noisy,Clean,mask=True):
    m_ibm=[]

    Noisy=librosa.db_to_power(Noisy)
    Clean=librosa.db_to_power(Clean)

    N = np.subtract(Noisy,Clean)
    N[N==0] += 0.000001
    Clean[Clean==0] += 0.000001

    apisnr= 20*np.log10(np.divide(Clean, N, out=np.zeros_like(Noisy), where=N!=0))

    apisnr= np.nan_to_num(apisnr,nan=100)
    me = np.mean(apisnr[apisnr<=80])
    print("MEAN OF A PRIORI SNR <= 80: " +str(me))
    apisnr = np.subtract(apisnr,me)


    if mask==True:
        m_ibm = np.divide(1,(1+np.exp(-0.1*apisnr)))
        return m_ibm
    else:
        return apisnr



if TRAIN==1:
    mask_irm = IRM2(yta,xta)
    mask_irm_test = IRM2(yte,xte)

    ## function differing to spnn.py , be careful , use 1/mask with IRM2 and clip

    mask_irm = np.array(mask_irm)
    mask_irm_test = np.array(mask_irm_test)

    mask_irm = np.clip(mask_irm,-80,60)
    mask_irm_test = np.clip(mask_irm_test,-80,60)

    mask_irm = np.divide(1, mask_irm, out=np.ones_like(mask_irm), where=mask_irm!=0)
    mask_irm = np.clip(mask_irm,0,1)

    print('Train Mask fitted. Continuing with Test Mask')

    mask_irm_test = np.divide(1, mask_irm_test, out=np.ones_like(mask_irm_test), where=mask_irm_test!=0)
    mask_irm_test = np.clip(mask_irm_test,0,1)



def inputs2(x_in,y_in,s,b,win_len):
    i=s
    c=0
    x=[]
    y=[]
    for bins in x_in[1]:
        x.append(create_batch_2(x_in,win_len,i))
        y.append(y_in[:,i])
        i=i+1
        if i >= s+b:
            break

    return x, y



def create_batch_2(ar_in,win_len,c):
    win = np.zeros((257,win_len))
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



def boilerplate(q,p):
    q=np.array(q)
    p=np.array(p)
    q = np.expand_dims(q, axis=3)

    return q,p


def prepro(batch):
    start=batch*200
    n=[]
    k=[]
    for i in range(0,200):
        q, p = inputs2(X_train_minmax,mask_irm,i+start,1,WIN_LEN)
        n.append(q)
        k.append(p)

    n=np.array(n)
    k=np.array(k)
    n= np.squeeze(n, axis=1)
    n = np.expand_dims(n, axis=3)
    k= np.squeeze(k,axis=1)

    return n,k

def prepro_test(batch):
    start=batch*1000
    if start >=150000:
        start=np.ceil((start/950)*3).astype(int)

    n=[]
    k=[]
    for i in range(0,1000):
        q, p = inputs2(X_test_minmax,mask_irm_test,i+start,1,WIN_LEN)
        n.append(q)
        k.append(p)

    n=np.array(n)
    k=np.array(k)
    n= np.squeeze(n, axis=1)
    n = np.expand_dims(n, axis=3)
    k= np.squeeze(k,axis=1)

    return n,k



def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
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

checkpoint_path2 = "training1/60cnn_sd.ckpt"

callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)]



if NET1==True:
    print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")
    noise_fft = keras.Input((257,WIN_LEN))


    group=[1]*128
    sum_of_bins=0
    ceil_bins=list(np.full(127,2))
    for k in range(0,len(ceil_bins)):
        print(k)
        ## FFT Bins getting split for processing with specific neurons
        sum_of_bins=sum_of_bins+ceil_bins[k]
        if k==0:
            group[k]= Lambda(lambda x: x[:,0:2,:], output_shape=((2,WIN_LEN)))(noise_fft)
            print(group[k])
        if k==127:
            group[k]= Lambda(lambda x: x[:,248:,:], output_shape=((3,WIN_LEN)))(noise_fft)
        else:
            print(int(sum_of_bins+ceil_bins[k]))
            group[k]=Lambda(lambda x: x[:,int(sum_of_bins):int(sum_of_bins+ceil_bins[k]),:], output_shape=((int(ceil_bins[k]),WIN_LEN)))(noise_fft)
            print(group[k])


    for e in range(0,len(ceil_bins)):
         group[e]=tf.keras.layers.Conv1D(64, 4, strides=1, padding='same',dilation_rate=1, activation='relu')(group[e])
         group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])
         group[e]=tf.keras.layers.Conv1D(64, 8, strides=1, padding='same',dilation_rate=2, activation='relu')(group[e])
         group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])
         group[e]=tf.keras.layers.Conv1D(64, 16, strides=1, padding='same',dilation_rate=4, activation='relu')(group[e])
         group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])

    for j in range(0,len(ceil_bins)):
         group[j]=tf.keras.layers.GlobalAveragePooling1D()(group[j])
        for b in range(0,len(ceil_bins)):
         group[b]=tf.expand_dims(group[b],1)
    for i in range(0,len(ceil_bins)):
         group[i]=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(group[i])

    x_Tensor = group[0]

    for g in range(1,60):
        x_Tensor = Concatenate(axis=1)([x_Tensor,group[g]])


    x_Tensor = Dense(60*64, activation='relu')(x_Tensor)
    x = tf.keras.layers.Dropout(0.05)(x_Tensor)
    outputs = tf.keras.layers.Dense(257, activation='sigmoid')(x)
    model = tf.keras.Model(noise_fft, outputs)



if NET2==True:
        BINS=257

        print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")
        ceil_bins=joblib.load("ceil_bins3.pkl")
        ceil_bins=list(ceil_bins)
        noise_fft = keras.Input((BINS,WIN_LEN))


        group=[1]*60
        sum_of_bins=0
        ceil_bins[59]=9

        for k in range(0,len(ceil_bins)):
            print(k)
            ## FFT Bins getting split for processing with specific neurons
            sum_of_bins=sum_of_bins+ceil_bins[k]
            if k==0:
                group[k]= Lambda(lambda x: x[:,0:2,:], output_shape=((2,WIN_LEN)))(noise_fft)
                print(group[k])
            if k==59:
                print( Lambda(lambda x: x[248:,:], output_shape=((9,16)))(noise_fft))
                group[k]= Lambda(lambda x: x[:,248:,:], output_shape=((9,16)))(noise_fft)
            else:
                print(int(sum_of_bins+ceil_bins[k]))
                group[k]=Lambda(lambda x: x[:,int(sum_of_bins):int(sum_of_bins+ceil_bins[k]),:], output_shape=((int(ceil_bins[k]),WIN_LEN)))(noise_fft)
                print(group[k])
        for e in range(0,len(ceil_bins)):

            group[e]=tf.keras.layers.Conv1D(64, 4, strides=1, padding='same',dilation_rate=1, activation='relu')(group[e])
            group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])
            group[e]=tf.keras.layers.Conv1D(64, 8, strides=1, padding='same',dilation_rate=2, activation='relu')(group[e])
            group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])
            group[e]=tf.keras.layers.Conv1D(64, 16, strides=1, padding='same',dilation_rate=4, activation='relu')(group[e])
            group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])

        for j in range(0,len(ceil_bins)):
            group[j]=tf.keras.layers.GlobalAveragePooling1D()(group[j])
        for b in range(0,len(ceil_bins)):
            group[b]=tf.expand_dims(group[b],1)
        for i in range(0,len(ceil_bins)):
            group[i]=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(group[i])


        x_Tensor = group[0]

        for g in range(1,60):
            x_Tensor = Concatenate(axis=1)([x_Tensor,group[g]])




        x_Tensor = Dense(60*64, activation='relu')(x_Tensor)
        x = tf.keras.layers.Dropout(0.05)(x_Tensor)
        outputs = tf.keras.layers.Dense(257, activation='sigmoid')(x)
        model = tf.keras.Model(noise_fft, outputs)



model.compile(keras.optimizers.Adam(),
       loss='mean_squared_error',
       metrics=['mae'])
testmetrics=[]
trainmetrics=[]

if TRAIN==1:
    """
    SD-training Strategy using model.fit on subdataset with learning rate decay
    saving net after every 50 sub datasets
    """
    print(model.summary())
    #model.load_weights("60cnn_oned_32win_adam2350.h5")
    for q in range(0,2400):
        print(q)
        n,k=prepro(q)
        n= np.squeeze(n,axis=3)
        model.fit(n,k, epochs=10,callbacks=callbacks)

        if (q%50)==0:
            model.save_weights("60cnn_SD"+str(q)+".h5")
            joblib.dump(testmetrics,"60cnn_SD"+str(q)+".pkl")
        if (q%1)==0:
            n,k = prepro_test(q)
            n= np.squeeze(n,axis=3)
            results= model.evaluate(n,k,verbose=1)
            print('test loss, test acc:', results)
            testmetrics.append(results)
if TEST==1:

    """
    Infering, see documentation in train_infer.py
    """
    #model= tf.keras.models.load_model("resnet_version2_40blocks_32WIN_adam2350.h5")
    model.load_weights("60cnn_oned_32win_adam2350.h5")


    print(model.summary())
    BATCH_LEN=800
    START_OUT=40
    END_OUT=60
    for h in range(START_OUT,END_OUT):
        print(".", end='', flush=True)
        n,ph = inputs2(X_test_minmax,PX_test,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
        n = np.array(n)
        n = np.expand_dims(n, axis=3)
        pre = model.predict(n)


        plot = pre
        plot=plot*80
        plot=plot-80
        import librosa
        import librosa.display
        fig = plt.figure()
        librosa.display.specshow(np.transpose(plot), y_axis='log', x_axis="time", sr=16000,hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/mask_res_'+str(h)+'.png', dpi=fig.dpi)
        pre = np.divide(1, pre, out=np.ones_like(pre), where=pre!=0)
        pre=np.transpose(pre)
        #pre[pre>=1.2] *= 1.5
        out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*pre
        noisy_out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
        spec_n = noisy_out

        ph=np.array(ph)
        ph= np.transpose(ph)
        out= librosa.db_to_amplitude(out)
        noisy_out= librosa.db_to_amplitude(noisy_out)


        out = out*ph
        noisy_out = noisy_out*ph

        fig = plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(out), y_axis='log', x_axis="time", sr=16000, hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/clean_res'+str(h)+'.png', dpi=fig.dpi)

        fig = plt.figure()
        librosa.display.specshow(spec_n, y_axis='log', x_axis="time", sr=16000, hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/noisy_res'+str(h)+'.png', dpi=fig.dpi)

        out = librosa.istft(out)
        noisy_out=librosa.istft(noisy_out)

        out = librosa.util.normalize(out)
        noisy_out = librosa.util.normalize(noisy_out)

        import scipy
        scipy.io.wavfile.write('predict/clean_res_'+str(h)+'.wav',16000,out)
        scipy.io.wavfile.write('predict/noisy_res_'+str(h)+'.wav',16000,noisy_out)

    print(str(END_OUT-START_OUT)+' clean/noisy spectrograms/wavs and masks saved!')
