"""
Res CNN Training on SD-Strategy
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
from keras.utils import normalize, to_categorical
from keras.layers import BatchNormalization

"""
SD Training Strategy, all necessary functions to train/test are within this file.
Processing functions are documented in spnn.py
For extensive code annotations see train_infer.py
"""

TRAIN=0
TEST=1
WIN_LEN=16

RESNETV1=0
RESNETV3=1

global lr0
lr0=0.001

def scheduler(epoch):
  """
  LearningRateScheduler as mentioned in thesis.
  """
  global lr0
  if epoch <= 1:
    lr0 = 0.001
    return 0.001
  else:
    return 0.001*(1/(1+epoch))

load_full_set=False

if load_full_set==False:
    print("Loading Datasets: ")
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



def IRM2(S,N):
    #Hier wird die divide funktion benutzt um ein Teilen durch 0 zu verhindern. Dies führt zu ungewollten Peaks in der Wav ausgabe
    M = []

    for i in range(len(S)):
        c = np.divide(S[i], N[i], out=np.ones_like(S[i]), where=N[i]!=0)
        M.append(c)

    return M


if TRAIN==1:
    mask_irm = IRM2(yta,xta)
    mask_irm_test = IRM2(yte,xte)
    mask_irm = np.array(mask_irm)
    mask_irm_test = np.array(mask_irm_test)
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

checkpoint_path2 = "training1/resnet_hidyn.ckpt"

callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)]

if RESNETV1==True:
    inputs = keras.Input(shape=(257, WIN_LEN,1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    num_res_net_blocks = 40
    for i in range(num_res_net_blocks):
      x = res_net_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(257, activation='sigmoid')(x)

    res_net_model = keras.Model(inputs, outputs)
    res_net_model.compile(keras.optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=['mae'])


if RESNETV3==True:

    inputs = keras.Input(shape=(257, WIN_LEN,1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    num_res_net_blocks = 20
    for i in range(num_res_net_blocks):
      x = res_net_block(x, 64, 3)

    x = layers.Conv2D(257, 1, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(257, activation='sigmoid')(x)

    res_net_model = keras.Model(inputs, outputs)

    res_net_model.compile(keras.optimizers.RMSprop(),
                  loss='mean_squared_error',
                  metrics=['mae'])

else:

    inputs = keras.Input(shape=(257, WIN_LEN,1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    num_res_net_blocks = 40
    for i in range(num_res_net_blocks):
      x = res_net_block(x, 64, 3)

    x = layers.Conv2D(257, 1, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(257, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(257, activation='sigmoid')(x)

    res_net_model = keras.Model(inputs, outputs)
    res_net_model.compile(keras.optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=['mae'])
testmetrics=[]
trainmetrics=[]

if TRAIN==1:
    """
    SD-training Strategy using model.fit on subdataset with learning rate decay
    saving net after every 50 sub datasets
    """
    print(res_net_model.summary())
    for q in range(0,2400):
        print(q)
        n,k=prepro(q)
        res_net_model.fit(n,k, epochs=10,callbacks=callbacks)

        if (q%50)==0:
            res_net_model.save("resnet_version3_40blocks_32WIN_adam"+str(q)+".h5")
            joblib.dump(testmetrics,"testmetrics"+str(q)+".pkl")
        if (q%1)==0:
            """
            savint train, test metrics
            """
            n,k = prepro_test(q)
            results= res_net_model.evaluate(n,k,verbose=1)
            print('test loss, test acc:', results)
            testmetrics.append(results)
if TEST==1:
    """
    Infering, see documentation in train_infer.py
    """

    model= tf.keras.models.load_model("resnet_version3_40blocks_32WIN_adam2350.h5")
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='res_net_model.png', show_shapes=True)
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
