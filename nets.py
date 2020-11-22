"""
Neural Network architectures used in the thesis. only the most important are shown here for clarity.
a file with all used architectures can be found in the experimental subfolder

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from spnn import *
from keras.models import model_from_json
from nets import *
from scipy import signal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Permute, Conv2D,MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, MaxPool2D,MaxPool1D,AvgPool1D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from spnn import *
import keras.backend as K


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x


class neural_net:
    """
    This class returns a neural network architecture, you can specify the optimizer, loss,
    contextwindow length, number of output bins and saved metrics
    """

    net_type = None
    WIN_LEN = None
    optimizer= None
    loss=None
    metrics= None
    BINS=None
    scaler=None

    def __init__(self, net_type,BINS,WIN_LEN,optimizer,loss,metrics):
        self.net_type=net_type
        self.WIN_LEN=WIN_LEN
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.BINS=BINS


    def get_optimizer(self,optimizer):
        if optimizer=="RMSprop":
            return keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9)
        if optimizer=="adam":
            return keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        if optimizer=="SGD":
            return keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
        if optimizer=="adamax":
            return keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        else:
            print("Optimizer not found, falling back to default RMSprop.....")
            return keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9)

    """NETWORK ARCHITECTURES"""

    def densenet(self,BINS,WIN_LEN, f=32):
      repetitions = 6, 12, 24, 16

      def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = layers.BatchNormalization()(x)
        x= keras.activations.relu(x)
        x = layers.Conv2D(f, k, strides=s, padding=p)(x)
        return x


      def dense_block(tensor, r):
        for _ in range(r):
          x = bn_rl_conv(tensor, 4*f)
          x = bn_rl_conv(x, f, 3)
          tensor = Concatenate()([tensor, x])
        return tensor


      def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x

      noise_fft = keras.Input((BINS,WIN_LEN,1))
      x = layers.Conv2D(64, 7, strides=2, padding='same')(noise_fft)
      x = MaxPool2D(3, strides=2, padding='same')(x)

      for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)

      x = GlobalAvgPool2D()(d)

      output = Dense(257, activation='sigmoid')(x)

      model = Model(noise_fft, output)
      return model


    def dense_resnet(self,BINS,WIN_LEN, f=32):
      def res_net_block(input_data, filters, conv_size):
        x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, input_data])
        x = layers.Activation('relu')(x)
        return x

      repetitions = 6, 12, 24, 16

      def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = layers.BatchNormalization()(x)
        x= keras.activations.relu(x)
        x = layers.Conv2D(f, 1, strides=s, padding=p)(x)
        x= res_net_block(x,f,k)
        return x


      def dense_block(tensor, r):
        for _ in range(r):
          x = bn_rl_conv(tensor, 4*f)
          x = bn_rl_conv(x, f, 3)
          tensor = Concatenate()([tensor, x])
        return tensor


      def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x

      noise_fft = keras.Input((BINS,WIN_LEN,1))


      x = layers.Conv2D(64, 7, strides=2, padding='same')(noise_fft)
      x = MaxPool2D(3, strides=2, padding='same')(x)

      for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)

      x = GlobalAvgPool2D()(d)

      output = Dense(257, activation='sigmoid')(x)

      model = Model(noise_fft, output)
      return model



    def fully_connected(self,BINS,WIN_LEN):
        inputs = keras.Input(shape=(BINS, WIN_LEN))
        x = layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dense(257*8)(x)
        x = tf.keras.layers.Dense(257*4)(x)
        #x = tf.keras.layers.Dense(257)(x)
        x = tf.keras.layers.Dense(257*4)(x)
        #x = tf.keras.layers.Dense(257*8)(x)
        #x = layers.Dropout(0.01)(x)
        outputs = layers.Dense(257, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)

        return model


    def resnet_baseline(self,BINS,WIN_LEN):
        inputs = keras.Input(shape=(BINS, WIN_LEN,1))
        #inputs=tf.expand_dims(inputs,2)
        x = layers.Conv2D(64, 3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        num_res_net_blocks = 40
        for i in range(num_res_net_blocks):
          x = res_net_block(x, 32, 3)

        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(257, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(257, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(BINS, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)

        return model


    def resnet_baseline64(self,BINS,WIN_LEN):
        inputs = keras.Input(shape=(BINS, WIN_LEN,1))
        #inputs=tf.expand_dims(inputs,2)
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3)(x)

        res_net_quantity = 40
        for i in range(res_net_quantity):
          x = res_net_block(x, 64, 3)

        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(257, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(BINS, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)

        return model


    def bidi_symmetric(self,BINS,WIN_LEN):
        inputs = keras.Input(shape=(257, WIN_LEN))
        x=tf.keras.layers.Permute((2,1))(inputs)
        x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True))(x)
        x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True))(x)
        x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512))(x)
        outputs=tf.keras.layers.Dense(60, activation=tf.nn.sigmoid)(x)
        model = tf.keras.Model(inputs, outputs)

        return model



    def cnn_oned(self,BINS,WIN_LEN):

            """
            Subband-D-DNN-LSTM Net with 23 Bands
            """

            print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")

            ceil_bins=joblib.load("ceil_bins.pkl")
            ceil_bins=list(ceil_bins)

            noise_fft = keras.Input((BINS,WIN_LEN))

            """Split up Subbands from STFT"""

            group=[1]*23
            sum_of_bins=0
            ceil_bins[22]=56

            for k in range(0,len(ceil_bins)):
                print(k)
                ## FFT Bins getting split for processing with specific neurons
                sum_of_bins=sum_of_bins+ceil_bins[k]
                if k==0:
                    group[k]= Lambda(lambda x: x[:,0:2], output_shape=((2,WIN_LEN)))(noise_fft)
                    print(group[k])
                if k==22:
                    print("K=22")
                    print( Lambda(lambda x: x[:,201:], output_shape=((56,WIN_LEN)))(noise_fft))
                    group[k]= Lambda(lambda x: x[:,201:], output_shape=((56,WIN_LEN)))(noise_fft)
                else:
                    print(int(sum_of_bins+ceil_bins[k]))
                    group[k]=Lambda(lambda x: x[:,int(sum_of_bins):int(sum_of_bins+ceil_bins[k])], output_shape=((int(ceil_bins[k]),WIN_LEN)))(noise_fft)
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


            x_Tensor = Concatenate(axis=1)([group[0],group[1],group[2],group[3],group[4],group[5],group[6],group[7],group[8], \
                                  group[9],group[10],group[11],group[12],group[13],group[14],group[15],group[16],group[17],\
                                  group[18],group[19],group[20],group[21],group[22]])


            x_Tensor = Dense(23*64, activation='relu')(x_Tensor)
            x = tf.keras.layers.Dropout(0.05)(x_Tensor)
            outputs = tf.keras.layers.Dense(257, activation='sigmoid')(x)
            model = tf.keras.Model(noise_fft, outputs)

            return model


    def cnn_oned_60(self,BINS,WIN_LEN):

            """
            Subband-D-DNN-LSTM Net with 60 Bands
            """

            print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")
            ceil_bins=joblib.load("ceil_bins3.pkl")
            ceil_bins=list(ceil_bins)
            #when using customLoss squeeze axis 3:
            noise_in = keras.Input((BINS,WIN_LEN,1))
            noise_fft=tf.squeeze(noise_in,3)

            """Split up Subbands from STFT"""
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

            """
            Concatenate Feature Vectors, init x_Tensor as first element
            """
            x_Tensor = group[0]
            for g in range(1,60):
                x_Tensor = Concatenate(axis=1)([x_Tensor,group[g]])


            x_Tensor = Dense(60*64, activation='relu')(x_Tensor)
            x = tf.keras.layers.Dropout(0.05)(x_Tensor)
            outputs = tf.keras.layers.Dense(257, activation='sigmoid')(x)
            model = tf.keras.Model(noise_in, outputs)


            return model



    def cnn_oned_60_pesqloss(self,BINS,WIN_LEN):
            print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")
            ceil_bins=joblib.load("ceil_bins3.pkl")
            ceil_bins=list(ceil_bins)


            noise_in = keras.Input((BINS,WIN_LEN,1))
            noise_fft=tf.squeeze(noise_in,3)
            passthrough_noisefft= noise_fft
            noise_fft=tf.keras.layers.BatchNormalization()(noise_fft)



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

            ## last frame of context window:
            curframe = passthrough_noisefft[:,:,-1]
            # reverse db to power spektrum
            stftoutput =  tf.math.multiply(tf.pow(10.0,(tf.math.divide(curframe,10.0))),outputs)
            model = tf.keras.Model(noise_in, [outputs,stftoutput])

            return model



    def cnn_oned_60_customloss(self,BINS,WIN_LEN):
            print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")
            ceil_bins=joblib.load("ceil_bins3.pkl")
            ceil_bins=list(ceil_bins)


            noise_in = keras.Input((BINS,WIN_LEN,1))
            pre_in= keras.Input((BINS,WIN_LEN,1))

            noise_fft=tf.squeeze(noise_in,3)
            pre_fft=tf.squeeze(pre_in,3)

            group=[1]*60
            sum_of_bins=0
            ceil_bins[59]=9

            group2=[1]*60
            sum_of_bins2=0

            for k in range(0,len(ceil_bins)):
                print(k)
                ## FFT Bins getting split for processing with specific neurons
                sum_of_bins2=sum_of_bins2+ceil_bins[k]
                if k==0:
                    group2[k]= Lambda(lambda x: x[:,0:2,:], output_shape=((2,WIN_LEN)))(pre_fft)
                    print(group[k])

                if k==59:

                    print( Lambda(lambda x: x[248:,:], output_shape=((9,32)))(pre_fft))
                    group2[k]= Lambda(lambda x: x[:,248:,:], output_shape=((9,32)))(pre_fft)
                else:
                    print(int(sum_of_bins+ceil_bins[k]))
                    group2[k]=Lambda(lambda x: x[:,int(sum_of_bins):int(sum_of_bins+ceil_bins[k]),:], output_shape=((int(ceil_bins[k]),WIN_LEN)))(pre_fft)
                    print(group2[k])


            for e in range(0,len(ceil_bins)):
                group2[e]=tf.keras.layers.Conv1D(64, 4, strides=1, padding='same',dilation_rate=1, activation='relu')(group2[e])
                group2[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group2[e])
                group2[e]=tf.keras.layers.Conv1D(64, 8, strides=1, padding='same',dilation_rate=2, activation='relu')(group2[e])
                group2[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group2[e])
                group2[e]=tf.keras.layers.Conv1D(64, 16, strides=1, padding='same',dilation_rate=4, activation='relu')(group2[e])
                group2[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group2[e])

            for j in range(0,len(ceil_bins)):
                group2[j]=tf.keras.layers.GlobalAveragePooling1D()(group2[j])

            for b in range(0,len(ceil_bins)):
                group2[b]=tf.expand_dims(group2[b],1)

            print("50 % net")


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

            for g in range(1,60):
                group[g] = Concatenate(axis=1)([group[g],group2[g]])


            for i in range(0,len(ceil_bins)):
                group[i]=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(group[i])

            x_Tensor = group[0]

            for g in range(1,60):
                x_Tensor = Concatenate(axis=1)([x_Tensor,group[g]])


            x_Tensor = Dense(60*64, activation='relu')(x_Tensor)
            x = tf.keras.layers.Dropout(0.05)(x_Tensor)
            outputs = tf.keras.layers.Dense(257, activation='sigmoid')(x)
            model = tf.keras.Model([noise_in,pre_in], outputs)
            return model



    def cnn_60_freqax(self,BINS,WIN_LEN):
            print("Attention: The current TF Version requirs weights to be saved seperatly in sparsly connected Nets")
            ceil_bins=joblib.load("ceil_bins3.pkl")
            ceil_bins=list(ceil_bins)
            noise_fft = keras.Input((BINS,WIN_LEN))

            infeat=tf.keras.layers.Permute((2,1))(noise_fft)

            ceil_bins=list(np.full(15,2))


            group=[1]*15
            sum_of_bins=0
            ceil_bins[14]=2

            for k in range(0,len(ceil_bins)):
                print(k)
                ## FFT Bins getting split for processing with specific neurons
                sum_of_bins=sum_of_bins+ceil_bins[k]
                if k==0:
                    group[k]= Lambda(lambda x: x[:,0:2,:], output_shape=((257,2)))(infeat)
                    print(group[k])
                if k==15:
                    group[k]= Lambda(lambda x: x[:,34:,:], output_shape=((257,2)))(infeat)
                else:
                    print(int(sum_of_bins+ceil_bins[k]))
                    group[k]=Lambda(lambda x: x[:,int(sum_of_bins):int(sum_of_bins+ceil_bins[k]),:], output_shape=((257,2)))(infeat)
                    print(group[k])

            for k in range(0,len(ceil_bins)):
                group[k]=tf.keras.layers.Permute((2,1))(group[k])
                print(group[k])

            for e in range(0,len(ceil_bins)):

                group[e]=tf.keras.layers.Conv1D(64, 2, strides=1, padding='same',dilation_rate=1, activation='relu')(group[e])
                group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])
                group[e]=tf.keras.layers.Conv1D(64, 4, strides=1, padding='same',dilation_rate=2, activation='relu')(group[e])
                group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])
                group[e]=tf.keras.layers.Conv1D(64, 8, strides=1, padding='same',dilation_rate=4, activation='relu')(group[e])
                group[e]=tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='same', data_format=None)(group[e])

            for j in range(0,len(ceil_bins)):
                group[j]=tf.keras.layers.GlobalAveragePooling1D()(group[j])
            for b in range(0,len(ceil_bins)):
                group[b]=tf.expand_dims(group[b],1)
            for i in range(0,len(ceil_bins)):
                group[i]=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(group[i])

            x_Tensor = group[0]

            for g in range(1,len(ceil_bins)):
                x_Tensor = Concatenate(axis=1)([x_Tensor,group[g]])

            outputs = tf.keras.layers.Dense(257, activation='sigmoid')(x_Tensor)
            model = tf.keras.Model(noise_fft, outputs)

            return model




    def return_model(self):
        if self.net_type=="resnet_baseline":
            model=self.resnet_baseline(self.BINS,self.WIN_LEN)
        if self.net_type=="1d_cnn":
           model=self.cnn_oned(self.BINS, self.WIN_LEN)
        if self.net_type=="cnn_oned_large":
           model=self.cnn_oned_large(self.BINS, self.WIN_LEN)
        if self.net_type=="bidi_symmetric":
           model=self.bidi_symmetric(self.BINS, self.WIN_LEN)
        if self.net_type=="fully_connected":
           model=self.fully_connected(self.BINS, self.WIN_LEN)
        if self.net_type=="densenet":
           model=self.densenet(self.BINS, self.WIN_LEN)
        if self.net_type=="resnet_baseline64":
           model=self.resnet_baseline64(self.BINS, self.WIN_LEN)
        if self.net_type=="cnn_oned_60":
           model=self.cnn_oned_60(self.BINS, self.WIN_LEN)



        model.compile(self.get_optimizer(self.optimizer),loss=self.loss,metrics=self.metrics)
        return model
