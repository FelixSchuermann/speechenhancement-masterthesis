
"""
Train and Infer Neural Networks for Speech Enhancement
- generator strategy
- net class loader


@author: Felix Schürmann, Masters Thesis on deep learning methods for speech enhancement
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
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
from keras.utils import normalize, to_categorical
from keras.layers import BatchNormalization
from spnn import *
from keras.models import model_from_json
from nets import *
from keras.models import Sequential
from scipy import signal
from tensorflow.keras import regularizers
from utils import my_utils as myut
from sklearn.utils import class_weight
from utils import pmsqe as pmsqe

from keras.utils import plot_model
from keras.utils.vis_utils import plot_model

cwd = os.getcwd()
"""
specify context window length and mask type
if ibm mask is used : ibm=True
"""
ibm=False
WIN_LEN=32


"""
multiple program functions are used in this file, use training, testing
or compare infered stft spectrogramms
"""

TRAIN=1
TEST=0
IBMINFER=0
CMP=0
APRIOINFER=0


def custom_loss_pes(y_true,y_pred):
    """
    defining a function for the PMSQE Loss.
    'we' reduces the impact of the PMSQE loss, to build a relation to the additional MSE loss
    """

    we=0.002
    pmsqe.init_constants(Fs=16000, Pow_factor=pmsqe.perceptual_constants.Pow_correc_factor_Hann, apply_SLL_equalization=True,apply_bark_equalization=True, apply_on_degraded=True, apply_degraded_gain_correction=True)
    pmsqe_loss = pmsqe.per_frame_PMSQE(y_true, y_pred, alpha = 0.1)
    return K.mean(we*(pmsqe_loss))


if TRAIN==1:
    # additional tensors for use in experimental nets
    #mytens = tf.ones((257), tf.float32)
    #mytens = tf.Variable(mytens)
    #lastframe = tf.reshape(mytens, [257])



    """ specify which net should be loaded, naming is done in nets.py """

    NET_TYPE="cnn_oned_60"

    """
    configuration of the net with class "neural_net", you can change the NET_TYPE, Number of output BINS, context window length,
    loss function,optimizer, as well as output metrics
    """

    net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
    #net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss=["mean_squared_error","binary_crossentropy"],metrics=["mae","acc"])
    #net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss=["mean_squared_logarithmic_error","binary_crossentropy"],metrics=["mae","acc"])

    """load model and pass into object"""
    model=net1.return_model()

    """do weights have to be loaded?"""
    #model.load_weights(cwd+"/lsmtnurbark20.h5")
    #model.load_weights("/media/hdd/database/de/media/hdd/tmp/"+str(NET_TYPE))

    """draw the architecture of the specified model to file"""
    tf.keras.utils.plot_model(model, to_file=str(NET_TYPE)+'_.png', show_shapes=True)

    """define checkpoint callback"""
    filepath=os.getcwd()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath+"/tmp/"+str(NET_TYPE),save_weights_only=True,monitor='val_mse',mode='max',save_best_only=False)


    print(model.summary())

    """define learn rate scheduler if needed"""
    def scheduler(epoch):
        if epoch <= 1:
            return 0.001
        else:
            #print(0.001 * tf.math.exp(0.1 * (10 - epoch)))
            return 0.001 * (0.1 * (10-epoch))

    lrsched= tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

    """Datasets are numbered, so that the training can be extended with a different dataset,
        standard dataset is #11 """

    for j in range(0,10):
        for q in range(11,13):

            """DataLoader returns all important input data, as well as training targets"""
            X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax, mask, mask_test,PX_train,PX_test,_,_, mel_xta,mel_xte,mel_target,mel_target_test,_,yte,_,yta,vada,vade = dataLoader(q,ibm)


            """Additional processing, clipping, transposing"""
            mask=mask.astype(np.float32)
            mask_test = mask_test.astype(np.float32)
            mask= np.clip(mask,0,1)
            mask_test= np.clip(mask_test,0,1)
            mask = np.transpose(mask)
            mask_test=np.transpose(mask_test)



            """Do Training with generator function, save metrics in history variable
                therefore pass input data, batch_size, contextwindow length and a Bool for dimension extension
                addtionaly set steps_per_epoch aswell as number of epochs"""


            history= model.fit_generator(batch_gen_standard(X_train_minmax,mask,WIN_LEN,200,False),steps_per_epoch=2400, epochs=10, validation_data=batch_gen_standard(X_test_minmax,mask_test,WIN_LEN,200,False), validation_steps=200, callbacks=[model_checkpoint_callback])
            #second output:
            #history= model.fit_generator(batch_gen_multiout(X_train_minmax,mask,vada,WIN_LEN,20,False),steps_per_epoch=24000, epochs=10, validation_data=batch_gen_multiout(X_test_minmax,mask_test,vade,WIN_LEN,20,False), validation_steps=1000, callbacks=[model_checkpoint_callback])

            """save weights after training"""
            model.save_weights(str(NET_TYPE)+str(j)+str(q)+".h5")



            """Plot Metrics after Training"""
            fig = plt.figure()

            plt.plot(history.history['mean_absolute_error'])
            plt.plot(history.history['val_mean_absolute_error'])

            #plt.plot(history.history['dense_1_mean_absolute_error'])
            #plt.plot(history.history['val_dense_1_mean_absolute_error'])
            #plt.plot(history.history['dense_acc'])
            #plt.plot(history.history['val_dense_acc'])
            #plt.plot(history.history['tf_op_layer_Mul_loss'])
            #plt.plot(history.history['val_dense_1_loss'])
            #plt.plot(history.history['val_tf_op_layer_Mul_loss'])

            plt.title('Model accuracy')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            #plt.legend(['MAE', 'VAL_MAE','ACC','VAL_ACC'], loc='upper left')
            fig.savefig(str(NET_TYPE)+str(j)+str(q)+'.png', dpi=fig.dpi)

            """Clear up RAM for secure loading of next database file"""

            del history
            del X_train_minmax
            del X_test_minmax
            del mask
            del mask_test
            del y_train_minmax
            del y_test_minmax

            import gc
            gc.collect()


if TEST==1:
    # ceil_bins defines bark cutoff frequencies on stft bins
    ceil_bins=joblib.load("ceil_bins2.pkl")

    """specify length(BATCH_LEN),start(h) of inference from test dataset (q)"""
    h=2
    BATCH_LEN=10000
    q=11
    WIN_LEN=32

    """NET configuration and loading"""

    print("Loading Model & Weights..")

    NET_TYPE="resnet_baseline64"
    net_regression=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
    regression=net_regression.return_model()
    print(regression.summary())
    regression.load_weights(cwd+"/models/resnet_40blocks_32WIN_adam2350.h5")


    """DataLoader for Inference"""

    print("Loading Data for Inference")
    X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax, mask, mask_test,PX_train,PX_test,_,_, _,_,_,_,xte,yte,xta,_,_,_ = dataLoader(q,ibm)
    print("done")

    """Generate Input Batch for Inference"""

    n,ph = inputs2(X_test_minmax,PX_test,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
    n=np.array(n)
    #optional dimension expansion depening on NET_TYPE:
    n = np.expand_dims(n,axis=3)


    print("Infering..")
    reg=regression.predict(n,verbose=1)
    #if NN has two outputs:
    #reg,ibm=regression.predict(n,verbose=1)
    #print(reg)


    plot = reg
    gain= reg
    """optional post-gain"""

    reg = np.power(reg,1.5)
    ## regression values are between 0 and 1, flipping them for multiplication in dB domain..
    ## multiplication of the gain mask in dB domain:
    reg = np.divide(1, reg, out=np.ones_like(reg), where=reg!=0)
    ## clipping values
    reg = np.clip(reg, a_min = 1, a_max = 100)


    #spectrogramm scaling
    plot=plot*80
    plot=plot-80


    """Plot infered Gain-Mask"""
    import librosa
    import librosa.display
    fig = plt.figure()
    librosa.display.specshow(np.transpose(plot), y_axis='log', x_axis="time", sr=16000,hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/mask_'+str(NET_TYPE)+str(h)+'.png', dpi=fig.dpi)


    #transpose for multiplication
    reg=np.transpose(reg)


    """Applying Gain Mask and reconstructing mixed Signal,
        aswell as undisturbed original in STFT Domain"""

    out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*reg
    noisy_out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    original = yte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]

    #tranpose phase for matching multiplication
    ph=np.array(ph)
    ph= np.transpose(ph)

    # clip infered stft from -80 to 0 dB
    out = np.clip(out, a_min = -80, a_max = 0)

    """Plot infered and original spectrogramm"""

    fig = plt.figure()
    librosa.display.specshow(out, y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/clean_'+str(NET_TYPE)+str(h)+'.png', dpi=fig.dpi)

    fig = plt.figure()
    librosa.display.specshow(original, y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/original'+str(NET_TYPE)+str(h)+'.png', dpi=fig.dpi)


    """Convert from dB Domain back to amplitude"""
    out= librosa.db_to_amplitude(out)
    noisy_out= librosa.db_to_amplitude(noisy_out)
    original = librosa.db_to_amplitude(original)
    """Apply phase"""

    out = out*ph
    noisy_out = noisy_out*ph
    original = original*ph



    spec_n = noisy_out
    """Plot Noisy spectrogramm"""
    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(spec_n), y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/noisy_'+str(NET_TYPE)+str(h)+'.png', dpi=fig.dpi)


    """Apply inverse STFT"""
    out = librosa.istft(out)
    noisy_out=librosa.istft(noisy_out)
    original = librosa.istft(original)

    """normalize Output Signal"""
    out = librosa.util.normalize(out)
    noisy_out = librosa.util.normalize(noisy_out)
    original = librosa.util.normalize(original)

    """Export WAV Files of enhanced speech, disturbed and original"""
    import scipy
    scipy.io.wavfile.write('predict/clean_'+str(NET_TYPE)+str(h)+'.wav',16000,out)
    scipy.io.wavfile.write('predict/noisy_'+str(NET_TYPE)+str(h)+'.wav',16000,noisy_out)
    scipy.io.wavfile.write('predict/original_'+str(NET_TYPE)+str(h)+'.wav',16000,original)


if IBMINFER==1:
    """Inferring for IBM only Model"""

    """specify length(BATCH_LEN),start(h) of inference from test dataset (q)"""
    h=0
    BATCH_LEN=100000
    q=11
    ibm=True
    WIN_LEN=64

    """DataLoader for Inference"""
    X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax, mask, mask_test,PX_train,PX_test,_,_, mel_xta,mel_xte,mel_target,mel_target_test,_,_,_ = dataLoader(q,ibm)


    print('Loading IBM Prediction Model')

    #model_ibm= tf.keras.models.load_model("bidi_symmetric3.h5")
    NET_TYPE="bidi_symmetric"
    net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(),metrics=["acc"])
    model_ibm=net1.return_model()
    print(model_ibm.summary())
    print("Loading model weights..")
    model_ibm.load_weights("bidi_symmetric211bidisymm_1106.h5")
    print("done!")


    """Get Batch for Inference"""
    n,_ = inputs2(X_test_minmax,PX_test,h*BATCH_LEN,BATCH_LEN,16)
    n=np.array(n)

    # validate shape:
    n= np.reshape(n,(100000,257,16))

    """IBM Infering"""
    print("Infering..")
    ibm_pre=model_ibm.predict(n,verbose=1)
    print(ibm_pre)


if CMP==1:

    """Infer multiple small comparable wav files and spectrogramms"""
    """Same functionality as TEST==1"""

    NET_TYPE="1d_cnn"
    #net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss=["mean_squared_error",custom_loss_pes],metrics=["mae"])
    WIN_LEN=64
    net1=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])

    model = net1.return_model()
    model.load_weights(cwd+"/models/1d_cnn5.h5")
    print(model.summary())

    q=11
    ibm=False

    #X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax, mask, mask_test,PX_train,PX_test,_,_, mel_xta,mel_xte,mel_target,mel_target_test,xte,yte,xta,yta,_,_ = dataLoader(q,ibm)
    X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax, mask, mask_test,PX_train,PX_test,_,_, _,_,_,_,xte,yte,xta,_,_,_ = dataLoader(q,ibm)

    mask_test=np.transpose(mask_test)
    mask_test = np.clip(mask_test,0,1)
    mask=np.transpose(mask)
    mask = np.clip(mask,0,1)

    BATCH_LEN=800
    START_OUT=10
    END_OUT=20
    for h in range(START_OUT,END_OUT):
        print(".", end='', flush=True)
        n,ph = inputs2(X_test_minmax,PX_test,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
        n = np.array(n)
        #n = np.expand_dims(n, axis=3)
        #pre, prepow = model.predict(n)
        pre = model.predict(n)

        pre = pre**1.4
        #pre[pre>=1.2] **= 2

        plot=pre
        plot=plot*80
        plot=plot-80

        import librosa
        import librosa.display
        fig = plt.figure()
        librosa.display.specshow(np.transpose(plot), y_axis='log', x_axis="time", sr=16000)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/freqax32WIN'+str(h)+'.png', dpi=fig.dpi)


        #pre = 1/pre
        pre = np.divide(1, pre, out=np.ones_like(pre), where=pre!=0)
        pre=np.transpose(pre)
        #pre[pre>=1.2] *= 1.5

        out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]*pre
        noisy_out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
        original = yte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]


        orimask = mask_test[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
        #orimask = mask[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
        orimask=orimask*80
        orimask=orimask-80

        spec_n = noisy_out

        ph=np.array(ph)

        ph= np.transpose(ph)

        out= librosa.db_to_amplitude(out)
        noisy_out= librosa.db_to_amplitude(noisy_out)

        out = out*ph
        noisy_out = noisy_out*ph
        #print(out.shape)

        fig = plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(out), y_axis='log', x_axis="time", sr=16000, hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/'+str(NET_TYPE)+'clean32WIN'+str(h)+'.png', dpi=fig.dpi)

        fig = plt.figure()
        librosa.display.specshow(spec_n, y_axis='log', x_axis="time", sr=16000, hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/'+str(NET_TYPE)+'noisy32WIN'+str(h)+'.png', dpi=fig.dpi)

        fig = plt.figure()
        librosa.display.specshow(original, y_axis='log', x_axis="time", sr=16000, hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/'+str(NET_TYPE)+'original32WIN'+str(h)+'.png', dpi=fig.dpi)

        fig = plt.figure()
        librosa.display.specshow(orimask, y_axis='log', x_axis="time", sr=16000, hop_length=128)
        plt.colorbar(format='%+2.0f dB')
        fig.savefig('predict/'+str(NET_TYPE)+'orimaske32WIN'+str(h)+'.png', dpi=fig.dpi)

        out = librosa.istft(out)
        noisy_out=librosa.istft(noisy_out)

        out = librosa.util.normalize(out)
        noisy_out = librosa.util.normalize(noisy_out)

        import scipy
        scipy.io.wavfile.write('predict/clean_res_'+str(h)+'.wav',16000,out)
        scipy.io.wavfile.write('predict/noisy_res_'+str(h)+'.wav',16000,noisy_out)

    print(str(END_OUT-START_OUT)+' clean/noisy spectrograms/wavs and masks saved!')

if APRIOINFER==1:
    """Infer a-priori estimates"""

    ceil_bins=joblib.load("ceil_bins2.pkl")

    """Data config"""
    h=2
    BATCH_LEN=10000
    q=11

    """Net config"""
    WIN_LEN=32
    NET_TYPE="resnet_baseline64"
    net_regression=neural_net(NET_TYPE,BINS=257,WIN_LEN=WIN_LEN,optimizer="adam",loss="mean_squared_error",metrics=["mae"])
    regression=net_regression.return_model()
    print(regression.summary())
    regression.load_weights("RESNET_aprioriSNR2350.h5")

    """DataLoader"""
    _, _, X_test_minmax, _, mask, mask_test,PX_train,PX_test,_,_, _,_,_,_,xte,yte,xta,_ = dataLoader(q,ibm)

    """Load Batch and Infer"""
    n,ph = inputs2(X_test_minmax,PX_test,h*BATCH_LEN,BATCH_LEN,WIN_LEN)
    n=np.array(n)
    n = np.expand_dims(n,axis=3)
    reg=regression.predict(n,verbose=1)

    """Reverse sigmoidal a-priori mapping for values in dB domain"""
    reg = reverse_aprioMask(reg)
    reg = np.clip(reg,-80,80)
    regm = np.subtract(reg,30)

    #histogramm plot
    myut.histplot(regm)

    """Apply Parametric Wiener Filter"""
    regm_lin=librosa.db_to_power(regm)
    wienergain = WienerGain(regm_lin,alpha=1,beta=1,parametric=True)



    out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    out = librosa.db_to_power(out)
    ## apply wiener gain in power domain
    wienergain= np.transpose(wienergain)
    out=out*wienergain
    out= librosa.power_to_db(out)
    noisy_out = xte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    original = yte[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]

    mask_test=np.transpose(mask_test)
    originalSNR = mask_test[:,h*BATCH_LEN:h*BATCH_LEN+BATCH_LEN]
    myut.histplot(originalSNR)

    """output spectrogramms and wav files as before"""

    ph=np.array(ph)
    ph= np.transpose(ph)

    out = np.clip(out, a_min = -80, a_max = 0)
    fig = plt.figure()

    librosa.display.specshow(out, y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/clean_res'+str(h)+'.png', dpi=fig.dpi)

    fig = plt.figure()
    librosa.display.specshow(original, y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/original_res'+str(h)+'.png', dpi=fig.dpi)


    out= librosa.db_to_amplitude(out)
    noisy_out= librosa.db_to_amplitude(noisy_out)
    original = librosa.db_to_amplitude(original)
    out = out*ph
    noisy_out = noisy_out*ph
    original = original*ph
    #print(out.shape)

    spec_n = noisy_out

    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(spec_n), y_axis='log', x_axis="time", sr=16000, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    fig.savefig('predict/noisy_res'+str(h)+'.png', dpi=fig.dpi)

    out = librosa.istft(out)
    noisy_out=librosa.istft(noisy_out)
    original = librosa.istft(original)

    out = librosa.util.normalize(out)
    noisy_out = librosa.util.normalize(noisy_out)
    original = librosa.util.normalize(original)

    import scipy
    scipy.io.wavfile.write('predict/clean_aprio'+str(h)+'.wav',16000,out)
    scipy.io.wavfile.write('predict/noisy_aprio'+str(h)+'.wav',16000,noisy_out)
    scipy.io.wavfile.write('predict/original_aprio'+str(h)+'.wav',16000,original)
