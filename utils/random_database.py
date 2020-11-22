"""
for creating a spectrum database of Clean|Noisy pairs + noisy phase for speech enhancement
Specify DIR of clean .wav files and noise wav files. Random files will be mixed together at random SNR.

use with: python3 random_database.py --id 1     to name database with ID 1

@author: Felix Schürmann

SNR Mixing from: https://github.com/Sato-Kunihiko/audio-SNR/
"""
import matplotlib.pyplot as plt
import os
import random
import sys
import librosa
import librosa.display
import argparse
import array
import math
import wave
import numpy as np
import joblib


"""specify noise and clean wav dirs"""
WAV_DIR  = os.getcwd()+"/onespeaker/testset"
NOISE_DIR = os.getcwd()+"/onespeaker/noise"

"""number of utterances used for dataset"""
UT_LEN=100


SNR = ['-5', '0', '5','10','15','20']

WAV=0

CREATE_TRAINING_DATA=1

#m_snr=int(SNR[0])
m_snr=1

"""ArgumentParser"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default='1', required=False)
    args = parser.parse_args()
    return args

training_data_clean=[]
training_data_noisy=[]
training_data_clean_specDB=[]
training_data_noisy_specDB=[]
wav_amplitude_clean_list=[]
wav_amplitude_noisy_list=[]

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def write_csv(noise,snr,id):
    import csv

    with open("noise_snr_mix_"+str(id)+".csv", "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        dict = [noise, snr]
        #writer.writerows(noise)
        #writer.writerows(snr)
        writer.writerows(dict)


def create_training_data():
    training_data_clean=[]
    training_data_noisy=[]
    training_data_clean_specDB=[]
    training_data_noisy_specDB=[]
    training_data_noisy_phase=[]
    training_data_clean_phase=[]
    snr_list=[]
    noise_list=[]
    wav_amplitude_clean_list=[]
    wav_amplitude_noisy_list=[]

    for x in range(0,UT_LEN):
        path = WAV_DIR
        rnd_file = random.choice(os.listdir(WAV_DIR))
        rnd_noise = random.choice(os.listdir(NOISE_DIR))
        rnd_snr = random.choice(SNR)


        clean_wav = wave.open(os.path.join(WAV_DIR,rnd_file), "r")
        noise_wav = wave.open(os.path.join(NOISE_DIR,rnd_noise), "r")

        clean_amp = cal_amp(clean_wav)
        noise_amp = cal_amp(noise_wav)

        clean_rms = cal_rms(clean_amp)

        start = random.randint(0, len(noise_amp)-len(clean_amp))
        divided_noise_amp = noise_amp[start: start + len(clean_amp)]
        noise_rms = cal_rms(divided_noise_amp)

        snr = rnd_snr
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

        adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = (clean_amp + adjusted_noise_amp)

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed_amp.min(axis=0)
                mixed_amp = mixed_amp * (reduction_rate)
                clean_amp = clean_amp * (reduction_rate)

        print (os.path.join(WAV_DIR,rnd_file) + " at: " + rnd_snr + "db SNR!" + "with "+str(rnd_noise) )

        fft_clean= librosa.stft(clean_amp,n_fft=512)
        fft_mixed= librosa.stft(mixed_amp,n_fft=512)


        magnitude_clean, phase_clean = librosa.magphase(fft_clean)
        magnitude_noisy, phase_noisy = librosa.magphase(fft_mixed)

        C = librosa.amplitude_to_db(magnitude_clean, ref=np.max)
        D = librosa.amplitude_to_db(magnitude_noisy, ref=np.amax)

        training_data_noisy_phase.append(phase_noisy)
        training_data_clean_phase.append(phase_clean)
        training_data_clean_specDB.append(C)
        training_data_noisy_specDB.append(D)

        snr_list.append(snr)
        noise_list.append(rnd_noise)

        x=x+1
        print(x)

    if WAV==0:
        write_csv(noise_list,snr_list,id)
        print("saving spectra")
        joblib.dump(training_data_clean_specDB, 'cleanspec_mix'+'_part'+str(id)+'.pkl')
        print("30%")
        joblib.dump(training_data_noisy_specDB, 'noisyspec_mix'+'_part'+str(id)+'.pkl')
        print("50%")
        joblib.dump(training_data_noisy_phase, 'noisyphase_mix'+'_part'+str(id)+'.pkl')
        print("70%")
        joblib.dump(training_data_clean_phase, 'cleanphase_mix'+'_part'+str(id)+'.pkl')
        print('100%')
    else:
        write_csv(noise_list,snr_list,id)
        print("saving spectra")
        joblib.dump(training_data_clean_specDB, 'cleanspec_mix'+'_db_'+str(m_snr)+'.pkl')
        print("30%")
        joblib.dump(training_data_noisy_specDB, 'noisyspec_mix'+'_db_'+str(m_snr)+'.pkl')
        print("50%")
        joblib.dump(training_data_noisy_phase, 'noisyphase_mix'+'_db_'+str(m_snr)+'.pkl')
        print("70%")
        joblib.dump(training_data_clean_phase, 'cleanphase_mix'+'_db_'+str(m_snr)+'.pkl')
        print('100%')



if __name__ == '__main__':
    args = get_args()
    id = args.id


    if CREATE_TRAINING_DATA==1:
        create_training_data() # Spectrum Clean / Noisy + dump
    else:
        pass
