from __future__ import print_function
import contextlib
import os
from os import path
import numpy as np
import pandas as pd
import scipy
import wave
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt
from pydub import AudioSegment
import librosa

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

def mfcc(path):
    ipd.Audio(path)
    sample_rate, audio = wavfile.read(path)

    audio = normalize_audio(audio)

    #audio framing
    hop_size = 15 #ms
    FFT_size = 2048
    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)


    #convert to frequency domain
    #for simplicity, hanning window is chosen.

    window = get_window("hann", FFT_size, fftbins=True)
    audio_win = audio_framed * window
    ind = 69

    #performing FFT
    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)

    #Calculating signal power

    audio_power = np.square(np.abs(audio_fft))

    #computing MEL-spaced filterbank, passing framed audio through them

    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 10

    #computing filter points
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

    #constructing filterbank
    filters = get_filters(filter_points, FFT_size)

    #area normalization
    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]

    #filtering the signal
    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)

    #generating cepstral coefficients
    dct_filter_num = 40

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficients = np.dot(dct_filters, audio_log)
    return cepstral_coefficients
def createModel(model):
    test_dim = 40
    maxlen = 100
    batch_size = 100
    nb_filter = 64
    filter_length_1 = 20
    filter_length_2 = 10
    hidden_dims = 250
    nb_epoch = 8
    nb_classes = 2

    model.add(Convolution1D(filters=100, kernel_size=4,
                            filter_length=20,
                            input_shape = (test_dim, 408),
                            border_mode='valid',
                            activation='relu',))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=100, kernel_size=4,
                            filter_length=10,
                            border_mode='same',
                            activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_length=2))
    model.add(Convolution1D(filters=100, kernel_size=4,
                            filter_length=10,
                            border_mode='same',
                            activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_length=2))

    #flatten output
    model.add(Flatten())

    #add vanilla hidden layer
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')


def train_model(df, model):
    X_train = []
    y_train = []
    for i in range(len(df)):
        X_train.append(mfcc(df['path'][i]))
        y_train.append(df['label'][i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model.fit(X_train, y_train, batch_size=100, nb_epoch=8)
    print("model trained")
    return model


maxduration=0
dict1 = {}
lang_models = {}
model = Sequential()
createModel(model)
for path in os.listdir(r"~/train"):
    maxduration=0
    for path2 in os.listdir(r"~/train\{}".format(path)):
        duration = librosa.get_duration(filename=r"~/train/{}/{}".format(path,path2))
        maxduration = max(maxduration,duration)
    dict1[path] = maxduration

for path in os.listdir(r"~/train"):
    maxduration=0
    normalized=[]
    for path2 in os.listdir(r"~/train\{}".format(path)):
        duration = librosa.get_duration(filename=r"~/train/{}/{}".format(path,path2))
        silent = AudioSegment.silent(duration=(dict1[path]-duration))
        sound = AudioSegment.from_mp3(path2)
        newsound = path2[:-3] + "wav"
        sound = sound + silent
        sound.export(newsound, format="wav")
        mfcc_sound = mfcc(newsound)
        mean_mfcc = np.mean(mfcc_sound, axis=0)
        normalized.append(mean_mfcc)
    flat = [a.ravel() for a in normalized]
    stacked = np.vstack(flat)
    df = pd.DataFrame(stacked)
    train_model(df, model)




