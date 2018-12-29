import librosa
import os
import numpy as np
import glob
import torch

N_FFT = 1024

def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''
    y, sr = librosa.load('../audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=24000, scale=True))

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=24000, n_mfcc=13))

def mel_transform(S, fs=48000):
    mel = librosa.filters.mel(fs, N_FFT)
    return  mel # shit sors

def transform_stft(signal, pad=0):
    D = librosa.stft(signal, n_fft=N_FFT)
    S, phase = librosa.magphase(D)
    S = np.log1p(S)
    if(pad):
        S = librosa.util.pad_center(S, pad)
    return S, phase

def to_mel(signal):
    signal = signal.flatten()
    signal, _ = transform_stft(signal)
    mel = mel_transform(signal)
    signal = np.matmul(mel, signal) # sd is troch tensor
    print(signal.shape)
    return signal

def filter_df(df):
    '''
    Function to filter audio files based on df columns
    df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
    english_residence,length_of_english_residence,native_language,other_languages,sex]
    :param df (DataFrame): Full unfiltered DataFrame
    :return (DataFrame): Filtered DataFrame
    '''

    # Example to filter arabic, mandarin, and english and limit to 73 audio files
    arabic,arabicy = [],[]
    mandarin,mandariny = [],[]
    english,englishy = [],[]
    
    for i in range(79):
        english.append(to_mel(get_wav("english"+str(i+1))))
        englishy.append(1)
        mandarin.append(to_mel(get_wav("mandarin"+str(i+1))))
        mandariny.append(2)
        arabic.append(to_mel(get_wav("arabic"+str(i+1))))
        arabicy.append(0)

    val = english + arabic + mandarin
    val2 = englishy + arabicy + mandariny
    df = {'wav':val,'native_language':val2}
    return df

def split_people(df,test_size=0.2):
    '''
    Create train test split of DataFrame
    :param df (DataFrame): Pandas DataFrame of audio files to be split
    :param test_size (float): Percentage of total files to be split into test
    :return X_train, X_test, y_train, y_test (tuple): Xs are list of df['language_num'] and Ys are df['native_language']
    '''
    return train_test_split(df['wav'],df['native_language'],test_size=test_size,random_state=1234)

def invert_spectrogram(result, a_content, fs, outpath):
    a = np.zeros_like(a_content)
    a[:a_content.shape[0],:] = np.exp(result[0,0].T) - 1
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    x = 0

    for i in range(500):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    librosa.output.write_wav(outpath, x, fs)

def read_audio_spectrum(filename):
    signal, fs = librosa.load(filename)
    S = librosa.stft(signal, N_FFT)
    final = np.log1p(np.abs(S[:,:430]))  
    return final, fs

def power_spectral(signal):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    S = librosa.stft(emphasized_signal, N_FFT)
    power_spectral=np.square(S)/N_FFT
    final = np.log1p(np.abs(power_spectral[:,:430]))  
    return final