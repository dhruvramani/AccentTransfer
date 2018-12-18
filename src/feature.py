import librosa
import os
import numpy as np
import glob
import torch

N_FFT = 1024

def mel_transform(S, fs=48000):
    mel = librosa.filters.mel(fs, N_FFT)
    meld = torch.Tensor(mel)
    Sd = torch.Tensor(S)
    return Sd, mel, meld

def transform_stft(signal, pad=0):
    D = librosa.stft(signal, n_fft=N_FFT)
    S, phase = librosa.magphase(D)
    S = np.log1p(S)
    if(pad):
        S = librosa.util.pad_center(S, pad)
    return S, phase

def transform_stft_new(signal):
    D = librosa.stft(signal, n_fft=N_FFT)
    S, phase = librosa.magphase(D)
    S = (D**2)/N_FFT
    S = np.log1p(S + 10e-7)
    # S = librosa.util.pad_center(S, 2500)
    return S

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
