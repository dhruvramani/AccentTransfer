import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from PIL import Image
from feature import *
import librosa
from torch import nn
import matplotlib
from models import *
from dataset import *
from torch.utils.data import DataLoader

from new_feature import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio(audio_path):
    signal, fs = librosa.load(audio_path)
    return signal, fs

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions

def to_mel(signal):
    #signal = signal.flatten()
    #signal, _ = transform_stft(signal)
    signal,phase=audioFileToSpectrogram(signal)
    mel = mel_transform(signal)
    signal = np.matmul(mel, signal) # sd is troch tensor
    print(signal.shape)
    return signal,phase

#def reconstruction(S, phase):
def reconstruction(spectrogram,mel, fftWindowSize = FFT, phaseIterations=10, phase=None):
    if phase is not None:
        # reconstructing the new complex matrix

        spectrogram1=np.matmul( np.transpose(mel) , spectrogram[:-1,:])
        #squaredAmplitudeAndSquaredPhase = np.power(spectrogram1, 2)
        #squaredPhase = np.power(phase, 2)
        #unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase[:,:-1] - squaredPhase, 0))
        #unexpd = np.sqrt(np.absolute(squaredAmplitudeAndSquaredPhase[:,:-1] - squaredPhase))
        amplitude = np.expm1(spectrogram1)
        #stftMatrix = amplitude + phase * 1j
        #stftMatrix=amplitude[:,:-1]*(np.cos(phase)+ 1j* np.sin(phase))
        stftMatrix=amplitude[:,:-1]+ 1j* phase
        audio = librosa.istft(stftMatrix)
        #print(phase[0])
        return audio*100

def mel_transform(S, fs=48000):
    mel = librosa.filters.mel(fs, FFT)
    return  mel # shit sors

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.astype(np.float32)
    inp = inp.flatten()
    inp, phase = transform_stft(inp, pad=False)
    mel = mel_transform(inp)
    inp = np.matmul(mel, inp)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp, phase,mel

def denoise(nparr, val=540):
    sub = nparr[:,val:val+10]#10
    return nparr - (np.mean(sub) + 3 * np.std(sub))
    #print(nparr.shape)

def main():

    trans_net = Transformation()
    trans_net = trans_net.to(device)
    trans_net.load_state_dict(torch.load('../save/transform/network.ckpt'))
    trans_net.load_state_dict(torch.load('../save/transform/network.ckpt'))
    
    #vdataset = ('/home/nevronas/dataset/', download=False)
    #dataloader = DataLoader(vdataset, batch_size=1)

    #audio, _ = next(iter(dataloader))
    # audio, fs = load_audio('/home/nevronas/dataset/accent/recordings/english2.wav')
    # target_audio, target_fs = load_audio('/home/nevronas/dataset/accent/manda.wav')
    # #style, fz = load_audio("/home/nevronas/Projects/Nevronas-Projects/Audio/AudioStyleTransfer/save/style/style_lady.wav")
    # audio = torch.Tensor(audio)#, torch.Tensor(style)
    # audio, phase, mel = inp_transform(audio)
    # target_audio = torch.Tensor(target_audio)#, torch.Tensor(style)
    # target_audio, target_phase, target_mel = inp_transform(target_audio)
    # #style, _ = inp_transform(style)
    # audio = audio.to(device)
    # out = trans_net(audio)
    # out = out[0].detach().cpu().numpy()
    # audio = audio[0].cpu().numpy()
    # #out2 = denoise(out[0])
    # target_audio = target_audio[0].cpu().numpy()
    # matplotlib.image.imsave('../save/plots/input/input_audio.png', audio[0])
    # matplotlib.image.imsave('../save/plots/output/accented_audio.png', out[0])
    
    # matplotlib.image.imsave('../save/plots/output/target_audio.png', target_audio[0])
    # aud_res = reconstruction(audio[0], phase, mel)
    # out_res = reconstruction(out[0][:-1, :-1], phase, mel)#[:, :-3])
    # #out_res = denoise(out_res)
    # librosa.output.write_wav("../save/plots/input/raw_audio.wav", aud_res, fs)
    # librosa.output.write_wav("../save/plots/output/raw_output.wav", out_res, fs)
    #invert_spectrogram(audio[0], audio[0], fs, '../save/plots/output/raw_audio.wav')

    #matplotlib.image.imsave('out.png', out[0])

    # Print out the image and the generated caption
    
    '''
    Save as numpy array
    with open("../save/plots/output/input_np.dat" ,"wb") as f:
        np.save(f, audio[0])
    with open("../save/plots/output/output_np.dat" ,"wb") as f:
        np.save(f, out[0])
    '''

    #New stuff

    # audio, sampleRate = loadAudioFile("english34")
    audio, sampleRate = load_audio('/home/nevronas/dataset/accent/manda.wav')
    #audio, phase = audioFileToSpectrogram(audio)
    #audio, sampleRate = load_audio('/home/nevronas/dataset/vctk/raw/p280_408.wav')
    audio,phase=to_mel(audio)
    audio, phase = audio[:,:500], phase[:,:500]
    audio = torch.Tensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)
    out = trans_net(audio)
    out = out[0].detach().cpu().numpy()
    saveSpectrogram(out[0], "../save/plots/output/accented_audio.png")
    #out = spectrogramToAudioFile(out[0], phase=phase
    mel=mel_transform(out[0])
    out = reconstruction(out[0],mel,phase=phase)
    saveAudioFile(out, "../save/plots/output/raw_output.wav", sampleRate)
    
        
    
if __name__ == '__main__':
    main()
