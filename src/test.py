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

def mel_transform(S, fs=48000):
    mel = librosa.filters.mel(fs, N_FFT)
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
    return inp, phase, mel

def denoise(nparr):
    sub = nparr[:, 540:550]
    return nparr - np.mean(sub)

def main():

    trans_net = Transformation()
    trans_net = trans_net.to(device)
    trans_net.load_state_dict(torch.load('../save/transform/network.ckpt'))
    trans_net.load_state_dict(torch.load('../save/transform/network.ckpt'))
    
    #vdataset = ('/home/nevronas/dataset/', download=False)
    #dataloader = DataLoader(vdataset, batch_size=1)

    #audio, _ = next(iter(dataloader))
    audio, fs = load_audio('/home/nevronas/dataset/accent/recordings/english2.wav')
    #style, fz = load_audio("/home/nevronas/Projects/Nevronas-Projects/Audio/AudioStyleTransfer/save/style/style_lady.wav")
    audio = torch.Tensor(audio)#, torch.Tensor(style)
    audio, phase, mel = inp_transform(audio)
    #style, _ = inp_transform(style)
    audio = audio.to(device)
    out = trans_net(audio)
    out = out[0].detach().cpu().numpy()
    audio = audio[0].cpu().numpy()
    out2 = denoise(out)
    matplotlib.image.imsave('../save/plots/input/audio.png', audio[0])
    matplotlib.image.imsave('../save/plots/output/stylized_audio.png', out[0])
    matplotlib.image.imsave('../save/plots/output/denoised_audio.png', out2[0])
    aud_res = reconstruction(audio[0], phase, mel)
    out_res = reconstruction(out[0][:-1, :-1], phase, mel)#[:, :-3])
    #out_res = denoise(out_res)
    librosa.output.write_wav("../save/plots/input/raw_audio.wav", aud_res, fs)
    librosa.output.write_wav("../save/plots/output/raw_output.wav", out_res, fs)
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
    
if __name__ == '__main__':
    main()
