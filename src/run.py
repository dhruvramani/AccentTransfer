import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import pickle 

import accuracy
from models import *
from feature import *
from dataset import *
from utils import progress_bar

import matplotlib.pyplot as plt
import matplotlib

from new_feature import *

parser = argparse.ArgumentParser(description='PyTorch Speech Accent Transfer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=15, help='Number of epochs to train.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=0.001, help='Weight decay (L2 penalty).')

parser.add_argument('--preparedata', type=bool, default=0, help='Recreate the dataset.')
parser.add_argument('--dataset', type=bool, default=1, help='select the dataset 1. Accent , 0. Language')

# Loss network trainer
parser.add_argument('--lresume', type=int, default=1, help='resume loss from checkpoint')
parser.add_argument('--loss_lr', type=float, default=0.0001, help='learning rate')

# Accent Network trainer
parser.add_argument('--aresume', type=int, default=0, help='resume accent network from checkpoint')
parser.add_argument('--accent_lr', type=float, default=0.0001
    , help='learning rate for accent network')

# Language Network trainer
parser.add_argument('--langresume', type=int, default=1, help='resume language network from checkpoint')
parser.add_argument('--lang_lr', type=float, default=0.0001
    , help='learning rate for language network')

args = parser.parse_args()

FILE_NAME = 'data.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

def load_audio(audio_path):
    signal, fs = librosa.load(audio_path)
    return signal, fs

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audio, captions = zip(*data)
    data = None
    del data
    audio = torch.stack(audio, 0)
    return audio, captions
'''
def mel_transform(S, fs=48000):
    mel = librosa.filters.mel(fs, N_FFT)
    return  mel 
def to_mel(signal):
    #signal = signal.flatten()
    signal, _ = audioFileToSpectrogram(signal)
    mel = mel_transform(signal)
    signal = np.matmul(mel, signal) # sd is troch tensor
    print(signal.shape)
    return signal 
'''

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    inp, _ = transform_stft(inp)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    return inp

def test_transform(inp):
    inp = inp.numpy()
    inp = inp.astype(np.float32)
    inp = inp.flatten()
    inp, phase = transform_stft(inp, pad=False)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp, phase

def convert_to_mel(audio):
    aud_np = audio.numpy()
    _, _, meld = mel_transform(aud_np) # fs = 480000, default
    audio = torch.matmul(meld, audio)
    return audio

def get_accent(path='../save/style/ger53.wav'):
    N_FFT = 1536
    signal, fs = librosa.load(path)
    del fs
    signal = librosa.stft(signal, n_fft=N_FFT)
    signal, phase = librosa.magphase(signal)
    del phase
    signal = np.log1p(signal)
    signal = signal[ :, 60:188]
    signal = torch.from_numpy(signal) # TODO : get style audio
    signal = signal.unsqueeze(0)
    return signal

best_acc, tsepoch, tstep, lsepoch, lstep, astep, asepoch, astype, lasepoch, lastep = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

mse = torch.nn.MSELoss() # MaskedMSE()
criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')

# Uncomment while training transformation network 
t_net = Transformation().to(device)
a_net = AlexNet().to(device)
encoder = Encoder().to(device)
decoder = Decoder().to(device)

if(args.preparedata):
    print('==> Preparing data..')
    if(args.dataset):
        filtered_df = filter_df(None)
    else:
        filtered_df = accent_new()

    X_train, X_test, y_train, y_test = split_people(filtered_df)

    train_count = Counter(y_train)
    test_count =  Counter(y_test)
    print('==> Creating segments..')
    X_train, y_train = make_segments(X_train, y_train)
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

    saveSpectrogram(X_train[0], "../save/plots/input/new_feature_test.png")
    print("==> Saving dataset..")
    if(args.dataset):
        with open("../save/dataset/data.dat", "wb") as f:
            data = (X_train, X_test, y_train, y_test)
            pickle.dump(data, f)
    else:
        with open("../save/dataset/lang_data.dat", "wb") as f:
            data = (X_train, X_test, y_train, y_test)
            pickle.dump(data, f)

else:
    print("==> Loading dataset..")
    if(args.dataset):
        with open("../save/dataset/data.dat", "rb") as f:
            (X_train, X_test, y_train, y_test) = pickle.load(f)
    else:
        with open("../save/dataset/lang_data.dat", "rb") as f:
            (X_train, X_test, y_train, y_test) = pickle.load(f)


if(args.lresume):
    with open("../save/transform/logs/lossn_train_loss.log", "w+") as f:
        pass 
    if(os.path.isfile('../save/loss/loss_encoder.ckpt')):
        encoder.load_state_dict(torch.load('../save/loss/loss_encoder.ckpt'))
        del decoder # no need to waste memory on this if resumed
        #decoder.load_state_dict(torch.load('../save/loss/loss_decoder.ckpt'))
        print("=> Loss Network : loaded")
    
    if(os.path.isfile("../save/loss/info.txt")):
        with open("../save/loss/info.txt", "r") as f:
            lsepoch, lstep = (int(i) for i in str(f.read()).split(" "))
            print("=> Loss Network : prev epoch found")

if(args.aresume):
    with open("../save/accent/logs/accentn_train_loss.log", "w+") as f:
        pass 
    if(os.path.isfile("../save/accent/network.ckpt")):
        a_net.load_state_dict(torch.load('../save/accent/network.ckpt'))
        print("=> Accent Network : loaded")

    if(os.path.isfile("../save/accent/info.txt")):
        with open("../save/accent/info.txt", "r") as f:
            asepoch, astep = (int(i) for i in str(f.read()).split(" "))
            print("=> Loss Network : prev epoch found")

if(args.langresume):
    with open("../save/language/logs/languagen_train_loss.log", "w+") as f:
        pass 
    if(os.path.isfile("../save/language/network.ckpt")):
        a_net.load_state_dict(torch.load('../save/language/network.ckpt'))
        print("=> Language Network : loaded")

    if(os.path.isfile("../save/language/info.txt")):
        with open("../save/language/info.txt", "r") as f:
            lasepoch, lastep = (int(i) for i in str(f.read()).split(" "))
            print("=> Language Network : prev epoch found")

if(args.resume):
    with open("../save/transform/logs/transform_train_loss.log", "w+") as f:
        pass 
    if(os.path.isfile('../save/transform/network.ckpt')):
        t_net.load_state_dict(torch.load('../save/transform/network.ckpt'))
        print('==> Transformation network : loaded')

    if(os.path.isfile("../save/transform/info.txt")):
        with open("../save/transform/info.txt", "r") as f:
            tsepoch, tstep = (int(i) for i in str(f.read()).split(" "))
        print("=> Transformation network : prev epoch found")


def train_accent(epoch):
    global astep
    trainset = AccentDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Accent Epoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = a_net.parameters()
    optimizer = optim.Adam(params, lr=args.accent_lr)#, momentum=0.9)#, weight_decay=5e-4)

    for batch_idx in range(astep, len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        y_pred = a_net(inputs)
        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/accent/logs/accentn_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/accent/logs/accentn_train_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(a_net.state_dict(), '../save/accent/network.ckpt')
        with open("../save/accent/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, batch_idx))

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    astep = 0
    print('=> Accent Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(dataloader)))

def train_lang(epoch):
    global astep
    trainset = LanguageDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Language Epoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = a_net.parameters()
    optimizer = optim.Adam(params, lr=args.accent_lr)#, momentum=0.9)#, weight_decay=5e-4)

    for batch_idx in range(astep, len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        y_pred = a_net(inputs)
        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/language/logs/languagen_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/language/logs/languagen_train_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(a_net.state_dict(), '../save/language/network.ckpt')
        with open("../save/language/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, batch_idx))

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    astep = 0
    print('=> Language Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(dataloader)))

def train_lossn(epoch):
    global lstep
    trainset = AccentDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    print('\n=> Loss Epoch: {}'.format(epoch))
    train_loss, total = 0, 0
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.loss_lr)
    
    for i in range(lstep, len(dataloader)):
        (audio, captions) = next(dataloader)
        
        del captions
        audio = audio.to(device)
        # Might have to remove the loop,, memory
        latent_space = encoder(audio)
        output = decoder(latent_space)
        optimizer.zero_grad()
        loss = mse(output[:,:,:-2,:-1], audio)
        loss.backward()
        optimizer.step()

        a1 = audio[0].cpu().numpy()
        a2 = output[0].detach().cpu().numpy()
        matplotlib.image.imsave('../save/plots/input/before.png', a1[0])
        matplotlib.image.imsave('../save/plots/input/after.png', a2[0])
        
        train_loss += loss.item()

        del audio, latent_space, output, loss

        with open("../save/loss/logs/lossn_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / (i - lstep +1)))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(encoder.state_dict(), '../save/loss/loss_encoder.ckpt')
        torch.save(decoder.state_dict(), '../save/loss/loss_decoder.ckpt')

        with open("../save/loss/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - lstep + 1)))

    lstep = 0
    print('=> Loss Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(dataloader)))

def train_transformation(epoch, accent_idx=1):
    global tstep
    print('\n=> Transformation Epoch: {}'.format(epoch))
    t_net.train()
    
    trainset = AccentDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    train_loss, tr_con, tr_acc, tr_mse = 0, 0, 0, 0

    params = t_net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    l_list = list(encoder.children())
    l_list = list(l_list[0].children())
    conten_activ = torch.nn.Sequential(*l_list[:-1]) # Not having batchnorm

    a_list = list(a_net.children())
    a_list = list(a_list[0].children())
    accent_activ = torch.nn.Sequential(*a_list[:-1])

    for param in conten_activ.parameters():
        param.requires_grad = False

    for param in accent_activ.parameters():
        param.requires_grad = False

    alpha, beta = 100, 2000 # TODO : CHANGEd from 7.5, 100
    gram = GramMatrix()
    accent_audio = get_accent()

    for i in range(tstep, len(dataloader)):
        (audio, captions) = next(dataloader)
        del captions
        audio = audio.to(device)

        optimizer.zero_grad()
        y_t = t_net(audio)

        content = conten_activ(audio)
        y_c = conten_activ(y_t)
        c_loss = mse(y_c[:, :, :-1, :-1], content)

        # y_apred = a_net(y_t)
        # y_a = torch.ones(y_apred.shape[0]).type(torch.LongTensor).to(device) * accent_idx
        # a_loss = criterion(y_apred, y_a)

        ## Remove this if Gram Matrix don't work
        a_loss = 0
        acc_aud = []
        for k in range(audio.size()[0]): # No. of accent audio == batch_size
            acc_aud.append(accent_audio)
        acc_aud = torch.stack(acc_aud).to(device)

        for ac_i in range(2, len(a_list)-4, 3): # NOTE : gets relu of 1, 2, 3
            ac_activ = torch.nn.Sequential(*a_list[:ac_i])
            for param in ac_activ.parameters():
                param.requires_grad = False

            y_a = gram(ac_activ(y_t))
            accent = gram(ac_activ(acc_aud))
        
            a_loss += mse(y_a, accent)
            
        del acc_aud 
        ###

        loss = alpha * c_loss + beta * a_loss

        train_loss, tr_con, tr_acc = train_loss + loss.item(), tr_con + c_loss.item(), tr_acc + a_loss.item()
        
        loss.backward()
        optimizer.step()

        a1 = audio[0].cpu().numpy()
        a2 = y_t[0].detach().cpu().numpy()
        matplotlib.image.imsave('../save/plots/transform/before.png', a1[0])
        matplotlib.image.imsave('../save/plots/transform/after.png', a2[0])

        del audio

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(t_net.state_dict(), '../save/transform/network.ckpt')
        with open("../save/transform/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/transform/logs/transform_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(loss))

        with open("../save/transform/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: {}, Con Loss: {}, Acc Loss: {} '.format(loss, c_loss, a_loss))

    tstep = 0
    print('=> Transformation Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, args.epochs, train_loss / len(dataloader)))


def test():
    # TODO : Test Later
    print('==> Testing network..')
    # Make predictions on full X_test mels
    y_predicted = accuracy.predict_class_all(create_segmented_mels(X_test), a_net)

    # Print statistics
    print(np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1))
    print(accuracy.confusion_matrix(y_predicted, y_test))
    print(accuracy.get_accuracy(y_predicted,y_test))

'''
encoder = Encoder().to(device)
decoder = Decoder().to(device)
for epoch in range(lsepoch, lsepoch + args.epochs):
    train_lossn(epoch)
'''

'''
a_net = AlexNet().to(device)
for epoch in range(lasepoch, lasepoch + args.epochs):
    train_lang(epoch)
    test()
'''

t_net = Transformation().to(device)
for epoch in range(tsepoch, tsepoch + args.epochs):
    train_transformation(epoch)

#test()
