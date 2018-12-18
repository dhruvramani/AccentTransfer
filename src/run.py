import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import *
from feature import *
from dataset import *
from vctk import VCTK
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Speech Accent Transfer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--resume', '-r', type=int, default=1, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=4, help='Number of epochs to train.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=1e-5, help='Weight decay (L2 penalty).')

# Loss network trainer
parser.add_argument('--lresume', type=int, default=1, help='resume loss from checkpoint')
parser.add_argument('--loss_lr', type=float, default=1e-4, help='learning rate')

# Accent Network trainer
parser.add_argument('--aresume', type=int, default=1, help='resume accent network from checkpoint')
parser.add_argument('--accent_lr', type=float, default=1e-3, help='learning rate fro accent network')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, tsepoch, tstep, lsepoch, lstep, asepoch, astype = 0, 0, 0, 0, 0, 0, 0

loss_fn = torch.nn.MSELoss() # MaskedMSE()
criterion = nn.CrossEntropyLoss()

print('==> Preparing data..')

# To get logs of current run only
with open("../save/transform/logs/transform_train_loss.log", "w+") as f:
    pass 

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

print('==> Creating networks..')
t_net = Transformation()
t_net = t_net.to(device)
a_net = AccentNet()
a_net = a_net.to(device)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

if(args.lresume):
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
    if(os.path.isfile("../save/accent/network.ckpt")):
        a_net.load_state_dict(torch.load('../save/accent/network.ckpt'))
        print("=> Accent Network : loaded")

    if(os.path.isfile("../save/accent/info.txt")):
        with open("../save/accent/info.txt", "r") as f:
            asepoch, astep = (int(i) for i in str(f.read()).split(" "))
            print("=> Loss Network : prev epoch found")

if(args.resume):
    if(os.path.isfile('../save/transform/trans_model.ckpt')):
        t_net.load_state_dict(torch.load('../save/transform/trans_model.ckpt'))
        print('==> Transformation network : loaded')

    if(os.path.isfile("../save/transform/info.txt")):
        with open("../save/transform/info.txt", "r") as f:
            tsepoch, tstep = (int(i) for i in str(f.read()).split(" "))
        print("=> Transformation network : prev epoch found")

def train_accent(epoch):
    global astep
    trainset = AccentDataset()
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    dataloader = iter(dataloader)
    print('\nEpoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = net.parameters()
    optimizer = optim.Adam(params, lr=args.accent_lr) #, momentum=0.9)#, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for batch_idx in range(astep, len(dataloader)):
        (inputs, targets) = next(dataloader)
        inputs, targets = inputs[0], targets[0] # batch_size == 1 ~= 1 sample
        targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        y_pred = net(inputs)
        loss = criterion(y_pred, targets)
        loss = loss / inputs.shape[0]
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/accent/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/accent/logs/train_acc", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/accent/network.ckpt')

        with open("../save/accent/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, batch_idx))

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        astep = 0

def train_lossn(epoch):
    global lstep
    vdataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
    dataloader = DataLoader(vdataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    dataloader = iter(dataloader)

    print('\n=> Loss Epoch: {}'.format(epoch))
    train_loss, total = 0, 0
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.loss_lr, weight_decay=args.decay)
    
    for i in range(lstep, len(dataloader)):
        (audios, captions) = next(dataloader)
        if(type(audios) == int):
            print("=> Loss Network : Chucked Sample")
            continue
        
        del captions
        audios = (audios[:, :, :, 0:500].to(device), audios[:, :, :, 500:1000].to(device))
        # Might have to remove the loop,, memory
        for audio in audios:
            latent_space = encoder(audio)
            output = decoder(latent_space)
            optimizer.zero_grad()
            loss = criterion(output, audio[:, :, :, :-3])
            loss.backward()
            optimizer.step()

        del audios
        train_loss += loss.item()

        with open("../save/loss/logs/lossn_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / (i - lstep +1)))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(encoder.state_dict(), '../save/loss/loss_encoder.ckpt')
        torch.save(decoder.state_dict(), '../save/loss/loss_decoder.ckpt')

        with open("models/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - lstep + 1)))

    lstep = 0
    del dataloader
    del vdataset
    print('=> Loss Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(data_loader)))

def train_transformation(epoch):
    global tstep
    print('\n=> Transformation Epoch: {}'.format(epoch))
    t_net.train()
    
    vdataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
    dataloader = DataLoader(vdataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    dataloader = iter(dataloader)

    train_loss = 0
    tr_con = 0
    tr_acc = 0
    tr_mse = 0

    params = t_net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    l_list = list(encoder.children())
    l_list = list(l_list[0].children())
    conten_activ = torch.nn.Sequential(*l_list[:-1]) # Not having batchnorm

    for param in conten_activ.parameters():
        param.requires_grad = False

    for param in a_net.parameters():
        param.requires_grad = False

    alpha, beta = 200, 100000 # TODO : CHANGEd from 7.5, 100
    for i in range(tstep, len(dataloader)):
        try :
            (audios, captions) = next(dataloader)
        except ValueError:
            break
        if(type(audios) == int):
            print("=> Transformation Network : Chucked Sample")
            continue

        audios = (audios[:, :, :, 0:300].to(device), audios[:, :, :, 300:600].to(device), audios[:, :, :, 600:900].to(device))
        for audio in audios : # LOL - splitting coz GPU
            optimizer.zero_grad()
            y_t = t_net(audio)

            content = conten_activ(audio)
            y_c = conten_activ(y_t)
            c_loss = loss_fn(y_c, content)

            mel_orig, mel_fake = convert_to_mel(audio), convert_to_mel(y_t)
            y_a, y_apred = a_net(mel_orig), a_net(mel_fake)
            a_loss = criterion(y_apred, y_a)

            loss = alpha * c_loss + beta * a_loss 

            train_loss = loss.item()
            tr_con = c_loss.item()
            tr_acc = a_loss.item()
        
            loss.backward()
            optimizer.step()

        del audios

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(t_net.state_dict(), '../save/transform/trans_model.ckpt')
        with open("../save/transform/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/transform/logs/transform_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss))

        progress_bar(i, len(dataloader), 'Loss: {}, Con Loss: {}, Acc Loss: {} '.format(train_loss, tr_con, tr_acc))

    tstep = 0
    del dataloader
    del vdataset
    print('=> Transformation Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, args.epochs, train_loss))


def test():
    global t_net
    t_net.load_state_dict(torch.load('../save/transform/trans_model.ckpt'))
    vdataset = VCTK('/home/nevronas/dataset/', download=False)
    #dataloader = DataLoader(vdataset, batch_size=1)
    #audio, _ = next(iter(dataloader))
    audio, fs = load_audio('/home/nevronas/dataset/vctk/raw/p225_308.wav')
    audio = torch.Tensor(audio)
    audio, phase = test_transform(audio)
    audio = audio.to(device)
    out = t_net(audio)
    out = out[0].detach().cpu().numpy()
    audio = audio[0].cpu().numpy()
    matplotlib.image.imsave('../save/plots/input/audio.png', audio[0])
    matplotlib.image.imsave('../save/plots/output/stylized_audio.png', out[0])
    aud_res = reconstruction(audio[0], phase)
    out_res = reconstruction(out[0], phase[:, :-3])
    librosa.output.write_wav("../save/plots/input/raw_audio.wav", aud_res, fs)
    librosa.output.write_wav("../save/plots/output/raw_output.wav", out_res, fs)
    print("Testing Finished")

'''
for epoch in range(lsepoch, lsepoch + args.epoch):
    train_lossn(epoch)
for epoch in range(asepoch, asepoch + args.epoch):
    train_accent(epoch)
'''
for epoch in range(tsepoch, tsepoch + args.epochs):
    train_transformation(epoch)

test()
