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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

import accuracy
from models import *
from feature import *
from dataset import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch Speech Accent Transfer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=0.001, help='Weight decay (L2 penalty).')

# Loss network trainer
parser.add_argument('--lresume', type=int, default=0, help='resume loss from checkpoint')
parser.add_argument('--loss_lr', type=float, default=0.001, help='learning rate')

# Accent Network trainer
parser.add_argument('--aresume', type=int, default=0, help='resume accent network from checkpoint')
parser.add_argument('--accent_lr', type=float, default=0.001
    , help='learning rate fro accent network')

args = parser.parse_args()

FILE_NAME = 'data.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

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

best_acc, tsepoch, tstep, lsepoch, lstep, asepoch, astype = 0, 0, 0, 0, 0, 0, 0

mse = torch.nn.MSELoss() # MaskedMSE()
criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')
t_net = Transformation().to(device)
a_net = AlexNet().to(device)
encoder = Encoder().to(device)
decoder = Decoder().to(device)

print('==> Preparing data..')
filtered_df = filter_df(None)
X_train, X_test, y_train, y_test = split_people(filtered_df)

train_count = Counter(y_train)
test_count =  Counter(y_test)
print('==> Creatting segments..')
X_train, y_train = make_segments(X_train, y_train)
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

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
        (audios, captions) = next(dataloader)
        
        del captions
        # Might have to remove the loop,, memory
        latent_space = encoder(audio)
        output = decoder(latent_space)
        optimizer.zero_grad()
        loss = mse(output, audio)
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

        with open("../save/loss/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - lstep + 1)))

    lstep = 0
    print('=> Loss Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(dataloader)))

def train_transformation(epoch):
    global tstep
    print('\n=> Transformation Epoch: {}'.format(epoch))
    t_net.train()
    
    trainset = AccentDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader))

    train_loss, tr_con, tr_acc, tr_mse = 0, 0, 0, 0

    params = t_net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    l_list = list(encoder.children())
    l_list = list(l_list[0].children())
    conten_activ = torch.nn.Sequential(*l_list[:-1]) # Not having batchnorm

    for param in conten_activ.parameters():
        param.requires_grad = False

    for param in a_net.parameters():
        param.requires_grad = False

    alpha, beta = 7.5, 100 # TODO : CHANGEd from 7.5, 100
    for i in range(tstep, len(dataloader)):
        (audios, captions) = next(dataloader)
        del captions
        optimizer.zero_grad()
        y_t = t_net(audio)

        content = conten_activ(audio)
        y_c = conten_activ(y_t)
        c_loss = mse(y_c, content)

        y_a, y_apred = a_net(audio), a_net(y_t)
        a_loss = criterion(y_apred, y_a)

        loss = alpha * c_loss + beta * a_loss 

        train_loss, tr_con, tr_acc = loss.item(), c_loss.item(), a_loss.item()
        
        loss.backward()
        optimizer.step()

        del audios

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(t_net.state_dict(), '../save/transform/network.ckpt')
        with open("../save/transform/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/transform/logs/transform_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss))

        with open("../save/transform/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: {}, Con Loss: {}, Acc Loss: {} '.format(train_loss, tr_con, tr_acc))

    tstep = 0
    print('=> Transformation Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, args.epochs, train_loss))


def test():
    # TODO : Test Later
    print('==> Testing network..')
    # Make predictions on full X_test mels
    y_predicted = accuracy.predict_class_all(create_segmented_mels(X_test), net)

    # Print statistics
    print(train_count)
    print(test_count)
    print(np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1))
    print(accuracy.confusion_matrix(y_predicted, y_test))
    print(accuracy.get_accuracy(y_predicted,y_test))

for epoch in range(lsepoch, lsepoch + args.epoch):
    train_lossn(epoch)
for epoch in range(asepoch, asepoch + args.epoch):
    train_accent(epoch)
for epoch in range(tsepoch, tsepoch + args.epochs):
    train_transformation(epoch)

#test()
