import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, gamma=2.0):
        mask = (gamma * target) / (target + 10e-8)
        self.loss = self.criterion(input * mask, target * mask)
        return self.loss


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



class GramMatrix(torch.nn.Module):
    def forward(self, y):
        # TODO - modify this
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        #gram = features.bmm(features_t)
        return gram

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class TransformationNetwork(torch.nn.Module):
    def __init__(self, n_channels=1):
        super(TransformationNetwork, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.e1 = down(64, 128)
        self.e2 = down(128, 256)
        self.e3 = down(256, 256)

        self.d2 = up(512, 128)
        self.d3 = up(256, 64)
        self.d4 = up(128, 64)
        self.outc = torch.nn.Conv2d(64, n_channels, 1)

    def forward(self, x):
        h1 = self.inc(x)
        h2 = self.e1(h1)
        h3 = self.e2(h2)
        h4 = self.e3(h3)

        h = self.d2(h4, h3)
        h = self.d3(h, h2)
        h = self.d4(h, h1)
        return self.outc(h)

class SoundNet(torch.nn.Module):
    # KL Divergence Loss - Refer https://github.com/Kajiyu/Modern_SoundNet/blob/master/soundnet.ipynb 
    def __init__(self):
        super(SoundNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = torch.nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = torch.nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = torch.nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = torch.nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = torch.nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = torch.nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = torch.nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = torch.nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8_1 = torch.nn.Conv1d(1024, 1000, 4, stride=2, padding=0)
        self.conv8_2 = torch.nn.Conv1d(1024, 401, 4, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(859000, 1000)
        self.fc2 = torch.nn.Linear(344459, 365)

    def forward(self, input_wav):
        x = self.pool1(F.relu(torch.nn.BatchNorm1d(16)(self.conv1(input_wav))))
        x = self.pool2(F.relu(torch.nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(torch.nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(torch.nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(torch.nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(torch.nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(torch.nn.BatchNorm1d(1024)(self.conv7(x)))
        x_object = Flatten()(F.relu(self.conv8_1(x)))
        x_place = Flatten()(F.relu(self.conv8_2(x)))
        x_object = self.fc1(x_object)
        x_place = self.fc2(x_place)
        y = [x_object, x_place]
        return y

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 225, 5, dilation=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.BatchNorm2d(225),
            nn.Conv2d(225, 256, 5, stride=2),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, 3, stride=2),   # b, 8, 2, 2
            # nn.ReLU(True),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 100, 3, stride=1),   # b, 8, 2, 2
            nn.ReLU(True),
            nn.BatchNorm2d(100)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            # nn.ConvTranspose2d(256, 256, 5, stride=2),  # b, 16, 5, 5
            # nn.ReLU(True),
            # nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 11, stride=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, 1, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class Transformation(torch.nn.Module):
    def __init__(self):
        super(Transformation, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        # self.enc.load_state_dict(torch.load('../save/transform/trans_encoder.ckpt'))
        # self.dec.load_state_dict(torch.load('../save/transform/trans_decoder.ckpt'))
        
    def forward(self, x):
        return self.dec(self.enc(x))


# To be changed with whatever @Assad uses
class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*14*13, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size()[0], 128*14*13)
        x = self.classifier(x)
        return x
