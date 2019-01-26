from torch.utils.data import Dataset
import numpy as np 
import torch

# Segmentation for Training 
def make_segments(mels,labels, COL_SIZE = 128):
    '''
    Makes segments of mel and attaches them to the labels
    :param mels: list of mels
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mel,label in zip(mels,labels):
        for start in range(0, int(mel.shape[1] / COL_SIZE)):
            segments.append(mel[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return (segments, seg_labels)

# Segmentation for testing
def segment_one(mel, COL_SIZE = 128):
    '''
    Creates segments from on mel image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mel (numpy array): mel array
    :return (numpy array): Segmented mel array
    '''
    segments = []
    for start in range(0, int(mel.shape[1] / COL_SIZE)):
        segments.append(mel[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mels(X_train):
    '''
    Creates segmented mels from X_train
    :param X_train: list of mels
    :return: segmented mels
    '''
    segmented_mels = []
    for mel in X_train:
        segmented_mels.append(segment_one(mel))
    return(segmented_mels)

class AccentDataset(Dataset):
    """Accent dataset."""

    def __init__(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_train = np.expand_dims(X_train,axis = 1)
        self.X_tens = torch.from_numpy(self.X_train)
        self.y_tens = torch.from_numpy(self.y_train)
        self.count = 0

    def __len__(self):
        return int(self.y_train.shape[0])

    def __getitem__(self, idx):
        return self.X_tens[idx].type(torch.FloatTensor), self.y_tens[idx].type(torch.LongTensor)


class LanguageDataset(Dataset):
    """Language dataset."""

    def __init__(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_train = np.expand_dims(X_train,axis = 1)
        self.X_tens = torch.from_numpy(self.X_train)
        self.y_tens = torch.from_numpy(self.y_train)
        self.count = 0

    def __len__(self):
        return int(self.y_train.shape[0])

    def __getitem__(self, idx):
        return self.X_tens[idx].type(torch.FloatTensor), self.y_tens[idx].type(torch.LongTensor)
