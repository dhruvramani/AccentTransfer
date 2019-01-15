import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
from os.path import basename
from math import ceil


FFT = 1536
ITER = 10

def loadAudioFile(filePath):
    audio, sampleRate = librosa.load('/home/nevronas/dataset/accent/recordings/{}.wav'.format(filePath))
    return audio, sampleRate

def saveAudioFile(audioFile, filePath, sampleRate):
    librosa.output.write_wav(filePath, audioFile, sampleRate, norm=True)

def expandToGrid(spectrogram, gridSize):
    # crop along both axes
    newY = ceil(spectrogram.shape[1] / gridSize) * gridSize
    newX = ceil(spectrogram.shape[0] / gridSize) * gridSize
    newSpectrogram = np.zeros((newX, newY))
    newSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
    return newSpectrogram

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioFile, fftWindowSize = FFT):
    spectrogram = librosa.stft(audioFile, fftWindowSize)
    phase = np.imag(spectrogram)
    amplitude = np.log1p(np.abs(spectrogram))
    print(amplitude.shape)
    return amplitude, phase

# This is the nutty one
def spectrogramToAudioFile(spectrogram, fftWindowSize = FFT, phaseIterations=10, phase=None):
    if phase is not None:
        # reconstructing the new complex matrix
        squaredAmplitudeAndSquaredPhase = np.power(spectrogram, 2)
        squaredPhase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase - squaredPhase, 0))
        amplitude = np.expm1(unexpd)
        stftMatrix = amplitude + phase * 1j
        audio = librosa.istft(stftMatrix)
    else:
        # phase reconstruction with successive approximation
        # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
        # for the algorithm used
        amplitude = np.exp(spectrogram) - 1
        for i in range(phaseIterations):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * (2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, fftWindowSize)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio

def loadSpectrogram(filePath):
    fileName = basename(filePath)
    if filePath.index("sampleRate") < 0:
        sampleRate == 22050
    else:
        sampleRate = int(fileName[fileName.index("sampleRate=") + 11:fileName.index(").png")])

    image = io.imread(filePath, as_grey=True)
    return image / np.max(image), sampleRate

def saveSpectrogram(spectrogram, filePath):
    spectrum = spectrogram
    
    image = np.clip((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)), 0, 1)

    # Low-contrast image warnings are not helpful, tyvm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(filePath, image)

def fileSuffix(title, **kwargs):
    return " (" + title + "".join(sorted([", " + i + "=" + str(kwargs[i]) for i in kwargs])) + ")"

def handleAudio(filePath, args):
    INPUT_FILE = filePath
    INPUT_FILENAME = basename(INPUT_FILE)
    audio, sampleRate = loadAudioFile(INPUT_FILE)
    spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize=FFT)
    SPECTROGRAM_FILENAME = INPUT_FILENAME + fileSuffix("Input Spectrogram", fft=FFT, iter=iter, sampleRate=sampleRate) + ".png"

    saveSpectrogram(spectrogram, SPECTROGRAM_FILENAME)

    handleImage(SPECTROGRAM_FILENAME, args, phase)


def handleImage(fileName, args, phase=None):
    spectrogram, sampleRate = loadSpectrogram(fileName)
    audio = spectrogramToAudioFile(spectrogram, fftWindowSize=FFT, phaseIterations=ITER)

    sanityCheck, phase = audioFileToSpectrogram(audio, fftWindowSize=FFT)
    saveSpectrogram(sanityCheck, fileName + fileSuffix("Output Spectrogram", fft=FFT, iter=ITER, sampleRate=sampleRate) + ".png")

    saveAudioFile(audio, fileName + fileSuffix("Output", fft=FFT, iter=ITER) + ".wav", sampleRate)

