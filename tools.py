# https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification


import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np 
import soundfile as sf
plt.cla()
plt.clf()
plt.close()
DataSetsPath = "./archive/Data/genres_original/"
genre = 'hiphop/'
filename = 'hiphop.00010.wav'

audioData = DataSetsPath + genre + filename 

def signal_plot(audioData):
    signal, sr = librosa.load(audioData,sr=44100 ) # sr * T 44100 * 30

    librosa.display.waveplot(signal,sr=sr)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.show()
    print(signal.shape)
    return signal, sr

# fft -> spectrum 
def fft_plot(signal,sr):
    fft = np.fft.fft(signal)
    magnitude= np.abs(fft)  
    frequency = np.linspace(0,sr,len(magnitude))


    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(magnitude)/2)]

    plt.plot(left_frequency,left_magnitude)
    plt.xlabel('Frequency')
    plt.ylabel('magnitude')
    plt.show()
    
    return left_frequency , left_magnitude


# short time fft - spectrumgram 
def log_spectrumgram_plot(signal,n_fft,hop_length):
    # n_fft = 2048
    # hop_length = 512 # shifting parameter
    stft = librosa.core.stft(signal,hop_length =hop_length, n_fft = n_fft)
    sr = 22050
    spectrogram = np.abs(stft)

    log_spectrumgram = librosa.amplitude_to_db(spectrogram)

    # librosa.display.specshow(log_spectrumgram,sr=sr,hop_length=hop_length)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.colorbar()
    # plt.show()
    return  log_spectrumgram

# short time fft - spectrumgram no log 
def spectrumgram_plot(signal,n_fft,hop_length):
    # n_fft = 2048
    # hop_length = 512 # shifting parameter
    sr = 22050
    stft = librosa.core.stft(signal,hop_length =hop_length, n_fft = n_fft)

    spectrogram = np.abs(stft)

    # librosa.display.specshow(spectrogram,sr=sr,hop_length=hop_length)
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.colorbar()
    # plt.show()
    return  spectrogram

# MFCCs
def mfcc_plot(signal,n_fft,hop_length,sr):
    MFCCs = librosa.feature.mfcc(signal,n_fft = n_fft, hop_length = hop_length, n_mfcc=13) 
    librosa.display.specshow(MFCCs,sr=sr,hop_length=hop_length)
    plt.xlabel('Time')
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.show()
    return MFCCs

def write_audio(signal, sr):
    sf.write('tets.wav',signal,sr)

if __name__ == '__main__':
    signal, sr = signal_plot(audioData)

    # write_audio(signal,sr)
    fft_plot(signal)
    n_fft = 2048
    hop_length = 512 # shifting parameter
    logSpectrum = log_spectrumgram_plot(signal,n_fft,hop_length)
    spectrumgram_plot(signal,n_fft,hop_length)
    mfcc_plot(signal,n_fft,hop_length,sr)





