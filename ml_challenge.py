import os
import librosa, librosa.display   #for audio processing
import speech_recognition as sr
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")

train_audio_path = './wav/'
# samples, sample_rate = librosa.load(train_audio_path+'male.wav')
# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(samples, sample_rate, alpha=0.8)
# plt.show()
# ipd.Audio(samples, rate=sample_rate)

all_data = []
all_rates = []
all_durations = []
all_freq = []
avg_dur = 0

# path = np.load('./path.npy')
# feat = np.load('./feat.npy', allow_pickle=True)

# print(path[0])
# print(feat[0])

def Average(lst): 
    return sum(lst) / len(lst) 

for audioFile in os.listdir('./wav'):
    samples, sample_rate = librosa.load(train_audio_path+audioFile, sr=None)
    rate, data = wavfile.read(train_audio_path+audioFile)
    all_data.append(data)
    all_rates.append(rate)
    all_freq.append(sample_rate)
    all_durations.append(len(samples)/sample_rate)

avg_dur = Average(all_durations)

print(avg_dur)

# print(all_data[0])