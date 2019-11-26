import os
import librosa, librosa.display   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

train_audio_path = './wav/'
train_file_path = './train.csv'
test_file_path = './test.csv'
path_np = './path.npy'
feat_np = './feat.npy'
all_data = []
all_rates = []
all_durations = []
all_freq = []
avg_dur = 0

trainData = pd.read_csv(train_file_path)
wordCount = trainData.groupby('word').count()
commands = list(wordCount["path"].keys())

# print(commands)
index = np.arange(len(commands))


plt.figure(figsize=(30,5))
plt.bar(commands, wordCount['path'])
plt.xlabel('Commands')
plt.xticks(index, commands, fontsize=12, rotation=60)
plt.ylabel("Count")
plt.show()



# samples, sample_rate = librosa.load(train_audio_path+'male.wav')
# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(samples, sample_rate, alpha=0.8)
# plt.show()
# ipd.Audio(samples, rate=sample_rate)



# path = np.load('./path.npy')
# feat = np.load('./feat.npy', allow_pickle=True)

# print(path[0])
# print(feat[0])

# def Average(lst): 
#     return sum(lst) / len(lst) 

# for audioFile in os.listdir(train_audio_path):
#     samples, sample_rate = librosa.load(train_audio_path+audioFile, sr=None)
#     rate, data = wavfile.read(train_audio_path+audioFile)
#     all_data.append(data)
#     all_rates.append(rate)
#     all_freq.append(sample_rate)
#     all_durations.append(len(samples)/sample_rate)

# avg_dur = Average(all_durations)

# print(avg_dur)

# print(all_data[0])