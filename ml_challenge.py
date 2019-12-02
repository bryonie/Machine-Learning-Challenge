import os
import librosa, librosa.display   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

warnings.filterwarnings("ignore")

# Variable Declarations
train_audio_path = './wav/'
train_file_path = './train.csv'
test_file_path = './test.csv'
path_np = './path.npy'
feat_np = './feat.npy'
labels = os.listdir(train_audio_path)
all_data = []
all_rates = []
all_durations = []
all_freq = []
all_waves = []
avg_dur = 0

# SOME ASSUMPTIONS
# train.csv - This is training data to help the model
# learn, therefore, it will be used as train and validate
# test.csv - This is the data to test the model after
# training to check if will output the correct
# predictions
#

# Loading data files
trainData = pd.read_csv(train_file_path)
testData = pd.read_csv(test_file_path)

pathData = np.load(path_np)
featData = np.load(feat_np, allow_pickle=True)
print(type(pathData))
print(type(featData))

# mfccData = zip(pathData, featData)

# Defining dataframe with features and their paths
featFrame = pd.DataFrame(featData)
pathFrame = pd.DataFrame(pathData)
pathFrame.columns = ['path']
feat_path_frame = pd.concat([featFrame, pathFrame], axis=1, join='inner')
feat_path_frame.columns = ["features", "path"]

le = LabelEncoder()

# Defining dataframes with train data word(endcoded) x feature 
# and testdata with features
train_feature_label_frame = trainData.merge(feat_path_frame, how='inner', on = ["path"])
train_feature_label_frame = train_feature_label_frame.drop(columns = "path")
train_feature_label_frame["word"] = le.fit_transform(train_feature_label_frame["word"])
test_feature_label_frame = testData.merge(feat_path_frame, how='inner', on = ["path"])
# test_feature_label_frame = test_feature_label_frame.drop(columns = "path")

print(train_feature_label_frame)
print(test_feature_label_frame)

# feat_mean_frame = featFrame.mean(axis=0)
# feat_std_frame = featFrame.std(axis=0)
# feat_min_frame = featFrame.min(axis=0)
# feat_max_frame = featFrame.max(axis=0)

# feat_frame_total = pd.concat([feat_mean_frame.to_frame(),
# feat_std_frame.to_frame(),feat_min_frame.to_frame(),
# feat_max_frame.to_frame()], axis= 1,join="inner")
# feat_frame_total.columns = ["mean", "std", "min", "max"]

# print(feat_frame_total)

# Getting list of all commands in train
wordCount = trainData.groupby('word').count()
commands = list(wordCount["path"].keys())

y = np_utils.to_categorical(train_feature_label_frame["word"], 
num_classes=len(commands))
classes = list(le.classes_)

# Splitting data from train into train and test data
x_train, x_val, y_train, y_val = train_test_split(np.array
(train_feature_label_frame["features"]),np.array(y),
stratify=y,test_size = 0.2,random_state=777,shuffle=True)

print("X Train:- \n{}\nY Train:- \n{}".format(x_train, y_train))
print("X Validate:- \n{}\nY Validate:-\n {}".format(x_val, y_val))


# Building 1D Convolution model
Shape = train_feature_label_frame.shape
print(Shape)
inputs = Input(Shape)

#First Layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(commands), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

# Define loss function
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Setting up easy stoping and model checkpoints
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# TRAINING LE MODEL!

history=model.fit(x_train, y_train ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

# Plots a graph that shows the performance of the model training
# plt.plot(history.history['loss'], label='train') 
# plt.plot(history.history['val_loss'], label='test') 
# plt.legend() plt.show()

# Loding in the best model (Not sure of the significants)
model=load_model('best_model.hdf5')

# Defining prediction function

##
# @param {*} audio features or samples
# @return {*} Text prediction from audio
#
def predict(audio):
    prob=model.predict(audio.reshape(1, Shape[0], Shape[1]))
    index=np.argmax(prob[0])
    return classes[index]

##
#  @params {DataFrame}
#  @return {null}
#  Using the test_feature_label_frame DataFrame, we 
#  test the models prediction ability
def predictFromData(df):
    # Assuming the data is just like the data in
    # test_feature_label_frame with only one column
    # which are the features

    for i, j in df.iterrows():
        if("path" in i):
            print("Audio: {}".format(j))
        elif("features" in i):
            print("Text: {}".format(predict(j)))


predictFromData(test_feature_label_frame)



# x_train = np.array(train_feature_label_frame["feature"])
# x_test = np.array(test_feature_label_frame["feature"])
# y_train = np.array(train_feature_label_frame["word"])
# y_test = np.array()





# wordCount = trainData.groupby('word').count()
# commands = list(wordCount["path"].keys())

# print(commands)
# index = np.arange(len(commands))


# plt.figure(figsize=(30,5))
# plt.bar(commands, wordCount['path'])
# plt.xlabel('Commands')
# plt.xticks(index, commands, fontsize=12, rotation=60)
# plt.ylabel("Count")
# plt.show()



# samples, sample_rate = librosa.load(train_audio_path+'male.wav')
# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(samples, sample_rate, alpha=0.8)
# plt.show()
# ipd.Audio(samples, rate=sample_rate)

# print(path[0])
# print(feat[0])

# def Average(lst): 
#     return sum(lst) / len(lst) 

# for audioFile in os.listdir(train_audio_path):
#     samples, sample_rate = librosa.load(train_audio_path+audioFile, sr=10000)
#     rate, data = wavfile.read(train_audio_path+audioFile)
#     samples = librosa.resample(samples, sample_rate, target_sr=10000)
#     all_waves.append(samples)
#     all_data.append(data)
#     all_rates.append(rate)
#     all_freq.append(sample_rate)
#     all_durations.append(len(samples)/sample_rate)

# avg_dur = Average(all_durations)

# print(avg_dur)

x_train = [] # Numpy Array of audio data to train with
x_test = [] # Numpy Array of audio data to test
y_train = [] # Numpy array of labels (in this case commands) to train with
y_test = [] # Numpy array of labels (in this case commands) to test

# print(all_data[0])

# le = LabelEncoder()
# y = le.fit_transform(commands)
# classes = list(le.classes_)


