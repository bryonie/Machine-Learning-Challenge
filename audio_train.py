#!/usr/bin/env python
# coding: utf-8

# IN[ ]:
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
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

warnings.filterwarnings("ignore")

# Variable Declarations
# wandb.init()
# config = wandb.config

# config.max_len = 11
# config.buckets = 20
# config.epochs = 50
# config.bacth_size: 64

train_audio_path = './wav/'
train_file_path = './train.csv'
test_file_path = './test.csv'
path_np = './path.npy'
feat_np = './feat.npy'

# Loading data files
trainData = pd.read_csv(train_file_path)
testData = pd.read_csv(test_file_path)

pathData = np.load(path_np)
featData = np.load(feat_np, allow_pickle=True)
# featArray = np.empty([featData.size, 99, 13])

# for i in range(0, featData.size):
#     if(featData[i].size > featArray[i].size):
#         result = np.resize(featData[i],(99, 13))
#     else:
#         result = np.zeros(featArray[i].shape)
#         result[:featData[i].shape[0], :featData[i].shape[1]] = featData[i]

#     featArray[i] = result

# featData = featData.reshape(
#     featData,
#     (

#     )
# )
# feat_path_data = np.dstack((featData, pathData))
# featArray = np.resize(featArray, (featArray.shape[0], featArray.shape[1]))
print(type(pathData))
print(type(featData))
# print(type(featArray))
print(featData.shape)
# print(featArray.shape)
# print(featData)
# print(featArray)


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
# train_feature_label_frame["features"] = train_feature_label_frame["features"].values.reshape(
#     train_feature_label_frame["features"].size,
#     99
# )
test_feature_label_frame = testData.merge(feat_path_frame, how='inner', on = ["path"])

# print(train_feature_label_frame["features"])

trainData["length"] = trainData.groupby('word')['word'].transform('count')
classes = list(np.unique(trainData.word))
num_class = len(classes)

class_dist = trainData.groupby(['word'])['length'].mean()
prob_dist = class_dist / class_dist.sum()

labels = train_feature_label_frame["word"]
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 
num_classes=len(classes))
temp_features = train_feature_label_frame["features"].to_numpy()
features = np.empty([temp_features.size, 99, 13])
for i in range(0, temp_features.size):
    if(temp_features[i].size > features[i].size):
        result = np.resize(temp_features[i],(99, 13))
    else:
        result = np.zeros(features[i].shape)
        result[:temp_features[i].shape[0], :temp_features[i].shape[1]] = temp_features[i]

    features[i] = result
print(len(temp_features))
# features = np.resize(temp_features, (temp_features.shape[0], 1))
# features = features.reshape(-1, features.shape[0], features.shape[1])
print(features.shape)
print(train_feature_label_frame["features"].shape)

# Splitting data from train into train and test data
x_train, x_val, y_train, y_val = train_test_split(
    features,
labels,stratify = labels, train_size = 0.8, test_size = 0.2,
random_state=777,shuffle=True)

# # # x_train = x_train.to_numpy()
# # # x_val = x_val.to_numpy()
# # print(x_train)
# # print(y_train)

# x_train = np.resize(x_train, (35, x_train.shape[0], x_train.shape[1]))#x_train.reshape(35, x_train.shape[0], x_train.shape[1])
print(x_train.shape)
# # print(x_val.shape)

# # # x_train = x_train.reshape(1, x_train.shape[0], 1)
# # # x_val = x_val.reshape(1, x_val.shape[0], 1)

# # # print(x_train.shape)
# # # print(x_val.shape)

# # y_train_hot = np_utils.to_categorical(y_train)
# # y_test_hot = np_utils.to_categorical(y_val)

# Building 1D Convolution model
shape = x_train.shape
inputs = Input(shape=(shape[1], shape[2]))

# #First Layer
# conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
# conv = MaxPooling1D(3)(conv)
# conv = Dropout(0.3)(conv)

# #Second layer
# conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
# conv = MaxPooling1D(3)(conv)
# conv = Dropout(0.3)(conv)

# #Third layer
# conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
# conv = MaxPooling1D(3)(conv)
# conv = Dropout(0.3)(conv)

# #Fourth layer
# conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
# conv = MaxPooling1D(3)(conv)
# conv = Dropout(0.3)(conv)

# #Flatten layer
# conv = Flatten()(conv)

# #Dense Layer 1
# conv = Dense(256, activation='relu')(conv)
# conv = Dropout(0.3)(conv)

# #Dense Layer 2
# conv = Dense(128, activation='relu')(conv)
# conv = Dropout(0.3)(conv)

# outputs = Dense(len(classes), activation='softmax')(conv)

# model = Model(inputs, outputs)
# model.summary()

# # Define loss function
# model.compile(loss='categorical_crossentropy',optimizer='adam',
# metrics=['accuracy'])

# Setting up easy stoping and model checkpoints
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
 patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', 
verbose=1, save_best_only=True, mode='max')

# history=model.fit(x_train, y_train ,epochs=100, callbacks=[es,mc], 
# batch_size=64, validation_data=(x_val,y_val), verbose=1)

model = Sequential()
model.add(Conv1D(16,1, activation='relu', strides=1, padding='same', input_shape=(shape[1], shape[2])))
model.add(Conv1D(32,3, activation='relu', strides=1, padding='same'))
model.add(Conv1D(64,3, activation='relu', strides=1, padding='same'))
# model.add(Conv2D(128,(5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dense(35, activation='relu'))
model.add(Dense(35, activation='softmax'))
model.summary()

# sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history=model.fit(x_train, y_train ,epochs=100, callbacks=[es,mc], 
batch_size=64, validation_data=(x_val,y_val), verbose=1)

