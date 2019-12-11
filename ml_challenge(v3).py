#!/usr/bin/env python
# coding: utf-8

# IN[ ]:
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, TimeDistributed, LSTM
from keras.models import Model, load_model, Sequential
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

# Loading data files
trainData = pd.read_csv(train_file_path)
testData = pd.read_csv(test_file_path)

pathData = np.load(path_np)
featData = np.load(feat_np, allow_pickle=True)


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
test_feature_label_frame = testData.merge(feat_path_frame, how='inner', on = ["path"])

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

##
# Coverting 1D Array of Features to a 3D Array
for i in range(0, temp_features.size):
    if(temp_features[i].size > features[i].size):
        result = np.resize(temp_features[i],(99, 13))
    else:
        result = np.zeros(features[i].shape)
        result[:temp_features[i].shape[0], :temp_features[i].shape[1]] = temp_features[i]

    features[i] = result

# Splitting data from train into train and test data
x_train, x_val, y_train, y_val = train_test_split(
    features,
labels,stratify = labels, train_size = 0.8, test_size = 0.2,
random_state=777,shuffle=True)

# Building 1D Convolution model
shape = x_train.shape
inputs = Input(shape=(shape[1], shape[2]))

##
# 1D Convolution + LSTM Modelmodel working with data 80:20, batch=32
model = Sequential()
model.add(Conv1D(16,1, activation='relu', strides=1, padding='same', input_shape=(shape[1], shape[2])))
model.add(Conv1D(32,3, activation='relu', strides=1, padding='same'))
model.add(Conv1D(128,3, activation='relu', strides=1, padding='same'))
model.add(MaxPooling1D(3))
model.add(LSTM(512, return_sequences=True, input_shape=(shape[1], shape[2])))
model.add(LSTM(65, return_sequences=True))
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(35, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Setting up easy stoping and model checkpoints
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
 patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('final_model.hdf5', monitor='val_acc', 
verbose=1, save_best_only=True, mode='max')

# Fit the model to train

history=model.fit(x_train, y_train ,epochs=100, callbacks=[es,mc], 
batch_size=32, validation_data=(x_val,y_val), verbose=1)

## 1D Conv(3). 80:20, batch=64  : acc:81.7%
## 1D Conv(3) LSTM(2). 80:20, batch=55 : acc:89.997%
## 1D Conv(3) LSTM(2). 70:30, batch=32 : acc:89.695%
## 1D Conv(3) LSTM(2). 80:20, batch=32 : acc:90.076%
## 1D Conv(4) LSTM(3). 80:20, batch=64 : acc:90.757%

results = {
    "Path" : [],
    "Word": []
}

Size = test_feature_label_frame["features"].size

for i in range(0,Size):
    if(test_feature_label_frame["features"][i].shape[0] > 99):
        features = np.resize(test_feature_label_frame["features"][i], (99, 13))
    else:
        features = np.zeros((99, 13))
        features[
            :test_feature_label_frame["features"][i].shape[0],
            :test_feature_label_frame["features"][i].shape[1]
        ] = test_feature_label_frame["features"][i]

    # Using the model to predict
    prob=model.predict(features.reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    # print("Prediction: {}".format(classes[index]))
    results["Path"].append(test_feature_label_frame["path"][i])
    results["Word"].append(classes[index])
    p = (i / Size) * 100
    print ("Creating Predict File: {}%".format(round(p,2)), end="\r")

# Saving the results to a csv file
Results = pd.DataFrame(results)
print("Saving Predict File")
Results.to_csv('results.csv', index=False)

