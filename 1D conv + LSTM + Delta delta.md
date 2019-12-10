```python
# IN[ ]:
import os
import librosa
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
features = np.empty([temp_features.size, 99, 39])

#adding delta & delta-delta coefficients
def d_d(arg):
    delta_delta_values = []
    for x in arg:
        delta_delta_coef = librosa.feature.delta(x, order=2)
        delta = librosa.feature.delta(x)
        all_feat = np.concatenate((x,delta_delta_coef,delta), axis=1)
        delta_delta_values.append(np.array(all_feat))
    return np.array(delta_delta_values)

delta_delta = d_d(temp_features)
delta_delta

##
# Coverting 1D Array of Features to a 3D Array
for i in range(0, delta_delta.size):
    if(delta_delta[i].size > features[i].size):
        result = np.resize(delta_delta[i],(99, 39))
    else:
        result = np.zeros(features[i].shape)
        result[:delta_delta[i].shape[0], :delta_delta[i].shape[1]] = delta_delta[i]

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
model.add(Conv1D(64,3, activation='relu', strides=1, padding='same'))
model.add(MaxPooling1D(3))
model.add(LSTM(128, return_sequences=True, input_shape=(shape[1], shape[2])))
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

# Calculation and outputing accuracy
y_pred = model.predict_classes(x_val, verbose=False)
print(accuracy_score(y_val, y_pred))

## 1D Conv. 80:20, batch=64  : acc:81.7% (first)
## 1D Conv LSTM. 80:20, batch=55 : acc:89.997% (first)
## 1D Conv LSTM. 70:30, batch=32 : acc:89.695% (first)
## 1D Conv LSTM. 80:20, batch=32 : acc:90.076% (first)


```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_1 (Conv1D)            (None, 99, 16)            640       
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 99, 32)            1568      
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 99, 64)            6208      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 33, 64)            0         
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 33, 128)           98816     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 33, 65)            50440     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 33, 65)            0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2145)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 35)                75110     
    =================================================================
    Total params: 232,782
    Trainable params: 232,782
    Non-trainable params: 0
    _________________________________________________________________
    Train on 75859 samples, validate on 18965 samples
    Epoch 1/100
    75859/75859 [==============================] - 911s 12ms/step - loss: 1.0461 - acc: 0.6955 - val_loss: 0.6337 - val_acc: 0.8118
    
    Epoch 00001: val_acc improved from -inf to 0.81181, saving model to final_model.hdf5
    Epoch 2/100
    75859/75859 [==============================] - 884s 12ms/step - loss: 0.5130 - acc: 0.8461 - val_loss: 0.4712 - val_acc: 0.8585
    
    Epoch 00002: val_acc improved from 0.81181 to 0.85848, saving model to final_model.hdf5
    Epoch 3/100
    75859/75859 [==============================] - 873s 12ms/step - loss: 0.3973 - acc: 0.8787 - val_loss: 0.4384 - val_acc: 0.8706
    
    Epoch 00003: val_acc improved from 0.85848 to 0.87055, saving model to final_model.hdf5
    Epoch 4/100
    75859/75859 [==============================] - 832s 11ms/step - loss: 0.3344 - acc: 0.8963 - val_loss: 0.3918 - val_acc: 0.8828
    
    Epoch 00004: val_acc improved from 0.87055 to 0.88278, saving model to final_model.hdf5
    Epoch 5/100
    75859/75859 [==============================] - 881s 12ms/step - loss: 0.2876 - acc: 0.9100 - val_loss: 0.3998 - val_acc: 0.8822
    
    Epoch 00005: val_acc did not improve from 0.88278
    Epoch 6/100
    75859/75859 [==============================] - 873s 12ms/step - loss: 0.2562 - acc: 0.9196 - val_loss: 0.3926 - val_acc: 0.8873
    
    Epoch 00006: val_acc improved from 0.88278 to 0.88727, saving model to final_model.hdf5
    Epoch 7/100
    75859/75859 [==============================] - 870s 11ms/step - loss: 0.2275 - acc: 0.9287 - val_loss: 0.3710 - val_acc: 0.8938
    
    Epoch 00007: val_acc improved from 0.88727 to 0.89375, saving model to final_model.hdf5
    Epoch 8/100
    75859/75859 [==============================] - 878s 12ms/step - loss: 0.2033 - acc: 0.9349 - val_loss: 0.3699 - val_acc: 0.8969
    
    Epoch 00008: val_acc improved from 0.89375 to 0.89692, saving model to final_model.hdf5
    Epoch 9/100
    75859/75859 [==============================] - 880s 12ms/step - loss: 0.1874 - acc: 0.9408 - val_loss: 0.3918 - val_acc: 0.8974
    
    Epoch 00009: val_acc improved from 0.89692 to 0.89744, saving model to final_model.hdf5
    Epoch 10/100
    75859/75859 [==============================] - 886s 12ms/step - loss: 0.1768 - acc: 0.9439 - val_loss: 0.3882 - val_acc: 0.8985
    
    Epoch 00010: val_acc improved from 0.89744 to 0.89850, saving model to final_model.hdf5
    Epoch 11/100
    75859/75859 [==============================] - 852s 11ms/step - loss: 0.1589 - acc: 0.9483 - val_loss: 0.3753 - val_acc: 0.8976
    
    Epoch 00011: val_acc did not improve from 0.89850
    Epoch 12/100
    75859/75859 [==============================] - 866s 11ms/step - loss: 0.1502 - acc: 0.9520 - val_loss: 0.4044 - val_acc: 0.8959
    
    Epoch 00012: val_acc did not improve from 0.89850
    Epoch 13/100
    75859/75859 [==============================] - 875s 12ms/step - loss: 0.1430 - acc: 0.9538 - val_loss: 0.4080 - val_acc: 0.8983
    
    Epoch 00013: val_acc did not improve from 0.89850
    Epoch 14/100
    75859/75859 [==============================] - 867s 11ms/step - loss: 0.1356 - acc: 0.9566 - val_loss: 0.4223 - val_acc: 0.8961
    
    Epoch 00014: val_acc did not improve from 0.89850
    Epoch 15/100
    75859/75859 [==============================] - 869s 11ms/step - loss: 0.1279 - acc: 0.9584 - val_loss: 0.4094 - val_acc: 0.8992
    
    Epoch 00015: val_acc improved from 0.89850 to 0.89918, saving model to final_model.hdf5
    Epoch 16/100
    75859/75859 [==============================] - 868s 11ms/step - loss: 0.1196 - acc: 0.9608 - val_loss: 0.4200 - val_acc: 0.9007
    
    Epoch 00016: val_acc improved from 0.89918 to 0.90066, saving model to final_model.hdf5
    Epoch 17/100
    75859/75859 [==============================] - 878s 12ms/step - loss: 0.1168 - acc: 0.9625 - val_loss: 0.4426 - val_acc: 0.8983
    
    Epoch 00017: val_acc did not improve from 0.90066
    Epoch 18/100
    75859/75859 [==============================] - 881s 12ms/step - loss: 0.1128 - acc: 0.9631 - val_loss: 0.4341 - val_acc: 0.8989
    
    Epoch 00018: val_acc did not improve from 0.90066
    Epoch 00018: early stopping



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-2-5b6c0aa7ed88> in <module>
        127 # Calculation and outputing accuracy
        128 y_pred = model.predict_classes(x_val, verbose=False)
    --> 129 print(accuracy_score(y_val, y_pred))
        130 
        131 ## 1D Conv. 80:20, batch=64  : acc:81.7% (first)


    ~\Documents\Python\Anaconda\lib\site-packages\sklearn\metrics\classification.py in accuracy_score(y_true, y_pred, normalize, sample_weight)
        174 
        175     # Compute accuracy for each possible representation
    --> 176     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
        177     check_consistent_length(y_true, y_pred, sample_weight)
        178     if y_type.startswith('multilabel'):


    ~\Documents\Python\Anaconda\lib\site-packages\sklearn\metrics\classification.py in _check_targets(y_true, y_pred)
         79     if len(y_type) > 1:
         80         raise ValueError("Classification metrics can't handle a mix of {0} "
    ---> 81                          "and {1} targets".format(type_true, type_pred))
         82 
         83     # We can't have more than one value on y_type => The set is no more needed


    ValueError: Classification metrics can't handle a mix of multilabel-indicator and multiclass targets



```python
# Calculation and outputing accuracy
y_pred = model.predict_classes(x_val, verbose=False)
print(accuracy_score(y_val, y_pred))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-4-0b9bd47517e1> in <module>
          1 # Calculation and outputing accuracy
          2 y_pred = model.predict_classes(x_val, verbose=False)
    ----> 3 print(accuracy_score(y_val, y_pred))
    

    ~\Documents\Python\Anaconda\lib\site-packages\sklearn\metrics\classification.py in accuracy_score(y_true, y_pred, normalize, sample_weight)
        174 
        175     # Compute accuracy for each possible representation
    --> 176     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
        177     check_consistent_length(y_true, y_pred, sample_weight)
        178     if y_type.startswith('multilabel'):


    ~\Documents\Python\Anaconda\lib\site-packages\sklearn\metrics\classification.py in _check_targets(y_true, y_pred)
         79     if len(y_type) > 1:
         80         raise ValueError("Classification metrics can't handle a mix of {0} "
    ---> 81                          "and {1} targets".format(type_true, type_pred))
         82 
         83     # We can't have more than one value on y_type => The set is no more needed


    ValueError: Classification metrics can't handle a mix of multilabel-indicator and multiclass targets



```python
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
```
