import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
from keras.models import load_model

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

classes = list(np.unique(trainData.word))

featFrame = pd.DataFrame(featData)
pathFrame = pd.DataFrame(pathData)
pathFrame.columns = ['path']
feat_path_frame = pd.concat([featFrame, pathFrame], axis=1, join='inner')
feat_path_frame.columns = ["features", "path"]

test_feature_label_frame = testData.merge(feat_path_frame, how ='inner', on = ["path"])

conv_model=load_model('best_model.hdf5')
conv_lstm_model=load_model('best_model_conv-lstm.hdf5')
conv_lstm_70_30_model=load_model('best_model_conv-lstm(70_30).hdf5')
conv_lstm_bat32_model=load_model('best_model_conv-lstm(b32).hdf5')
start = 0
stop = test_feature_label_frame["features"].size
results = {
    "Path" : [],
    "Word": []
}

for i in range(start,stop):
    if(test_feature_label_frame["features"][i].shape[0] > 99):
        features = np.resize(test_feature_label_frame["features"][i], (99, 13))
    else:
        features = np.zeros((99, 13))
        features[
            :test_feature_label_frame["features"][i].shape[0],
            :test_feature_label_frame["features"][i].shape[1]
        ] = test_feature_label_frame["features"][i]

    # Conv 1D Predict word
    prob=conv_model.predict(features.reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    print("Conv 1D: {}".format(classes[index]))

    # Conv 1D & LSTM Predict word
    prob=conv_lstm_model.predict(features.reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    print("Conv 1D - LSTM: {}".format(classes[index]))

    # Conv 1D & LSTM train 70 test 30 Predict word
    prob=conv_lstm_70_30_model.predict(features.reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    print("Conv 1D - LSTM (70:30): {}".format(classes[index]))

    # Conv 1D & LSTM batch size 32 Predict word
    prob=conv_lstm_bat32_model.predict(features.reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    print("Conv 1D - LSTM (batch 32): {}".format(classes[index]))
    results["Path"].append(test_feature_label_frame["path"][i])
    results["Word"].append(classes[index])

    # # Play Audio
    # wave_obj = sa.WaveObject.from_wave_file(train_audio_path+test_feature_label_frame["path"][i])
    # play_obj = wave_obj.play()
    # play_obj.wait_done()

Results = pd.DataFrame(results)

Results.to_csv('result_test.csv')


