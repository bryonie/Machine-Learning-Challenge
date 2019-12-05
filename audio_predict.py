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

test_feature_label_frame = testData.merge(feat_path_frame, how='inner', on = ["path"])

conv_model=load_model('best_model.hdf5')
conv_lstm_model=load_model('best_model_conv-lstm.hdf5')
start = 20
stop = 30

for i in range(start,stop):
    # Conv 1D Predict word
    prob=conv_model.predict(test_feature_label_frame["features"][i].reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    print(classes[index])

    # Conv 1D & LSTM Predict word
    prob=conv_lstm_model.predict(test_feature_label_frame["features"][i].reshape(-1, 99, 13))
    index=np.argmax(prob[0])
    print(classes[index])

    # Play Audio
    wave_obj = sa.WaveObject.from_wave_file(train_audio_path+test_feature_label_frame["path"][i])
    play_obj = wave_obj.play()
    play_obj.wait_done()


