import pandas as pd
import numpy as np
from packages import hsmm_acc
from packages import preprocessing as prep
from pyhsmm.util.general import stateseq_hamming_error
import os
import pickle
import matplotlib.pyplot as plt
import statistics as stats


folder = os.path.join(os.getcwd(), 'ukb_data')
files_list = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


fs = 52                 # sampling frequency
epochLength = 5*fs      # select epoch length for 5s epoch
all_data = []
all_labels =[]
all_data_5s = []
all_labels_5s =[]

if dataset == 'ukb':
    # keep subjets with 'good' accel data

if dataset == 'irvine':

    folder = os.path.join(os.getcwd(), 'data')
    files_list = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))

    for file in files_list:

        df = pd.read_csv(file, names=['sample','x', 'y', 'z', 'label']).drop('sample', axis=1)
        labels = df.label
        df = df.drop('label', axis=1)
        df = prep.magnitude(df)
        df = prep.filter(df, filterOrder=4, fCutoff=0.5)

        labels_5s = prep.get_epoch_labels(labels, epochLength)

        df_5s = pd.DataFrame()
        df_5s['magnitude'] = prep.epoch_filtered(df, epochLength=epochLength)
        df_5s['Xaxis'] = prep.epoch_axis(df, epochLength=epochLength)[0]
        df_5s['Yaxis'] = prep.epoch_axis(df, epochLength=epochLength)[1]
        df_5s['Zaxis'] = prep.epoch_axis(df, epochLength=epochLength)[2]
        df_5s = prep.extract_epoch_features(df, df_5s, epochLength=epochLength)

        all_labels_5s.append(labels_5s.to_numpy())
        all_labels.append(labels.to_numpy())
        all_data.append(df.to_numpy())
        all_data_5s.append(df_5s.to_numpy())

with open('labels.pkl', 'wb') as f:
    pickle.dump(all_labels, f)
with open('labels_5s.pkl', 'wb') as f:
    pickle.dump(all_labels_5s, f)
with open('data.pkl', 'wb') as f:
    pickle.dump(all_data, f)
with open('data_5s.pkl', 'wb') as f:
    pickle.dump(all_data_5s, f)
