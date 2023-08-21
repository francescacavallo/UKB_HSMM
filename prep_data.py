import os
import pandas as pd
import pdb
import pickle
from sklearn.preprocessing import LabelEncoder
import winsound

# Get data quality list
# data_file = "G:/UKB_data/3_Data/1_Main_dataset/ukb42402.csv"
# quality_df = pd.read_csv(data_file, usecols=['90015-0.0', 'eid'], index_col='eid')
# quality_df = quality_df.drop(1000015) #1049295

# Get 3 axis from epoch files and labels from TimeSeries
# This is the data extracted using https://github.com/OxWearables/biobankAccelerometerAnalysis.
# The methods are identical, only difference is that we use 5-second epochs.
folder ='D:/ukb/accelerometer-dwnld/dec2021_5s'
files_list = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
all_data = []
all_labels = []

for filepath in files_list:

    filename = os.path.basename(filepath)

    if int(filename[:7]) in quality_df.index:

        if 'epoch' in filename: # get x, y, z axis and magnitude
            df = pd.read_csv(filepath, compression='gzip', usecols=range(1,114))#['enmoTrunc', 'xMean', 'yMean', 'zMean'])
            all_data.append(df.to_numpy())

        if 'timeSeries' in filename: # get label
            df = pd.read_csv(filepath, usecols=['moderate', 'sedentary', 'sleep', 'tasks-light', 'walking'])
            # Check for bad labelling
            if df.moderate.describe()['max'] !=1:
                for nbeep in range(0,4):
                    winsound.Beep(1000, 700)
                breakpoint()

            # Label encoding
            for i, col in enumerate(df.columns):
                df[col] = df[col].apply(lambda x: i if x==1 else x)
            df['label'] = df.sum(axis=1) ## CAST TO INT

            all_labels.append(df['label'].to_numpy())


# save file in ukb_data folder
with open('labels_5s.pkl', 'wb') as f:
    pickle.dump(all_labels, f)
with open('data_5s_allfeats.pkl', 'wb') as f:
    pickle.dump(all_data, f)

for nbeep in range(0,4):
    winsound.Beep(1000, 700)
breakpoint()
