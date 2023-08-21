import pandas as pd
import numpy as np
from packages import hsmm_acc
from packages import preprocessing as prep
from pyhsmm.util.general import stateseq_hamming_error
import os
import pickle
import _pickle as cPickle
import matplotlib.pyplot as plt
import statistics as stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, balanced_accuracy_score

# Import model
with open('Models/10/model2_magnitude.pickle','rb') as file:
    posteriormodel = cPickle.load(file)

# Import data
with open('ukb_data/data_5s.pkl', 'rb') as f:
    data = pickle.load(f)
data = data[:10]

with open('ukb_data/labels_5s.pkl', 'rb') as f:
    labels = pickle.load(f)
labels=labels[:10]

# Get predicted states for each participant
states = posteriormodel.stateseqs
breakpoint()

# Order states based on acceleration magnitude
means = pd.DataFrame([o.params['mu'] for o in posteriormodel.obs_distns])
new_idx =  means.iloc[:,0].sort_values(ascending=False).index.values
lookup = tuple(zip(range(0,5), new_idx))
print(lookup)

for a in states:
    a_copy = np.copy(a)
    for row in lookup:
        a[a_copy == row[0]] = row[1]

# Order labels by magnitude
all_df = pd.DataFrame()
for i, array in enumerate(data):
    df = pd.DataFrame(array, columns=['Magnitude', 'Xaxis','Yaxis','Zaxis'])
    df['label'] = labels[i]
    all_df = pd.concat([all_df, df], axis=0)
new_idx = all_df.groupby('label').mean()['Magnitude'].sort_values(ascending=False).index.values
lookup = tuple(zip(range(0,5), new_idx))
print(lookup)

for a in labels:
    a_copy = np.copy(a)
    for row in lookup:
        a[a_copy == row[0]] = row[1]

# Metrics
y_true = np.concatenate(labels).ravel()
y_pred = np.concatenate(states).ravel()

print('Balanced accuracy', balanced_accuracy_score(y_true, y_pred))
print('Recall score', recall_score(y_true, y_pred, average=None))
print('Precision score', precision_score(y_true, y_pred, average=None))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()
