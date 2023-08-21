import pandas as pd
import numpy as np
from packages import hsmm_acc
from packages import preprocessing as prep
from pyhsmm.util.general import stateseq_hamming_error
import os
import _pickle as cPickle
import pickle
import matplotlib.pyplot as plt
import statistics as stats
import winsound

# Import data
with open('ukb_data/data_5s_angles.pkl', 'rb') as f:
    data = pickle.load(f)

# Choose features
n = 499 # n participants

new_data=[]
for array in data:
    new_array = array[1:, [0]] # : for all features, 0 for magnitude, 1:3 axis, 4:6 angles
    if new_array.ndim==1:
        new_array = new_array.reshape(new_array.shape[0], 1)
    new_data.append(new_array)

data = new_data[:n]

# Infer model
print('Inference started, model 6')

Nmax = 5
posteriormodel, ham_error = hsmm_acc.hsmm_train(data, Nmax, trunc=500, resampling=10, visualise=False, max_hamming=0.05)

# Plot Hamming error
plt.figure()
# color = iter(plt.cm.viridis(np.linspace(0, 1,5)))
# col = next(color)
plt.plot(ham_error, color='black')
plt.xlabel('Resampling iteration', fontsize=15)
plt.ylabel('Hamming error', fontsize=15)
plt.tick_params(labelsize=15)
plt.ylim(0,1)
plt.savefig('499_model6_hamming_error_magnitude.png')

# Save model
with open('499_model6_magnitude.pickle','wb') as outfile:
    cPickle.dump(posteriormodel, outfile, protocol=-1)

#winsound.Beep(1500, 2000)
