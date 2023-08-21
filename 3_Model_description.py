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

# Import model
with open('Models/499/499_model6_magnitude.pickle','rb') as file:
    model = cPickle.load(file)

# col_names = ['Magnitude', 'x_angle', 'y_angle', 'z_angle']
# col_names = ['Magnitude', 'x_axis', 'y_axis', 'z_axis']
col_names = ['Magnitude']

# Plot states_sequence on magnitude only
n_samples = 3000

# Single plot
data = model.states_list[0].data[0:n_samples]
model.plot_stateseq(0, plot_slice=slice(n_samples))
plt.plot(data, linewidth=0.8, color='black')
plt.ylabel('Acceleration (mg)', fontsize=40)
plt.xlim(0, n_samples)
plt.ylim(data.min(), data.max())
plt.xlabel('Sample number', fontsize=40)
plt.tick_params(labelsize=30, length=5)
plt.show()
breakpoint()

# Subplots
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10))

data = model.states_list[0].data[0:n_samples]
ax1.plot(data, linewidth=1, color='black')#
ax1.tick_params(labelsize=20, length=5)
ax1.set_ylabel('Acceleration (mg)', fontsize=20)
ax1.set_xlim(0, n_samples)
ax1.set_ylim(data.min(), data.max())

model.plot_stateseq(0, plot_slice=slice(n_samples), ax = ax2)
ax2.tick_params(labelsize=20, length=5)
ax2.set_ylabel('Acceleration (mg)', fontsize=20)
for tk in ax2.get_yticklabels():
        tk.set_visible(True)
ax2.set_xlim(0, n_samples)
ax2.set_ylim(data.min(), data.max())

plt.xlabel('Sample number', fontsize=20)
breakpoint()
plt.savefig('499_model6_magnitude_stateseq.png')

# Gaussian parameters
means = pd.DataFrame([o.params['mu'] for o in model.obs_distns], columns=col_names)
sigmas = pd.DataFrame([o.params['sigma'][0] for o in model.obs_distns], columns=col_names)

# Duration
durations = pd.DataFrame([o.params['lmbda'] for o in model.dur_distns], columns=['Duration'])
durations['duration_sec'] = ["{} sec".format(int(round(x*5))) for x in durations['Duration']]

concat = pd.concat([means, sigmas], axis=1, keys=['mean', 'sigma'])
concat = concat.reorder_levels(order=[1,0], axis=1)
concat[('duration', 'lmbda')] = durations['duration_sec']
concat = concat[concat.columns.sort_values()]
print(concat)

# Transition probablilites
df_trans = pd.DataFrame(model.trans_distn.trans_matrix)
print(df_trans)
