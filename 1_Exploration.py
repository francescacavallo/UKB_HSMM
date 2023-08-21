import pandas as pd
import numpy as np
from packages import preprocessing as prep
import os
import pickle
import matplotlib.pyplot as plt
import statistics as stats

# Import data
with open('ukb_data/data_5s.pkl', 'rb') as f:
    all_data = pickle.load(f)
with open('ukb_data/labels_5s.pkl', 'rb') as f:
    all_labels = pickle.load(f)


# Plot magnitude and/or axis
for n in range(0, 4):
    plt.figure()
    plt.plot(all_data[0][:17000, n], linewidth=1, color='black')#
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    plt.xlabel('Sample number', fontsize=15)
    plt.ylabel('Acceleration', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.xlim(0, 17000)
    plt.ylim(-1, 1)
    if n==0: plt.ylim(0,1.2)
    plt.savefig(str(n)+'_axis.png')

# new_labels = []
# for array in all_labels:
#     new_labels.append
activities =['Moderate activity', 'Sedentary', 'Sleep', 'Light tasks', 'Walking']

# Histogram: duration distribution (by observation)
activity_len_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
for array in all_labels:
    count = 0
    for i in range(1, array.shape[0]):
        if array[i] == array[i-1]:
            count +=1
        else:
            activity_len_dict[array[i-1]].append(count)
            count = 0

color = iter(plt.cm.viridis(np.linspace(0, 1,5)))
fig, ax = plt.subplots(3,2, figsize=(10,10))
ax = ax.ravel()
for key, item in activity_len_dict.items():
    col = next(color)
    ax[key].hist(item, bins=1000, color=col)
    ax[key].set_title(activities[key], fontsize=15)
    ax[key].tick_params(labelsize=15)
fig.delaxes(ax[-1])
plt.tight_layout()
plt.show()

# Mean state duration
count = 0
sum = 0
for key in activity_len_dict:
    count += 1
    sum += stats.mean(activity_len_dict[key])
print('Mean state duration: ', round(sum*5/count), 'seconds')
#breakpoint()

# Boxplot: time features for each label
all_df = pd.DataFrame()
for i, array in enumerate(all_data):
    df = pd.DataFrame(array, columns=['Magnitude', 'Xaxis','Yaxis','Zaxis'])#, 'Mean', 'STD', 'Variance', 'Median', 'Min', 'Max', '25q', '75q', 'MAD'])
    #df = df.drop('Variance', axis=1)
    df['label'] = all_labels[i]
    all_df = pd.concat([all_df, df], axis=0)
#breakpoint()
fig, axs = plt.subplots(2,2, figsize=(10, 5), sharey=False)
all_df.boxplot(by='label', ax=axs)
plt.tight_layout()
plt.show()
