
import os
import _pickle as cPickle
import pickle
import numpy as np

n = 499

with open('ukb_data/data_5s.pkl', 'rb') as f:
    data = pickle.load(f)
data = data[:n]

new_data = []

for i, array in enumerate(data):
    print(i)

    new_array = np.empty([array.shape[0], 7])

    for j, epoch in enumerate(array):

        accx = epoch[1]
        accy = epoch[2]
        accz = epoch[3]

        xangle = np.arctan(accx/np.sqrt(accx**2 + accz**2))*180/np.pi
        yangle = np.arctan(accy/np.sqrt(accx**2 + accz**2))*180/np.pi
        zangle = np.arctan(accz/np.sqrt(accx**2 + accy**2))*180/np.pi

        epoch = np.append(epoch, xangle)
        epoch = np.append(epoch, yangle)
        epoch = np.append(epoch, zangle)

        new_array[j] = epoch

    new_data.append(new_array)

breakpoint()

with open('data_5s_angles.pkl', 'wb') as f:
    pickle.dump(new_data, f)
