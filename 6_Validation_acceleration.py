import pandas as pd
import numpy as np
import os
import pickle
import _pickle as cPickle
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import levene, shapiro
from sklearn.metrics.pairwise import cosine_similarity

n = 499

# Import model
with open('Models/'+str(n)+'/model1_magnitude.pickle','rb') as file:
    model = cPickle.load(file)

# Import data
with open('ukb_data/data_5s.pkl', 'rb') as f:
    data = pickle.load(f)
data = data[:n]

with open('ukb_data/labels_5s.pkl', 'rb') as f:
    labels = pickle.load(f)
labels=labels[:n]

# Import inferred stateseqs
states = model.stateseqs

# Order states by acceleration magnitude (from highest to lowest)
means = pd.DataFrame([o.params['mu'] for o in model.obs_distns])
new_idx =  means.iloc[:,0].sort_values(ascending=False).index.values
lookup = tuple(zip(range(0,5), new_idx))
print(lookup)

for a in states:
    a_copy = np.copy(a)
    for row in lookup:
        a[a_copy == row[0]] = row[1]

# Get "true" time spent in each activity
tot_counts = []
for p in labels:
    unique, counts = np.unique(p, return_counts=True)
    tot_counts.append(counts)

true_time = pd.DataFrame(tot_counts, columns=['Moderate activity', 'Sedentary', 'Sleep', 'Light tasks', 'Walking'])
true_time = true_time.apply(lambda x: x*5/60/60/7)
true_time = true_time.reindex(['Moderate activity', 'Walking', 'Light tasks', 'Sedentary', 'Sleep'], axis=1)

# Get "pedicted" time spent in each activity
tot_counts = []
for p in states:
    unique, counts = np.unique(p, return_counts=True)
    tot_counts.append(counts)
pred_time = pd.DataFrame(tot_counts)
pred_time = pred_time.apply(lambda x: x*5/60/60/7)

# Order states by average acceleration
accel = pd.DataFrame([o.params['mu'] for o in model.obs_distns])
new_idx =  accel.iloc[:,0].sort_values(ascending=False).index.values

pred_time = pred_time.reindex(new_idx, axis=1)
pred_time.columns = ['Sleep', 'Sedentary', 'Light tasks', 'Moderate activity', 'Walking']

# Cosine similarity
pred_time_temp = pred_time.dropna()
true_time_temp = true_time[true_time.index.isin(pred_time_temp.index)]
n = true_time_temp.shape[0]

for activity in true_time.columns:
    metric = cosine_similarity(true_time_temp[activity].values.reshape(1,n),
                               pred_time_temp[activity].values.reshape(1,n))
    print('Cosine similarity for %s = %f' %(activity, metric))


# Predict BMI from activity
df = pd.read_csv('G:/UKB_data/3_Data/5_Subsets/00_prediabetes_markers_assessmentCentre.csv', index_col='eid')
eid = pd.read_csv(os.path.join(os.getcwd(), 'ukb_data/eid.csv'), header=None)
df = df[df.index.isin(eid.iloc[:,0])]

pred_time.index = eid.values.reshape(1,eid.shape[0])[0]
true_time.index = eid.values.reshape(1,eid.shape[0])[0]

df_true = pd.concat([df, true_time], axis=1, join='inner').dropna()
df_pred = pd.concat([df, pred_time], axis=1, join='inner').dropna()

for activity in true_time.columns:
    print(pg.linear_regression(df_true[activity], df_true[' BMI']))
    print(pg.linear_regression(df_pred[activity], df_pred[' BMI']))


# Boxplots
fig, ax = plt.subplots(1,2)
true_time.boxplot(ax=ax.flatten()[0])
pred_time.boxplot(ax=ax.flatten()[1])
plt.show()



# ttest
# for i, activity in enumerate(true_time.columns):
#     x = true_time.loc[:, activity]
#     y = pred_time.loc[:, i]
#     stat, p_lev = levene(x, y)
#     shap_x = shapiro(x).pvalue
#     shap_y = shapiro(y).pvalue
#
#     print(activity)
#     print('Levenes test p = %f' % p_lev)
#     print('Shapiro test for predicted time: p = %f; for true time: p = %f' % (shap_x, shap_y))
#     print(pingouin.ttest(x, y, correction=True))
#     print('\n')

#
# means = pd.DataFrame([o.params['mu'] for o in posteriormodel.obs_distns])
# if means.shape[1]==1 :
#     means.columns = ['Magnitude']
# else:
#     means.columns = ['Magnitude', 'x-axis', 'y-axis', 'z-axis']
# new_idx =  means.iloc[:,0].sort_values(ascending=False).index.values
# lookup = tuple(zip(range(0,5), new_idx))
# print(lookup)
#
# for a in states:
#     a_copy = np.copy(a)
#     for row in lookup:
#         a[a_copy == row[0]] = row[1]
