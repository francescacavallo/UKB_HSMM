import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
#import _pickle as cPickle
import pingouin as pg
from scipy.stats import levene, shapiro
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score
from packages.helper_funct import boxplot_sorted, ilr_accel, giavarina_analysis
from packages.helper_funct import giavarina_plot, regression_analysis, sort_states
from packages.helper_funct import ilr_accel_mini, pearsonr_ci
import scipy.stats as stats
import statistics
import statsmodels.api as sm


n = 499
sorting_by = 'acceleration' # 'duration'
#features = ['Magnitude','Xangle','Yangle','Zangle']
# features = ['Magnitude', 'Xaxis','Yaxis','Zaxis']
features = ['Magnitude']
model_name = '499_model4_magnitude_3angles'#_3angles'

####################################################################################
# Import
####################################################################################
with open('Models/'+str(n)+'/'+model_name+'.pickle','rb') as file:
    model = pickle.load(file)

# Import data
with open('ukb_data/data_5s.pkl', 'rb') as f:
    data = pickle.load(f)
data = data[:n]

with open('ukb_data/labels_5s.pkl', 'rb') as f:
    labels = pickle.load(f)
labels=labels[:n]


####################################################################################
# Get "true" time spent in each activity
####################################################################################
tot_counts = []
for p in labels:
    unique, counts = np.unique(p, return_counts=True)
    tot_counts.append(counts)

true_time = pd.DataFrame(tot_counts, columns=['Moderate activity', 'Sedentary', 'Sleep', 'Light tasks', 'Walking'])
true_time = true_time.apply(lambda x: x*5/60/60/7)
true_time = true_time.reindex(['Moderate activity', 'Walking', 'Light tasks', 'Sedentary', 'Sleep'], axis=1)
true_time['PA']=true_time[['Moderate activity','Walking','Light tasks']].sum(axis=1)
breakpoint()
####################################################################################
# Get "pedicted" time spent in each activity
####################################################################################

# Import inferred stateseqs
states = model.stateseqs

tot_counts = []
for p in states:
    unique, counts = np.unique(p, return_counts=True)
    tot_counts.append(counts)
pred_time = pd.DataFrame(tot_counts)
pred_time = pred_time.apply(lambda x: x*5/60/60/7)

# Order states
pred_time = sort_states(model, pred_time, by=sorting_by)

pred_time['PA']=pred_time[['Moderate activity','Walking','Light tasks']].sum(axis=1)

####################################################################################
# Measure agreement
####################################################################################
pred_time_temp = pred_time.dropna()
true_time_temp = true_time[true_time.index.isin(pred_time_temp.index)]
n = true_time_temp.shape[0]

# Correlation
for activity in true_time.columns:
    ro = true_time_temp[activity].corr(pred_time_temp[activity])
    r, p, lo, hi = pearsonr_ci(true_time_temp[activity],pred_time_temp[activity],alpha=0.05)
    print(f'Correlation for {activity:s} = {ro:0.2f} (p = {p:0.2f}), [{lo:0.2f}, {hi:0.2f}]') #%(activity, ro, r, p, lo, hi))

# Correlation plot
fig, axs = plt.subplots(3,2, figsize=(12,10))

for activity, ax in zip(true_time.columns, axs.flat):
    ax.set(title= activity,
                xlabel='True', ylabel='Predicted')
    ax.scatter(true_time_temp[activity], pred_time_temp[activity],
                    c='k', s=20, marker='o')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x,  c='grey', ls='--', label='Line of Equality')

plt.tight_layout()
plt.show()

# Cohens Kappa
kappa = []
for i in range(0, len(labels)):
    kappa.append(cohen_kappa_score(labels[i], states[i]))
kappa_mean = statistics.mean(kappa)
kappa_std = statistics.stdev(kappa)
print('Cohens kappa mean (std) %f (%f)' %(kappa_mean, kappa_std))

# Bland-Altman Analysis
fig, axs = plt.subplots(3,2, figsize=(12,10))
summary = pd.DataFrame()

for activity, ax in zip(true_time.columns, axs.flat):

    df_giav = pd.concat([true_time_temp[activity], pred_time_temp[activity]], axis=1)
    df_giav.columns = [activity+'_true', activity+'_pred']
    df_giav = df_giav.apply(np.log) # log transform variables
    giav_results, results = giavarina_analysis(df_giav)
    summary = pd.concat([summary, results], axis=0)
    #giavarina_plot(giav_results, ax)
    sm.graphics.mean_diff_plot(df_giav.iloc[:,0], df_giav.iloc[:,1], ax = ax)
    ax.set_title(activity, fontsize=15, fontweight='bold')

summary.to_csv('blandaltman_summary_'+model_name+'_by'+sorting_by+'.csv')
plt.tight_layout()
plt.savefig('blandaltman_plot_'+model_name+'_by'+sorting_by+'.png')

####################################################################################
# Import UKB data
####################################################################################

df = pd.read_csv('G:/UKB_data/3_Data/5_Subsets/00_prediabetes_markers_assessmentCentre.csv', index_col='eid')
eid = pd.read_csv(os.path.join(os.getcwd(), 'ukb_data/eid.csv'), header=None)
df = df[df.index.isin(eid.iloc[:,0])]
df.columns = df.columns.str.lstrip()
#
pred_time.index = eid.values.reshape(1,eid.shape[0])[0]
true_time.index = eid.values.reshape(1,eid.shape[0])[0]

####################################################################################
# Predict BMI/WC from activity --> LINEAR
####################################################################################
# df_true = pd.concat([df, true_time], axis=1, join='inner').dropna()
# df_pred = pd.concat([df, pred_time], axis=1, join='inner').dropna()
#
# print('\n Linear - BMI')
# for activity in true_time.columns:
#     print(pg.linear_regression(df_true[activity], df_true[' BMI']))
#     print(pg.linear_regression(df_pred[activity], df_pred[' BMI']))
#
# print('\n Linear - WC')
# for activity in true_time.columns:
#     print(pg.linear_regression(df_true[activity], df_true[' Waist circumference']))
#     print(pg.linear_regression(df_pred[activity], df_pred[' Waist circumference']))


####################################################################################
# Predict BMI/WC from activity --> COMPOSITIONAL
####################################################################################
# true_comp = ilr_accel(true_time)
# pred_comp = ilr_accel(pred_time)
# df_true_comp = pd.concat([df, true_comp], axis=1, join='inner').dropna()
# df_pred_comp = pd.concat([df, pred_comp], axis=1, join='inner').dropna()
#
# ols_summary = pd.DataFrame(columns=['outcome', 'label', 'names', 'coef', 'se', 'T', 'pval',
#                                     'r2', 'adj_r2', 'CI[2.5%]', 'CI[97.5%]'])
#
# ols_summary = regression_analysis(true_time.columns, 'BMI',
#                                   df_true_comp, df_pred_comp, ols_summary)
# ols_summary = regression_analysis(true_time.columns, 'Waist circumference',
#                                   df_true_comp, df_pred_comp, ols_summary)
#
# ols_summary.to_csv('_ols_summary_'+model_name+'_by'+sorting_by+'.csv')


####################################################################################
# Predict BMI/WC from activity --> COMPOSITIONAL --> overall PA-SB-Sleep
####################################################################################
true_comp = ilr_accel_mini(true_time)
pred_comp = ilr_accel_mini(pred_time)
df_true_comp = pd.concat([df, true_comp], axis=1, join='inner').dropna()
df_pred_comp = pd.concat([df, pred_comp], axis=1, join='inner').dropna()


ols_summary = pd.DataFrame(columns=['outcome', 'label', 'names', 'coef', 'se', 'T', 'pval',
                                    'r2', 'adj_r2', 'CI[2.5%]', 'CI[97.5%]'])

ols_summary = regression_analysis(['Sedentary', 'PA', 'Sleep'], 'BMI',
                                  df_true_comp, df_pred_comp, ols_summary)
ols_summary = regression_analysis(['Sedentary', 'PA', 'Sleep'], 'Waist circumference',
                                  df_true_comp, df_pred_comp, ols_summary)

ols_summary.to_csv('_ols_summary_'+model_name+'_by'+sorting_by+'_compositePA.csv')
exit()
####################################################################################
# Boxplots
####################################################################################
plt.close('all')
plt.style.use('tableau-colorblind10')
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(1,2, figsize=(12,6))
boxplot_sorted(true_time, ax=ax.flatten()[0])
boxplot_sorted(pred_time, ax=ax.flatten()[1])
plt.suptitle('Total daily time spent in activities', fontsize=25)
ax[0].set_ylabel('Time (hours/day)', fontsize=20)
ax[0].set_title('True', fontsize=20)
ax[1].set_ylabel('Time (hours/day)', fontsize=20)
ax[1].set_title('Inferred', fontsize=20)
plt.tight_layout()

#plt.savefig(model_name+'_boxplot_by'+sorting_by+'_totalTime.png')



####################################################################################
# Legacy
####################################################################################


#     # Calculations
#     df_giav = pd.concat([true_time_temp[activity], pred_time_temp[activity]], axis=1)
#     means = df_giav.mean(axis=1)
#     diffs = df_giav.diff(axis=1).iloc[:, -1]
#     percent_diffs = diffs / means * 100
#     bias = np.mean(percent_diffs)
#     sd = np.std(diffs, ddof=1)
#     upper_loa = bias + 2 * sd
#     lower_loa = bias - 2 * sd
#
#     # Sample size
#     n = df_giav.shape[0]
#     # Variance
#     var = sd**2
#     # Standard error of the bias
#     se_bias = np.sqrt(var / n)
#     # Standard error of the limits of agreement
#     se_loas = np.sqrt(3 * var / n)
#     # Endpoints of the range that contains 95% of the Student’s t distribution
#     t_interval = stats.t.interval(alpha=0.95, df=n - 1)
#     # Confidence intervals
#     ci_bias = bias + np.array(t_interval) * se_bias
#     ci_upperloa = upper_loa + np.array(t_interval) * se_loas
#     ci_lowerloa = lower_loa + np.array(t_interval) * se_loas
#
#     # Plot
#     ax.set(
#         title='Giavarina Plot',
#         xlabel='Mean (hours/day)', ylabel='Percentage Difference (%)'
#     )
#
#     # Scatter plot
#     ax.scatter(means, percent_diffs, c='k', s=20, alpha=0.6, marker='o')
#     # Plot the zero line
#     ax.axhline(y=0, c='k', lw=0.5)
#     # Plot the bias and the limits of agreement
#     ax.axhline(y=upper_loa, c='grey', ls='--')
#     ax.axhline(y=bias, c='grey', ls='--')
#     ax.axhline(y=lower_loa, c='grey', ls='--')
#     # Get axis limits
#     left, right = ax.get_xlim()
#     bottom, top = ax.get_ylim()
#     # Increase the y-axis limits to create space for the confidence intervals
#     max_y = max(abs(ci_lowerloa[0]), abs(ci_upperloa[1]), abs(bottom), abs(top))
#     ax.set_ylim(-max_y * 1.1, max_y * 1.1)
#     # Set x-axis limits
#     domain = right - left
#     ax.set_xlim(left, left + domain * 1.1)
#     # Add the annotations
#     ax.annotate('+1.96×SD', (right, upper_loa), (0, 7), textcoords='offset pixels')
#     ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -25), textcoords='offset pixels')
#     ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels')
#     ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -25), textcoords='offset pixels')
#     ax.annotate('-1.96×SD', (right, lower_loa), (0, 7), textcoords='offset pixels')
#     ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -25), textcoords='offset pixels')
#     # Plot the confidence intervals
#     ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--')
#     ax.plot([left] * 2, list(ci_bias), c='grey', ls='--')
#     ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--')
#     # Plot the confidence intervals' caps
#     x_range = [left - domain * 0.025, left + domain * 0.025]
#     ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--')
#     ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--')
#     ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--')
#     ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--')
#     ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--')
#     ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--')
#
# plt.tight_layout()
# plt.show()






# Get acceleration
# tmp_data = np.empty((0,1))
# tmp_labels = np.empty((0,1))
# for i, array in enumerate(data):
#     print(i)
#     tmp_data = np.append(tmp_data, array[:,0])
#     tmp_labels = np.append(tmp_labels, labels[i])
# true_accel = pd.DataFrame(np.append(tmp_data.reshape(tmp_data.shape[0], 1),
#                             tmp_labels.reshape(tmp_labels.shape[0], 1), axis=1))
# breakpoint()
# tmp_data = np.empty((0,1))
# tmp_labels = np.empty((0,1))
# for i, array in enumerate(model.states_list):
#     print(i)
#     tmp_data = np.append(tmp_data, array.data[:,0])
#     tmp_labels = np.append(tmp_labels, labels[i])
# pred_accel = pd.DataFrame(np.append(tmp_data.reshape(tmp_data.shape[0], 1),
#                             tmp_labels.reshape(tmp_labels.shape[0], 1), axis=1))
#
# breakpoint()
# fig, ax = plt.subplots(1,2, figsize=(12,6))
# boxplot_sorted(true_accel, ax=ax.flatten()[0])
# boxplot_sorted(pred_accel, ax=ax.flatten()[1])
# plt.suptitle('Mean acceleration per activity', fontsize=25)
# ax[0].set_ylabel('Mean acceleration (mg)', fontsize=20)
# ax[0].set_title('True', fontsize=20)
# ax[1].set_ylabel('Mean acceleration (mg)', fontsize=20)
# ax[1].set_title('Inferred', fontsize=20)
# plt.show()



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
