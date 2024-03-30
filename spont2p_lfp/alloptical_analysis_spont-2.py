import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import sys;
sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/Vape')

# save_path = '/Users/prajayshah/Documents/data-to-process/2020-03-18/2020-03-18_t-019_t-020.pkl'
pkl_path_1 = "/Volumes/Extreme SSD/oxford-data/2020-03-03/2020-03-03_spont-baseline.pkl"
pkl_path_2 = "/Volumes/Extreme SSD/oxford-data/2020-03-03/2020-03-03_spont-4ap-distal.pkl"

experiment = 'RL066 _ spont-imaging _ pre and post-4ap'

#%% IMPORT PKL
with open(pkl_path_1, 'rb') as f:
    exp_obj_pre = pickle.load(f)

with open(pkl_path_2, 'rb') as f:
    exp_obj_post = pickle.load(f)

#%% calculate total neuronal activity by integrating DECONVOLVED neuronal activity and normalizing by total length of time
from sklearn.metrics import auc
# pre 4ap total neural activity
area_pre = [auc(np.arange(len(exp_obj_pre.spks_smooth_binned[i])), exp_obj_pre.spks_smooth_binned[i])
        for i in range(len(exp_obj_pre.spks_smooth_binned))]

imaging_len_secs_pre = exp_obj_pre.spks_smooth_binned.shape[1] / exp_obj_pre.fps
total_neural_activity_pre = np.array(area_pre) / imaging_len_secs_pre

# post 4ap total neural activity
area_post = [auc(np.arange(len(exp_obj_post.spks_smooth_binned[i])), exp_obj_post.spks_smooth_binned[i])
        for i in range(len(exp_obj_post.spks_smooth_binned))]

imaging_len_secs_post = exp_obj_post.spks_smooth_binned.shape[1] / exp_obj_post.fps
total_neural_activity_post = np.array(area_post) / imaging_len_secs_post

#%% calculate total neuronal activity by integrating DFF neuronal activity and normalizing by total length of time
from sklearn.metrics import auc
# pre 4ap total neural activity
area_pre = [auc(np.arange(len(exp_obj_pre.dff[i])), exp_obj_pre.dff[i])
        for i in range(len(exp_obj_pre.dff))]

imaging_len_secs_pre = exp_obj_pre.dff.shape[1] / exp_obj_pre.fps
total_neural_activity_pre = np.array(area_pre) / imaging_len_secs_pre

# post 4ap total neural activity
area_post = [auc(np.arange(len(exp_obj_post.dff[i])), exp_obj_post.dff[i])
        for i in range(len(exp_obj_post.dff))]

imaging_len_secs_post = exp_obj_post.dff.shape[1] / exp_obj_post.fps
total_neural_activity_post = np.array(area_post) / imaging_len_secs_post


#%% plot histogram and cum. summation graph of total neuronal activity
plt.hist(total_neural_activity_pre, bins=100, density=True, color='grey', edgecolor='none', alpha=0.3)
plt.hist(total_neural_activity_post, bins=100, density=True, color='green', edgecolor='none')
plt.suptitle('%s - histogram of dFF neural activity level' % experiment, y=1.0)
plt.show()

#%% # plot the cumulative function
plt.figure(figsize=(4,4.5))
for i in (total_neural_activity_pre, total_neural_activity_post):
    values, base = np.histogram(i, bins=1000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], np.cumsum(values) / np.cumsum(values)[-1])
plt.axhline(y=1.0, linestyle='--', color='black')
# plt.xlim((0,150))
plt.xscale('log')
plt.ylabel("Proportion")
plt.xlabel("neural activity (a.u.)")
plt.suptitle('%s - cum. summation of neural activity level' % experiment, y=1.0, fontsize=10)
plt.savefig("/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/%s pre- and post-4ap activity cells.svg" % experiment)
plt.show()

#%% ### plot PCA
# TODO make sure you are actually calculating PCA on the correct variables.
pca_pre = PCA(n_components=exp_obj_post.spks_smooth.shape[0])
pca_pre.fit(exp_obj_pre.spks_smooth)
pca_result_pre = pd.DataFrame(pca_pre.transform(exp_obj_pre.spks_smooth)) #, columns=['PCA%i' % i for i in range(275)])


pca_post = PCA(n_components=exp_obj_post.spks_smooth.shape[0])
pca_post.fit(exp_obj_post.spks_smooth)
pca_result_post = pd.DataFrame(pca_post.transform(exp_obj_post.spks_smooth)) #, columns=['PCA%i' % i for i in range(275)])

#%% ### plot variance per PC
plt.figure(figsize=(4.5, 4))
plt.plot(pca_pre.explained_variance_ratio_[:10**2], c='black')
plt.plot(pca_post.explained_variance_ratio_[:10**2], c='red')
# TODO fit power law exponent to the data
x = np.linspace(1,10**2,10**2)
y = 1/(x**1.0); print(y)
plt.plot(x, y, linestyle='--', c='gray')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('variance')
plt.xlabel('PC components')
plt.suptitle('%s - PCA' % experiment, y=1.0, fontsize=10)
plt.savefig("/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/%s PCA.svg" % experiment)
plt.show()

