import alloptical_utils_pj as ao
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import sys;
sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/Vape')
import utils.utils_funcs as uf #from Vape
from numba import njit

# pkl_path = '/Users/prajayshah/Documents/data-to-process/2020-03-18/2020-03-18_t-019_t-020.pkl'
# pkl_path = "/Volumes/Extreme SSD/oxford-data/2020-03-03/2020-03-03_spont-baseline.pkl"
# s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-03/suite2p/spont-baseline/plane0'
# experiment = 'RL066 _ spont-baseline'

pkl_path = "/Volumes/Extreme SSD/oxford-data/2020-03-03/2020-03-03_spont-4ap-distal.pkl"
s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-03/suite2p/spont-4ap-distal/plane0'
experiment = 'RL066: spont-4ap-distal'


#%%
with open(pkl_path, 'rb') as f:
    exp_obj = pickle.load(f)

# s2p_path_4ap = '/Volumes/Extreme SSD/oxford-data/2020-03-19/suite2p/spont-4ap/plane0'


# flu, spks, stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)
# flu_4ap, spks_4ap, stat = uf.s2p_loader(s2p_path_4ap, subtract_neuropil=True)
#
# data = [flu, flu_4ap]
# data = [spks, spks_4ap]
# color = ['grey', 'green']

#%% new experiment object
# exp_obj = ao.twopimaging(suite2p_path=s2p_path)

# add bad seizure frames
exp_obj.seizure_frames = [list(range(0, 5900)), list(range(2810 * 4, 3240 * 4)), list(range(3450 * 4, 4190 * 4)), list(range(4530 * 4, 4930 * 4)),
                          list(range(5020 * 4, 5930 * 4)), list(range(6800 * 4, 7570 * 4)), list(range(7830 * 4, 8460 * 4)),
                          list(range(8670 * 4, 35800))]
l = [item for i in exp_obj.seizure_frames for item in i]

a = list(range(35800))
exp_obj.good_frames = [i for i in a if i not in l]

# delete indices with seizure frames from spks
exp_obj.spks_ = exp_obj.spks[:, exp_obj.good_frames]
# exp_obj.spks_ = np.delete(exp_obj.spks, l, axis=1)

#%% SAVE exp_obj as PKL #######################################################################################
# Pickle the exp_object output to save it for analysis
with open(pkl_path, 'wb') as f:
        pickle.dump(exp_obj, f)
print("pkl saved to %s" % pkl_path)

#%% filter suite2p results ####################################################################################
def _good_cells(exp_obj, min_radius_pix, max_radius_pix):
    good_cells = []
    radius_ = []
    for i in range(len(exp_obj.cell_id)):
        print('Processing cell #:', i)
        raw = list(exp_obj.raw[i])
        raw_dff = ao.normalize_dff(raw)
        std = np.std(raw_dff, axis=0)

        x = []
        y = []
        for j in np.arange(len(raw_dff), step=4):
            avg = np.mean(raw_dff[j:j+4])
            if avg > np.mean(raw_dff)+2.5*std:
              x.append(j)
              y.append(avg)

        radius = exp_obj.radius[i]
        radius_.append(radius)

        if len(x) > 0 and radius > min_radius_pix and radius < max_radius_pix:
            good_cells.append(exp_obj.cell_id[i])
    print('# of good cells found: %s (out of %s ROIs)'% (len(good_cells), len(exp_obj.cell_id)))
    exp_obj.good_cells = good_cells

    to_plot = radius_
    n, bins, patches = plt.hist(to_plot, 200)
    plt.axvline(min_radius_pix)
    plt.axvline(max_radius_pix)
    plt.suptitle('radius distribution', y=0.95)
    plt.show()
_good_cells(exp_obj, min_radius_pix=2.5, max_radius_pix=5.5)

cell_ids = exp_obj.cell_id
raws = exp_obj.raw
radiuses = exp_obj.radius
@njit
def _good_cells(cell_ids, raws, radiuses, min_radius_pix, max_radius_pix):
    good_cells = []
    for i in range(len(cell_ids)):
        print(i, " out of ", len(cell_ids), " cells")
        raw_ = raws[i]
        raw_dff = ao.normalize_dff_jit(raw_)
        std_ = raw_dff.std()

        a = []
        y = []
        for j in np.arange(len(raw_dff), step=4):  # think about whether or not you need to change this number to be dependent on the imaging fps
            avg = np.mean(raw_dff[j:j+4])
            if avg > np.mean(raw_dff)+2.5*std_: # if the avg of 4 frames is greater than the threshold then save the result
                a.append(j)
                y.append(avg)

        radius = radiuses[i]

        if len(a) > 0 and radius > min_radius_pix and radius < max_radius_pix:
            good_cells.append(cell_ids[i])
    print('# of good cells found: ', len(good_cells), ' (out of ', len(cell_ids), ' ROIs)')
    return good_cells
# initial quick run to allow numba to compile the function - not sure if this is actually creating time savings
good_cells = _good_cells(cell_ids=cell_ids[:3], raws=raws, radiuses=radiuses,
                         min_radius_pix=2.5, max_radius_pix=8.5)
exp_obj.good_cells = _good_cells(cell_ids=cell_ids, raws=raws, radiuses=radiuses,
                                 min_radius_pix=2.5, max_radius_pix=8.5)

# more efficient moving average calculation below!!!!!!!!!!
# - calculate moving average and find any value above std. threshold
cell_ids = exp_obj.cell_id
raws = exp_obj.raw
radiuses = exp_obj.radius
@njit
def moving_average(a, n=4):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
@njit
def _good_cells(cell_ids, raws, radiuses, min_radius_pix, max_radius_pix):
    good_cells = []
    len_cell_ids = len(cell_ids)
    for i in range(len_cell_ids):
        # print(i, " out of ", len(cell_ids), " cells")
        raw_ = raws[i]
        raw_dff = ao.normalize_dff_jit(raw_)
        std_ = raw_dff.std()

        raw_dff_ = moving_average(raw_dff, n=4)

        thr = np.mean(raw_dff) + 2.5 * std_
        e = np.where(raw_dff_ > thr)
        # y = raw_dff_[e]

        radius = radiuses[i]

        if len(e) > 0 and radius > min_radius_pix and radius < max_radius_pix:
            good_cells.append(cell_ids[i])

        if i % 100 == 0:  # print the progress once every 100k iterations
            print(i, " out of ", len_cell_ids, " cells done")
    print('# of good cells found: ', len(good_cells), ' (out of ', len_cell_ids, ' ROIs)')
    return good_cells
exp_obj.good_cells = _good_cells(cell_ids=cell_ids, raws=raws, radiuses=radiuses,
                                 min_radius_pix=2.5, max_radius_pix=7)
exp_obj.fps = 30

#%% plot example suite2p traces of selected number of cells
def plot_flu_overlap_plots(dff_array, title='', x_label='Time (seconds)', save_path=None, alpha=1):
    # make rolling average for these plots
    w = 30
    dff_array = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array])

    len_ = len(dff_array)
    fig, ax = plt.subplots(figsize=(20, 6))
    for i in range(len_):
        ax.plot(dff_array[i] + i*200, linewidth=1, color='black', alpha=alpha)

    # ax.margins(0)
    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = int(round(item / exp_obj.fps))
    ax.set_xticklabels(labels)
    ax.set_title(title, horizontalalignment='center', verticalalignment='top', pad=20, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xlabel(x_label)

    if save_path:
        plt.savefig(save_path)
    plt.show()

rois = [95, 16]

exp_obj.dff = ao.normalize_dff(exp_obj.raw[:10, :], threshold=50)
plot_flu_overlap_plots(dff_array=exp_obj.dff[:10, 10000:22000], title=(experiment + ' - flu of %s cells' % len(exp_obj.dff)), alpha=0.5)
                       # save_path="/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/%s: 10 cells.svg" % experiment)

#%% (quick) plot suite2p traces ###############################################################################
plt.figure(figsize=(50, 3))
for i in range(200,250):
    plt.plot(exp_obj.raw[i][exp_obj.good_frames], linewidth=0.2, alpha=0.2)
    plt.plot(exp_obj.spks[i][exp_obj.good_frames], linewidth=0.2, alpha=0.2)
# plt.xlim(0, exp_obj.spks.shape[1])
plt.xlim(0, len(exp_obj.good_frames))
plt.show()

#%% Gaussian filter of spks data and then plotting
def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def frames2sigma(frames):
    return np.sqrt(frames / 2)

from scipy.ndimage import gaussian_filter
exp_obj.spks_smooth_ = np.asarray([gaussian_filter(a, sigma=frames2sigma(frames=15)) for a in exp_obj.spks_])  # TODO this is Matthias's suggested metric for calculating sigma, need to confirm

def plot_spks(cells):
    plt.figure(figsize=(30, 10))
    plt.margins(0)
    #plt.plot(a, color="black")
    for i in cells:
        plt.plot(exp_obj.spks_smooth_[i] + 100 * i, linewidth=1)
        plt.plot(exp_obj.spks[i][exp_obj.good_frames] + 100 * i, linewidth=1, alpha=0.2, color='black')
    plt.suptitle('%s - gaussian smoothing of deconvolved neural activity' % experiment, y=1.0)
    plt.show()
plot_spks(cells=range(100, 105))

# cut out the last frame(s) to make it easier to factor and reshape for binning below
exp_obj.spks_smooth_ = exp_obj.spks_smooth_[:, :-2]
# exp_obj.spks_ = exp_obj.spks_[:, :-2]

# binning of spks data
def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

# rebin
exp_obj.spks_smooth_binned = rebin(exp_obj.spks_smooth_, (exp_obj.spks_smooth_.shape[0], int(exp_obj.spks_smooth_.shape[1]/4)))
exp_obj.spks_binned = rebin(exp_obj.spks_, (exp_obj.spks_.shape[0], int(exp_obj.spks_.shape[1]/4)))


#%% plot dFF fluorescence traces
flu_ = exp_obj.raw[:50]
data_dff = []
for i in flu_:
    dff = ao.normalize_dff(i)
    data_dff.append(dff)

plt.figure(figsize=(50,3))
for i in range(len(dff)):
    plt.plot(dff[i], linewidth=0.05)
plt.xlim(0, len(exp_obj.raw[0]))
plt.show()


#%% ### calculate total neuronal activity by integrating neuronal activity and normalizing by total length of time
from sklearn.metrics import auc
area = [auc(np.arange(len(exp_obj.spks_smooth_binned[i])), exp_obj.spks_smooth_binned[i])
        for i in range(len(exp_obj.spks_smooth_binned))]

imaging_len_secs = exp_obj.spks_smooth_binned.shape[1] / exp_obj.fps

total_neural_activity = np.array(area) / imaging_len_secs

plt.hist(total_neural_activity, bins=100, density=True, color='grey', edgecolor='none')
plt.axvline(np.mean(total_neural_activity), color='black')
plt.suptitle('%s - histogram of neural activity level' % experiment, y=1.0)
plt.show()

# plot the cumulative function
values, base = np.histogram(total_neural_activity, bins=300)
cumulative = np.cumsum(values)
plt.plot(base[:-1], np.cumsum(values) / np.cumsum(values)[-1], c='blue')
plt.ylabel("Proportion")
plt.suptitle('%s - cum. summation of neural activity level' % experiment, y=1.0)
plt.show()


#%% ### correlation analysis ###########################################################################
# calculate and plotting corr matrix heatmap with pandas
def plot_corr(dataframe=None, corr_mtx=None, size=10, title=None, save_path=None):
    '''Plot a graphical correlation matrix for a dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    # Compute the correlation matrix for the received dataframe
    if corr_mtx is None:
        corr = dataframe.corr()
    elif corr_mtx is not None:
        corr = corr_mtx

    # Plot the correlation matrix
    plt.figure(figsize=(size, size))
    cax = plt.imshow(corr, cmap='bwr')
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    # plt.yticks(range(len(corr.columns)), corr.columns)

    plt.xticks(None)
    plt.yticks(None)

    # Add the colorbar legend
    plt.colorbar(cax, ticks=[-0.40, -0.20, 0, 0.20, 0.40], aspect=40, shrink=.8)
    plt.clim(-0.40, 0.40)
    plt.suptitle(title, y=1.0)
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return corr

df = pd.DataFrame(exp_obj.spks_smooth_binned.T)
exp_obj.corr = plot_corr(dataframe=df, size=18)  # when corr_matrix hasn't been generated yet


# with added heirarchical clustering
import scipy.cluster.hierarchy as sch
X = exp_obj.corr.values
d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
df_ordered = df.reindex_axis(columns, axis=1)

exp_obj.corr_ordered = plot_corr(dataframe=df_ordered, size=10)  # when corr_matrix hasn't been generated yet

# when just need to plot corr_matrix
exp_obj.corr = plot_corr(corr_mtx=exp_obj.corr, size=10)
exp_obj.corr_ordered = plot_corr(corr_mtx=exp_obj.corr_ordered, size=4, title='%s: correlation - ordered' % experiment,
                                 save_path="/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/%s - correlation - ordered.svg" % experiment)

#%% heatmap of gaussian smoothed spks, using df_ that is hierarchical clustered

import seaborn as sns
plt.figure(figsize=(12, 4))
sns.heatmap(df_.T, cmap='binary', xticklabels=False, yticklabels=False, vmax=20)
plt.show()


#%% histogram of cell-to-cell correlation values ###### TODO add time shuffling control to the quantification
# calculate and plot histogram of correlation coefficients
corr_mtx = np.corrcoef(exp_obj.spks_smooth_binned)
corr_values = corr_mtx[np.triu_indices(corr_mtx.shape[0], k=1)]
plt.hist(corr_values, bins=5000, density=True, color='grey', edgecolor='green')
plt.axvline(np.mean(corr_values), color='black')
plt.show()

import seaborn as sns
sns.distplot(corr_values, kde=True, bins=5000, hist=True, hist_kws={'edgecolor': 'grey'}, color='black')
plt.axvline(np.mean(corr_values))
plt.show()

# depracated below
for i in exp_obj.spks_smooth:
    b = np.corrcoef(i)
    a = np.triu(b)
    np.fill_diagonal(a,0)
    c = list(a.flat)
    d = [i for i in c if i != 0]

    # plot histogram of correlation coefficient densities
    n, bins, patches = plt.hist(d, 400, density=True)
    plt.axvline(np.mean(d))
plt.show()

#%% ### PCA on dataset ############################################################################################
# TODO make sure you are actually calculating PCA on the correct variables.
pca = PCA(n_components=exp_obj.spks_smooth.shape[0])
pca.fit(exp_obj.spks_smooth)
pca_result = pd.DataFrame(pca.transform(exp_obj.spks_smooth)) #, columns=['PCA%i' % i for i in range(275)])
sv = pca.singular_values_
su = (sv / sum(sv))

#%%
# plot variance per PC
plt.plot(pca.explained_variance_ratio_[:10**2])
# TODO fit power law exponent to the data
x = np.linspace(1,10**2,10**2)
y = 1/(x**1.1)
plt.plot(x, y)
plt.yscale('log')
plt.xscale('log')
plt.show()

