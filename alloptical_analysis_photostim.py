# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys

sys.path.append('/home/pshah/Documents/code/')
# sys.path.append('/home/pshah/Documents/code/Vape/utils/')
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
from utils import funcs_pj as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw
import tifffile as tf

###### IMPORT pkl file containing data in form of expobj
trial = 't-011'
experiment = 'RL108: photostim-post4ap-%s' % trial
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)


with open(pkl_path, 'rb') as f:
    print('importing expobj for "%s %s" from: %s' % (date, experiment, pkl_path))
    expobj = pickle.load(f)
    print('DONE IMPORT.')

if hasattr(expobj, 'paq_rate'):
    pass
else:
    print('need to run paqProcessing to update paq attr.s in expobj')
    expobj.paqProcessing(); expobj.save_pkl()

# %% ANALYSIS STEPS FOR SEIZURE TRIALS ONLY!!

expobj.avg_sub_l, im_sub_l, im_diff_l = expobj.avg_seizure_images(
    baseline_tiff="/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_t-005/2020-12-18_t-005_Cycle00001_Ch3.tif",
    frames_last=1000)

# counter = 0
# for i in avg_sub_l:
#     plt.imshow(i); plt.suptitle('%s' % counter); plt.show()
#     counter += 1

expobj.avg_stim_images(stim_timings=expobj.stim_start_frames, peri_frames=50, to_plot=False, save_img=True)
expobj.save_pkl()

for i in range(len(expobj.avg_sub_l)):
    img = pj.rotate_img_avg(expobj.avg_sub_l[i], angle=90)
    # PCA decomposition of the avg_seizure images
    img_compressed = pj.pca_decomp_image(img, components=1, plot_quant=True)


# MAKE SUBSELECTED TIFFS OF INVIDUAL SEIZURES BASED ON THEIR START AND STOP FRAMES
expobj._subselect_sz_tiffs(onsets=expobj.stims_bf_sz, offsets=expobj.stims_af_sz)


# %% classifying stims as in_sz or out_sz or before_sz or after_sz

expobj.stims_in_sz = [stim for stim in expobj.stim_start_frames if stim in expobj.seizure_frames]
expobj.stims_out_sz = [stim for stim in expobj.stim_start_frames if stim not in expobj.seizure_frames]
expobj.stims_bf_sz = [stim for stim in expobj.stim_start_frames
                      for sz_start in expobj.seizure_lfp_onsets
                      if 0 < (sz_start - stim) < 5 * expobj.fps]  # select stims that occur within 5 seconds before of the sz onset
expobj.stims_af_sz = [stim for stim in expobj.stim_start_frames
                      for sz_start in expobj.seizure_lfp_offsets
                      if 0 < -1 * (sz_start - stim) < 5 * expobj.fps]  # select stims that occur within 5 seconds afterof the sz offset
print('\n|- stims_in_sz:', expobj.stims_in_sz, '\n|- stims_out_sz:', expobj.stims_out_sz,
      '\n|- stims_bf_sz:', expobj.stims_bf_sz, '\n|- stims_af_sz:', expobj.stims_af_sz)
aoplot.plot_lfp_stims(expobj)
expobj.save_pkl()

# %% classifying cells as in or out of the current seizure location in the FOV

# draw boundary on the image in ImageJ and save results as CSV


# import the CSV file in and classify cells by their location in or out of seizure

# moved this to utils.funcs_pj
def plot_cell_loc(expobj, cells: list, color: str = 'pink', show: bool = True):
    """
    plots an image of the FOV to show the locations of cells given in cells list.
    :param expobj: alloptical or 2p imaging object
    :param color: str to specify color of the scatter plot for cells
    :param cells: list of cells to plot
    :param show: if True, show the plot at the end of the function
    """
    black = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')
    plt.imshow(black)

    for cell in cells:
        y, x = expobj.stat[cell]['med']
        plt.scatter(x=x, y=y, edgecolors=color, facecolors='none', linewidths=0.8)

    if show:
        plt.show()
# csv_path = "/home/pshah/mnt/qnap/Analysis/2020-12-18/2020-12-18_t-013/2020-12-18_t-013_post_border.csv"

# need to run this twice to correct for mis-assignment of cells (look at results and then find out which stims need to be flipped)
flip_stims = [1424, 1572, 1720,
              3944, 4092, 4240, 4388, 4537,
              7650, 7798, 7946, 8094, 8242, 8391,
              11059, 11207, 11355, 11504, 11652, 11800, 11948]  # specify here the stims where the flip=False leads to incorrect assignment

print('working on classifying cells for stims start frames:')
for on, off in zip(expobj.stims_bf_sz, expobj.stims_af_sz):
    stims_of_interest = [stim for stim in expobj.stim_start_frames if on <= stim <= off]
    print('|-', stims_of_interest)

    expobj.cells_sz_stim = {}
    for stim in stims_of_interest:
        sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path, trial, stim)
        if stim in flip_stims:
            flip = True
        else:
            flip = False

        in_sz = expobj.classify_cells_sz(sz_border_path, to_plot=True, title='%s' % stim, flip=flip)
        expobj.cells_sz_stim[stim] = in_sz  # for each stim, there will be a list of cells that will be classified as in seizure or out of seizure


# %%


# %% photostim analysis - PLOT avg over all photstim. trials traces from PHOTOSTIM TARGETTED cells

# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
x = np.asarray([i for i in expobj.targets_dfstdF_avg])
# y_label = 'pct. dFF (normalized to prestim period)'
y_label = 'dFstdF (normalized to prestim period)'

aoplot.plot_photostim_avg(dff_array=x, expobj=expobj, stim_duration=expobj.duration_frames, pre_stim=expobj.pre_stim,
                          post_stim=expobj.post_stim,
                          title=(experiment + '- responses of all photostim targets'),
                          y_label=y_label, x_label='Time post-stimulation (seconds)')

# %% plot entire trace of individual targeted cells as super clean subplots, with the same y-axis lims

expobj.raw_targets = [expobj.raw[expobj.cell_id.index(i)] for i in expobj.good_photostim_cells_all]
expobj.dff_targets = aoutils.normalize_dff(np.array(expobj.raw_targets))
expobj.targets_dff_base = aoutils.normalize_dff_baseline(
    arr=expobj.raw_df.loc[[str(x) for x in expobj.s2p_cell_targets]],
    baseline_array=expobj.baseline_raw_df)
# plot_photostim_subplots(dff_array=dff_targets,
#                 title=(experiment + '%s responses of responsive cells' % len(expobj.good_photostim_cells_stim_responses_dFF)))
to_plot = expobj.dff_targets
# to_plot = expobj.targets_dff_base.to_numpy()
aoplot.plot_photostim_overlap_plots(dff_array=to_plot, expobj=expobj,
                                    title=(experiment + '-'))

aoplot.plot_photostim_subplots(dff_array=to_plot, expobj=expobj, x_label='Frames',
                               y_label='Raw Flu',
                               title=(experiment + '-'))

# # plot the photostim targeted cells as a heatmap
# dff_array = expobj.dff_targets[:, :]
# w = 10
# dff_array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array]
# dff_array = np.asarray(dff_array)
#
# plt.figure(figsize=(5, 10));
# sns.heatmap(dff_array, cmap='RdBu_r', vmin=0, vmax=500);
# plt.show()


# %%
# measure, for each cell, the pct of trials in which the dFF > 20% post stim (normalized to pre-stim avgF for the trial and cell)
# can plot this as a bar plot for now showing the distribution of the reliability measurement

expobj.reliability = aoutils.calculate_reliability(expobj=expobj, dfstdf_threshold=0.3)
pj.bar_with_points(data=[list(expobj.reliability.values())], x_tick_labels=['post-4ap'], ylims=[0, 100], bar=False,
                   title='reliability of stim responses', expand_size_x=2)

# %%
pre_4ap_reliability = list(expobj.reliability.values())
post_4ap_reliabilty = list(expobj.reliability.values())  # reimport another expobj for post4ap trial

pj.bar_with_points(data=[pre_4ap_reliability, post_4ap_reliabilty], x_tick_labels=['pre-4ap', 'post-4ap'],
                   ylims=[0, 100], bar=False, title='reliability of stim responses', expand_size=1.2)

# %% plot response of ALL cells across whole trace in FOV after photostim - plotting not yet working properly
# collect photostim timed average dff traces
title = 'All cells dFF'
all_cells_dff = []
good_std_cells = []

# %% calculate and plot average response of cells in response to all stims as a bar graph


# there's a bunch of very high dFF responses of cells
high_responders = expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500].index.values
# expobj.dff_all_cells.iloc[high_responders[0], 1:]
# list(expobj.dff_all_cells.iloc[high_responders[0], 1:])
# idx = expobj.cell_id.index(1668);
# aoplot.plot_flu_trace(expobj=expobj, idx=idx, to_plot='dff', size_factor=2)

# TODO need to troubleshoot how these scripts are calculating the post stim responses for the non-targets because some of them seem ridiculously off

# remove cells with very high average response values from the dff dataframe

## using pj.bar_with_points() for a nice bar graph
group1 = list(expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'photostim target'][
                  'Avg. dF/stdF response'])
group2 = list(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'non-target']['Avg. dF/stdF response'])
pj.bar_with_points(data=[group1, group2], x_tick_labels=['photostim target', 'non-target'], xlims=[0, 0.6],
                   ylims=[0, 1.5], bar=False,
                   colors=['red', 'black'], title=experiment, y_label='Avg dF/stdF response', expand_size_y=1.3,
                   expand_size_x=1.4)

# %% plot heatmap of average of all stims. per cell for each stim. group
# - need to find a way to sort these responses that similar cells are sorted together

stim_timings = [str(i) for i in expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing
average_responses = expobj.dfstdf_all_cells[stim_timings].mean(axis=1).tolist()

# make heatmap of responses across all cells across all stims
df_ = expobj.dfstdf_all_cells[stim_timings]  # select appropriate stim time reponses from the pandas df
df_ = df_[df_.columns].astype(float)

plt.figure(figsize=(5, 15));
sns.heatmap(df_, cmap='seismic', vmin=-5, vmax=5, cbar_kws={"shrink": 0.25});
plt.show()

# %% plot imshow XY locations with average response of ALL cells in FOV

# TODO transfer this FOV cell location mapped response plot to the aoplot script

from matplotlib.colors import LinearSegmentedColormap, ColorConverter

# make a matrix containing pixel locations and responses at each of those pixels
responses = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')
for n in expobj.good_cells:
    idx = expobj.cell_id.index(n)
    ypix = expobj.stat[idx]['ypix']
    xpix = expobj.stat[idx]['xpix']
    responses[ypix, xpix] = 100. + 1 * round(average_responses[expobj.good_cells.index(n)], 2)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)


c = ColorConverter().to_rgb
# rvb = make_colormap([c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
cmap_1 = make_colormap([c('black'), 0.01, c('blue'), c('white'), 0.495, c('white'), 0.495, c('white'), c('red')])

# mask some 'bad' data, in your case you would have: data < 0.05
responses = np.ma.masked_where(responses < 0.05, responses)
cmap = plt.cm.bwr
cmap.set_bad(color='black')

plt.figure(figsize=(7, 7))
im = plt.imshow(responses, cmap=cmap)
cb = plt.colorbar(im, fraction=0.046, pad=0.04)
cb.set_label('dF/stdF')
plt.clim(98, 102)
for (x, y) in expobj.target_coords_all:
    plt.scatter(x=x, y=y, edgecolors='green', facecolors='none', linewidths=0.8)
plt.suptitle((experiment + ' - avg. stim responses - targets in green'), y=0.95, fontsize=10)
plt.show()
# plt.savefig(
#     "/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/avg_pop_responses_%s.svg" % experiment)






#########################################################################################################################
#### END OF CODE THAT HAS BEEN REVIEWED SO FAR ##########################################################################
#########################################################################################################################

#%%












# %% define cells in proximity of the targeted cell and plot the flu of those pre and post-4ap
# - maybe make like a heatmap around the cell that is being photostimed
# Action plan:
# - make a dictionary for every cell that was targeted (and found in suite2p) that contains:
#   - coordinates of the cell
#   - trials that were successful in raising the fluorescence at least 30% over pre-stim period
#   - other cells that are in 300um proximity of the targeted cell

# same as calculating repsonses and assigning to pixel areas, but by coordinates now
group = 0
responses_group_1_ = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')
for n in filter(lambda n: n not in expobj.good_photostim_cells_all, expobj.good_cells):
    idx = expobj.cell_id.index(n)
    ypix = int(expobj.stat[idx]['med'][0])
    xpix = int(expobj.stat[idx]['med'][1])
    responses_group_1_[ypix, xpix] = 100 + 1 * round(average_responses[group][expobj.good_cells.index(n)], 2)

pixels_200 = round(200. / expobj.pix_sz_x)
pixels_20 = round(20. / expobj.pix_sz_x)

prox_responses = np.zeros((pixels_200 * 2, pixels_200 * 2), dtype='uint16')
for cell in expobj.good_photostim_cells_all:
    # cell = expobj.good_photostim_cells_all[0]
    # define annulus around the targeted cell
    y = int(expobj.stat[expobj.cell_id.index(cell)]['med'][0])
    x = int(expobj.stat[expobj.cell_id.index(cell)]['med'][1])

    arr = np.zeros((expobj.frame_x, expobj.frame_x))
    rr, cc = draw.circle(y, x, radius=pixels_200, shape=arr.shape)
    arr[rr, cc] = 1
    rr, cc = draw.circle(y, x, radius=pixels_20, shape=arr.shape)
    arr[rr, cc] = 0
    # plt.imshow(arr); plt.show() # check shape of the annulus

    # find all cells that are not photostim targeted cells, and are in proximity to the cell of interest
    for cell2 in filter(lambda cell2: cell2 not in expobj.good_photostim_cells_all, expobj.good_cells):
        y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
        x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
        if arr[y_loc, x_loc] == 1.0:
            loc_ = [pixels_200 + y_loc - y, pixels_200 + x_loc - x]
            prox_responses[loc_[0] - 2:loc_[0] + 2, loc_[1] - 2:loc_[1] + 2] = responses_group_1_[y_loc, x_loc]
            # prox_responses[loc_[0], loc_[1]] = responses_group_1_[y_loc, x_loc]
        prox_responses[pixels_200 - pixels_20:pixels_200 + pixels_20,
        pixels_200 - pixels_20:pixels_200 + pixels_20] = 500  # add in the 20um box around the cell of interest

prox_responses = np.ma.masked_where(prox_responses < 0.05, prox_responses)
cmap = plt.cm.bwr
cmap.set_bad(color='black')

plt.imshow(prox_responses, cmap=cmap)
cb = plt.colorbar()
cb.set_label('dF/preF')
plt.clim(80, 120)
plt.suptitle((experiment + '- avg. stim responses - Group %s' % group), y=1.00)
plt.show()

# %%
# plot response over distance from photostim. target cell to non-target cell in proximity
import math

d = {}
d['cell_pairs'] = []
d['distance'] = []
d['response_of_target'] = []
d['response_of_non_target'] = []
for cell in expobj.good_photostim_cells[0]:
    y = int(expobj.stat[expobj.cell_id.index(cell)]['med'][0])
    x = int(expobj.stat[expobj.cell_id.index(cell)]['med'][1])

    arr = np.zeros((expobj.frame_x, expobj.frame_x))
    rr, cc = draw.circle(y, x, radius=pixels_200, shape=arr.shape)
    arr[rr, cc] = 1
    rr, cc = draw.circle(y, x, radius=pixels_20, shape=arr.shape)
    arr[rr, cc] = 0  # delete selecting from the 20um around the targeted cell

    for cell2 in filter(lambda cell2: cell2 not in expobj.good_photostim_cells_all, expobj.good_cells):
        y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
        x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
        if arr[y_loc, x_loc] == 1.0:
            d['cell_pairs'].append('%s_%s' % (cell, cell2))
            d['distance'].append(math.hypot(y_loc - y, x_loc - x) * expobj.pix_sz_x)
            d['response_of_target'].append(average_responses[0][expobj.good_cells.index(cell)])
            d['response_of_non_target'].append(average_responses[0][expobj.good_cells.index(cell2)])

df_dist_resp = pd.DataFrame(d)

# plot distance vs. photostimulation response
plt.figure()
plt.scatter(x=df_dist_resp['distance'], y=df_dist_resp['response_of_non_target'])
plt.show()

# %%
# TODO calculate probability of stimulation in 10x10um micron bins around targeted cell

all_x = []
all_y = []
for cell2 in expobj.good_cells:
    y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
    x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
    all_x.append(x_loc)
    all_y.append(y_loc)


def binned_amplitudes_2d(all_x, all_y, responses_of_cells, response_metric='dF/preF', bins=35, title=experiment):
    """
    :param all_x: list of x coords of cells in dataset
    :param all_y: list of y coords of cells in dataset
    :param responses_of_cells: list of responses of cells to plots
    :param bins: integer - number of bins to split FOV in (along one axis)
    :return: plot of binned 2d histograms
    """

    all_amps_real = responses_of_cells  # list of photostim. responses
    denominator, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins)
    numerator, _, _ = np.histogram2d(all_x, all_y, bins=bins, weights=all_amps_real)
    h = numerator / denominator  # divide the overall
    Y, X = np.meshgrid(xedges, yedges)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    mesh1 = ax[0].pcolormesh(X, Y, h, cmap='RdBu_r', vmin=-20.0, vmax=20.0)
    ax[0].set_xlabel('Cortical distance (um)')
    ax[0].set_ylabel('Cortical distance (um)')
    ax[0].set_aspect('equal')

    range_ = max(all_x) - min(all_x)  # range of pixel values
    labels = [item for item in ax[0].get_xticks()]
    length = expobj.pix_sz_x * range_
    for item in labels:
        labels[labels.index(item)] = int(round(item / range_ * length))
    ax[0].set_yticklabels(labels)
    ax[0].set_xticklabels(labels)

    # ax[0].set_ylim([expobj.pix_sz_x*expobj.frame_x, 0])
    ax[0].set_title('Binned responses (%s um bins)' % round(length / bins))
    cb = plt.colorbar(mesh1, ax=ax[0])
    cb.set_label(response_metric)

    mesh2 = ax[1].pcolormesh(X, Y, denominator.astype(int), cmap='inferno', vmin=0, vmax=np.amax(denominator))
    ax[1].set_xlabel('Cortical distance (um)')
    ax[1].set_ylabel('Cortical distance (um)')
    ax[1].set_aspect('equal')
    labels = [item for item in ax[1].get_xticks()]
    for item in labels:
        length = expobj.pix_sz_x * range_
        labels[labels.index(item)] = int(round(item / range_ * length))
    ax[1].set_yticklabels(labels)
    ax[1].set_xticklabels(labels)

    # ax[1].set_ylim([expobj.pix_sz_x*expobj.frame_x, 0])
    ax[1].set_title('Number of cells in bin; %s total cells' % len(responses_of_cells))
    cb = plt.colorbar(mesh2, ax=ax[1])
    cb.set_label('num cells')

    plt.suptitle(title, horizontalalignment='center', verticalalignment='top', y=1.0)
    plt.show()


binned_amplitudes_2d(all_x, all_y, responses_of_cells=average_responses[0],
                     title='%s - slm group1 - whole FOV' % experiment)  # 2d spatial binned responses of all cells in average_responses argument
binned_amplitudes_2d(all_x, all_y, responses_of_cells=average_responses[1],
                     title='%s - slm group2 - whole FOV' % experiment)  # 2d spatial binned responses of all cells in average_responses argument

group = 1
e = {}
e['cell_pairs'] = []
e['distance'] = []
e['response_of_target'] = []
e['response_of_non_target'] = []
e['norm_location - x'] = []
e['norm_location - y'] = []
for cell in expobj.good_photostim_cells[0]:
    # cell = expobj.good_photostim_cells[0][0]
    y = int(expobj.stat[expobj.cell_id.index(cell)]['med'][0])
    x = int(expobj.stat[expobj.cell_id.index(cell)]['med'][1])

    # make a square array around the cell of interest
    arr = np.zeros((expobj.frame_x, expobj.frame_x))
    coords = draw.rectangle(start=(y - pixels_200, x - pixels_200), extent=pixels_200 * 2)
    # coords = draw.rectangle(start=(0,100), extent=pixels_200)
    arr[coords] = 1
    coords = draw.rectangle(start=(y - pixels_20, x - pixels_20), extent=pixels_20 * 2)
    arr[coords] = 0
    # plt.imshow(arr); plt.show() # show the created array if needed

    for cell2 in filter(lambda cell2: cell2 not in expobj.good_photostim_cells_all, expobj.good_cells):
        y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
        x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
        if arr[y_loc, x_loc] == 1.0:
            e['norm_location - y'].append(round(pixels_200 + y_loc - y))
            e['norm_location - x'].append(round(pixels_200 + x_loc - x))
            e['cell_pairs'].append('%s_%s' % (cell, cell2))
            e['distance'].append(math.hypot(y_loc - y, x_loc - x) * expobj.pix_sz_x)
            e['response_of_target'].append(average_responses[group][expobj.good_cells.index(cell)])
            e['response_of_non_target'].append(average_responses[group][expobj.good_cells.index(
                cell2)])  # note that SLM group #1 has been hardcorded in! # #

df_dist_resp_rec = pd.DataFrame(e)

binned_amplitudes_2d(all_x=list(df_dist_resp_rec['norm_location - x']),
                     all_y=list(df_dist_resp_rec['norm_location - y']),
                     responses_of_cells=list(df_dist_resp_rec['response_of_non_target']), bins=20,
                     response_metric='dF/preF',
                     title=(
                             experiment + ' - slm group %s - targeted cell proximity' % group))  # 2d spatial binned repsonses of all cells in average_responses argument

# %%

# next multiply the annulus array with a matrix of cell coords (with responses) responses_group_1


# TODO photostimulation of targeted cells before CSD, just after CSD, and a while after CSD


# TODO photostimulation of targeted cells before seizure, just after seizure, and a while after seizure


# %%

cells_dff_exc = []
cells_dff_inh = []
for cell in expobj.good_cells:
    if cell in expobj.cell_id:
        cell_idx = expobj.cell_id.index(cell)
        flu = []
        for stim in stim_timings:
            # frames_to_plot = list(range(stim-8, stim+35))
            flu.append(expobj.raw[cell_idx][stim - pre_stim:stim + post_stim])

        flu_dff = []
        for trace in flu:
            mean = np.mean(trace[0:pre_stim])
            trace_dff = ((trace - mean) / mean) * 100
            flu_dff.append(trace_dff)

        all_cells_dff.append(np.mean(flu_dff, axis=0))

        thresh = np.mean(np.mean(flu_dff, axis=0)[pre_stim + 10:pre_stim + 100])
        if thresh > 30:
            good_std_cells.append(cell)
            good_std_cells_dff_exc.append(np.mean(flu_dff, axis=0))
        elif thresh < -30:
            good_std_cells.append(cell)
            good_std_cells_dff_inh.append(np.mean(flu_dff, axis=0))

        flu_std = []
        std = np.std(flu)
        mean = np.mean(flu[0:pre_stim])
        for trace in flu:
            df_stdf = (trace - mean) / std
            flu_std.append(df_stdf)

        # thresh = np.mean(np.mean(flu_std, axis=0)[pre_stim+10:pre_stim+30])
        #
        # if thresh > 1*std:
        #     good_std_cells.append(cell)
        #     good_std_cells_dff_exc.append(np.mean(flu_dff, axis=0))
        # elif thresh < -1*std:
        #     good_std_cells.append(cell)
        #     good_std_cells_dff_inh.append(np.mean(flu_dff, axis=0))

        print('Pre-stim mean:', mean)
        print('Pre-stim std:', std)
        print('Post-stim dff:', thresh)
        print('                            ')

        # flu_avg = np.mean(flu_dff, axis=0)
        # std = np.std(flu_dff, axis=0)
        # ci = 1.960 * (std/np.sqrt(len(flu_dff))) # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
        # x = list(range(-pre_stim, post_stim))
        # y = flu_avg
        #
        # fig, ax = plt.subplots()
        # ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1) # plot confidence interval
        # ax.axvspan(0, 10, alpha=0.2, color='red')
        # ax.plot(x, y)
        # fig.suptitle('Cell %s' % cell)
        # plt.show()

aoutils.plot_photostim_avg(dff_array=all_cells_dff, pre_stim=pre_stim, post_stim=post_stim, title=title)

################
cell_idx = expobj.cell_id.index(3863)
std = np.std(expobj.raw[cell_idx])
mean = np.mean(expobj.raw[cell_idx])

plt.figure(figsize=(50, 3))
fig, ax = plt.subplots()
ax.axhline(mean + 2.5 * std)
plt.plot(expobj.raw[cell_idx])
fig.show()


################ ARCHIVED CODE BELOW

# %% THE FOLLOWING FUNCS HAVE BEEN MOVED TO all_optical_utils.py AS THEY ARE NOW MOSTLY STABLE
@njit
def moving_average(a, n=4):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


@njit
def _good_cells(cell_ids, raws, photostim_frames, radiuses, min_radius_pix, max_radius_pix):
    good_cells = []
    len_cell_ids = len(cell_ids)
    for i in range(len_cell_ids):
        # print(i, " out of ", len(cell_ids), " cells")
        raw = raws[i]
        raw_ = np.delete(raw, photostim_frames)
        raw_dff = aoutils.normalize_dff_jit(raw_)
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


def _good_photostim_cells(expobj, groups_per_stim=1, std_thresh=1, dff_threshold=None, pre_stim=expobj.pre_stim,
                          post_stim=expobj.post_stim):
    '''
    make sure to specify std threshold to use for filtering
    the pre-stim and post-stim args specify which pre-stim and post-stim frames to consider for filtering
    '''
    expobj.good_photostim_cells = []
    expobj.good_photostim_cells_responses = []
    expobj.good_photostim_cells_stim_responses_dF_stdF = []
    expobj.good_photostim_cells_stim_responses_dFF = []
    total = 0  # use to tally up how many cells across all groups are filtered in
    total_considered = 0  # use to tally up how many cells were looked at for their photostim response.
    for group in range(len(expobj.s2p_photostim_targets)):
        print('\nGroup %s' % group)
        stim_timings = expobj.stim_start_frames[group::int(expobj.n_groups / groups_per_stim)]
        title = 'SLM photostim Group #%s' % group
        targeted_cells = [cell for cell in expobj.s2p_photostim_targets[group] if cell in expobj.good_cells]

        # collect photostim timed average dff traces of photostim targets
        targets_dff = []
        pre_stim = pre_stim
        post_stim = post_stim
        for cell in targeted_cells:
            # print('considering cell # %s' % cell)
            if cell in expobj.cell_id:
                cell_idx = expobj.cell_id.index(cell)
                flu = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
                       stim not in expobj.seizure_frames]

                flu_dff = []
                for trace in flu:
                    mean = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean) / mean) * 100
                    flu_dff.append(trace_dff)

                targets_dff.append(np.mean(flu_dff, axis=0))

        # FILTER CELLS WHERE PHOTOSTIMULATED TARGETS FIRE > 1*std ABOVE PRE-STIM
        good_photostim_responses = {}
        good_photostim_cells = []
        good_targets_dF_stdF = []
        good_targets_dff = []
        std_thresh = std_thresh
        for cell in targeted_cells:
            trace = targets_dff[
                targeted_cells.index(cell)]  # trace = averaged dff trace across all photostims. for this cell
            pre_stim_trace = trace[:pre_stim]
            # post_stim_trace = trace[pre_stim + expobj.duration_frames:post_stim]
            mean_pre = np.mean(pre_stim_trace)
            std_pre = np.std(pre_stim_trace)
            # mean_post = np.mean(post_stim_trace[:10])
            dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
            # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
            response = np.mean(trace[
                               pre_stim + expobj.duration_frames:pre_stim + 3 * expobj.duration_frames])  # calculate the dF over pre-stim mean F response within the response window
            if dff_threshold is None:
                thresh_ = mean_pre + std_thresh * std_pre
            else:
                thresh_ = mean_pre + dff_threshold  # need to triple check before using
            if response > thresh_:  # test if the response passes threshold
                good_photostim_responses[cell] = response
                good_photostim_cells.append(cell)
                good_targets_dF_stdF.append(dF_stdF)
                good_targets_dff.append(trace)
                print('Cell #%s - dFF post-stim: %s (threshold value = %s)' % (cell, response, thresh_))

        expobj.good_photostim_cells.append(good_photostim_cells)
        expobj.good_photostim_cells_responses.append(good_photostim_responses)
        expobj.good_photostim_cells_stim_responses_dF_stdF.append(good_targets_dF_stdF)
        expobj.good_photostim_cells_stim_responses_dFF.append(good_targets_dff)

        print('%s cells filtered out of %s s2p target cells' % (len(good_photostim_cells), len(targeted_cells)))
        total += len(good_photostim_cells)
        total_considered += len(targeted_cells)

    expobj.good_photostim_cells_all = [y for x in expobj.good_photostim_cells for y in x]
    print('\nTotal number of good photostim responsive cells found: %s (out of %s)' % (total, total_considered))


# box plot with overlaid scatter plot with seaborn
plt.rcParams['figure.figsize'] = (20, 7)
sns.set_theme(style="ticks")
sns.catplot(x='group', y='Avg. dF/stdF response', data=expobj.average_responses_dfstdf, alpha=0.8, aspect=0.75,
            height=3.5)
ax = sns.boxplot(x='group', y='Avg. dF/stdF response', data=expobj.average_responses_dfstdf, color='white', fliersize=0,
                 width=0.5)
for i, box in enumerate(ax.artists):
    box.set_alpha(0.3)
    box.set_edgecolor('black')
    box.set_facecolor('white')
    for j in range(6 * i, 6 * (i + 1)):
        ax.lines[j].set_color('black')
        ax.lines[j].set_alpha(0.3)
# plt.savefig("/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/target_responses_avg %s.svg" % experiment)
plt.setp(ax.get_xticklabels(), rotation=45)
plt.suptitle('%s' % experiment, fontsize=10)
plt.show()


def plot_flu_trace(expobj, idx, slm_group=None, to_plot='raw'):
    raw = expobj.raw[idx]
    raw_ = np.delete(raw, expobj.photostim_frames)
    raw_dff = aoutils.normalize_dff(raw_)
    std_dff = np.std(raw_dff, axis=0)
    std = np.std(raw_, axis=0)

    x = []
    # y = []
    for j in np.arange(len(raw_dff), step=4):
        avg = np.mean(raw_dff[j:j + 4])
        if avg > np.mean(raw_dff) + 2.5 * std_dff:
            x.append(j)
            # y.append(0)

    if to_plot == 'raw':
        to_plot_ = raw
        to_thresh = std
    elif to_plot == 'dff':
        to_plot_ = raw_dff
        to_thresh = std_dff
    else:
        ValueError('specify to_plot as either "raw" or "dff"')

    plt.figure(figsize=(20, 3))
    plt.plot(to_plot_, linewidth=0.1)
    if to_plot == 'raw':
        plt.suptitle(('raw flu for cell #%s' % expobj.cell_id[idx]), horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=15, y=1.00)
    elif to_plot == 'dff':
        plt.scatter(x, y=[0] * len(x), c='r', linewidth=0.10)
        plt.axhline(y=np.mean(to_plot_) + 2.5 * to_thresh, c='green')
        plt.suptitle(('%s flu for cell #%s' % (to_plot, expobj.cell_id[idx])), horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=15, y=1.00)

    if slm_group is not None:
        for i in expobj.stim_start_frames[slm_group::expobj.n_groups]:
            plt.axvline(x=i - 1, c='gray', alpha=0.1)

    if len(expobj.seizure_frames) > 0:
        plt.scatter(expobj.seizure_frames, y=[-20] * len(x), c='g', linewidth=0.10)

    # plt.ylim(0, 300)
    plt.show()


def plot_photostim_avg(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None,
                       x_label=None, y_label=None, savepath=None):
    dff_array = dff_array[:, :pre_stim + post_stim]
    len_ = len(dff_array)
    flu_avg = np.mean(dff_array, axis=0)
    x = list(range(-pre_stim, post_stim))

    fig, ax = plt.subplots()
    ax.margins(0)
    ax.axvspan(0, stim_duration, alpha=0.2, color='green')
    for cell_trace in dff_array:
        ax.plot(x, cell_trace, linewidth=1, alpha=0.8)
    ax.plot(x, flu_avg, color='black', linewidth=2)  # plot median trace
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    ax.set_title((title + ' - %s' % len_ + ' cells'), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = round(item / expobj.fps, 1)
    ax.set_xticklabels(labels)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savepath:
        plt.savefig(savepath)
    plt.show()


def calculate_reliability(expobj, groups_per_stim=1, dff_threshold=20, pre_stim=expobj.pre_stim,
                          post_stim=expobj.post_stim):
    '''calculates the percentage of successful photoresponsive trials for each targeted cell, where success is post
     stim response over the dff_threshold'''
    reliability = {}  # dict will be used to store the reliability results for each targeted cell
    targets_dff_all_stimtrials = {}  # dict will contain the peri-stim dFF for each cell by the cell_idx
    stim_timings = expobj.stim_start_frames
    for group in range(len(expobj.s2p_photostim_targets)):
        print('\nProcessing Group %s' % group)
        if groups_per_stim == 1:
            stim_timings = expobj.stim_start_frames[group::expobj.n_groups]

        targeted_cells = [cell for cell in expobj.s2p_photostim_targets[group] if cell in expobj.good_cells]

        # collect photostim timed average dff traces of photostim targets
        for cell in targeted_cells:
            # print('considering cell # %s' % cell)
            if cell in expobj.cell_id:
                cell_idx = expobj.cell_id.index(cell)
                flu = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
                       stim not in expobj.seizure_frames]

                flu_dff = []
                success = 0
                for trace in flu:
                    # calculate dFF (noramlized to pre-stim) for each trace
                    mean = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean) / mean) * 100
                    flu_dff.append(trace_dff)

                    # calculate if the current trace beats dff_threshold for calculating reliability (note that this happens over a specific window just after the photostim)
                    response = np.mean(trace[
                                       pre_stim + expobj.duration_frames:pre_stim + 3 * expobj.duration_frames])  # calculate the dF over pre-stim mean F response within the response window
                    if response >= round(dff_threshold):
                        success += 1

                targets_dff_all_stimtrials[cell_idx] = np.array(
                    flu_dff)  # add the trials x peri-stim dFF as an array for each cell
                reliability[cell_idx] = success / len(stim_timings) * 100.
    print(reliability)
    return reliability, targets_dff_all_stimtrials


d = {}
# d['group'] = [int(expobj.good_photostim_cells.index(x)) for x in expobj.good_photostim_cells for y in x]
d['group'] = ['non-target'] * (len(expobj.good_cells))
for stim in expobj.stim_start_frames:
    d['%s' % stim] = [None] * len(expobj.good_cells)
df = pd.DataFrame(d, index=expobj.good_cells)
# population dataframe
for group in cell_groups:
    # hard coded number of stim. groups as the 0 and 1 in the list of this for loop
    if group == 'non-target':
        for stim in expobj.stim_start_frames:
            cells = [i for i in expobj.good_cells if i not in expobj.good_photostim_cells_all]
            for cell in cells:
                cell_idx = expobj.cell_id.index(cell)
                trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                mean_pre = np.mean(trace[0:expobj.pre_stim])
                trace_dff = ((trace - mean_pre) / abs(mean_pre))  # * 100
                std_pre = np.std(trace[0:expobj.pre_stim])
                # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
                dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
                response = np.mean(trace_dff[
                                   expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])
                df.at[cell, '%s' % stim] = round(response, 4)
    elif 'SLM Group' in group:
        cells = expobj.good_photostim_cells[int(group[-1])]
        for stim in expobj.stim_start_frames:
            for cell in cells:
                cell_idx = expobj.cell_id.index(cell)
                trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                mean_pre = np.mean(trace[0:expobj.pre_stim])
                trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
                std_pre = np.std(trace[0:expobj.pre_stim])
                # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
                dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
                response = np.mean(trace_dff[
                                   expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])
                df.at[cell, '%s' % stim] = round(response, 4)
                df.at[cell, 'group'] = group


# moved to the alloptical_plotting.py file
def bar_with_points(data, title='', x_tick_labels=[], points=True, bar=True, colors=['black'], ylims=None, xlims=None,
                    x_label=None, y_label=None, alpha=0.2, savepath=None):
    """
    general purpose function for plotting a bar graph of multiple categories with the individual datapoints shown
    as well. The latter is achieved by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph.

    :param data: list; provide data from each category as a list and then group all into one list
    :param title: str; title of the graph
    :param x_tick_labels: labels to use for categories on x axis
    :param points: bool; if True plot individual data points for each category in data using scatter function
    :param bar: bool, if True plot the bar, if False plot only the mean line
    :param colors: colors (by category) to use for each x group
    :param ylims: tuple; y axis limits
    :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
    :param x_label: x axis label
    :param y_label: y axis label
    :param alpha: transparency of the individual points when plotted in the scatter
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :return: matplotlib plot
    """

    w = 0.3  # bar width
    x = list(range(len(data)))
    y = data

    # plt.figure(figsize=(2, 10))
    fig, ax = plt.subplots(figsize=(2 * len(x), 3))
    if not bar:
        for i in x:
            ax.plot(np.linspace(x[i] - w / 2, x[i] + w / 2, 3), [np.mean(yi) for yi in y] * 3, color=colors[i])
        lw = 0,
        edgecolor = None
    else:
        edgecolor = 'black',
        lw = 1

    # plot bar graph, or if no bar (when lw = 0) then use it to plot the error bars
    ax.bar(x,
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi) for yi in y],  # error bars
           capsize=3,  # error bar cap width in points
           width=w,  # bar width
           linewidth=lw,  # width of the bar edges
           # tick_label=x_tick_labels,
           edgecolor=edgecolor,
           color=(0, 0, 0, 0),  # face color transparent
           )
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)

    if xlims:
        ax.set_xlim([xlims[0] - 1, xlims[1] + 1])
    elif len(x) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims = [-1, 1]
        ax.set_xlim(xlims)

    if points:
        for i in x:
            # distribute scatter randomly across whole width of bar
            ax.scatter(x[i] + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i], alpha=alpha)

    if ylims:
        ax.set_ylim(ylims)
    elif len(x) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    ax.set_title((title), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savepath:
        plt.savefig(savepath)
    plt.show()


# moved to the alloptical_processing_photostim.py file
# ###### IMPORT pkl file containing expobj

# determine which frames to retrieve from the overall total s2p output
trials = ['t-005', 't-006', 't-008', 't-009', 't-010']
total_frames_stitched = 0;
fr_curr_trial = None
for trial_ in trials:
    pkl_path_2 = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
    with open(pkl_path_2, 'rb') as f:
        expobj = pickle.load(f)
        # import suite2p data
    total_frames_stitched += expobj.n_frames
    if trial_ == trial:
        fr_curr_trial = [total_frames_stitched - expobj.n_frames, total_frames_stitched]

with open(pkl_path, 'rb') as f:
    expobj = pickle.load(f)

# suite2p processing on expobj

s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-pre-4ap-08x/plane0'
# s2p_path = '/Users/prajayshah/Documents/data-to-process/2020-12-18/suite2p/alloptical-2p-pre-4ap-08x/plane0'
# flu, spks, stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)


# s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-18/suite2p/photostim-4ap_stitched/plane0'
expobj.s2pProcessing(s2p_path=s2p_path, subset_frames=fr_curr_trial, subtract_neuropil=True)
# if needed for pkl expobj generated from older versions of Vape
expobj.target_coords_all = expobj.target_coords
expobj.s2p_targets()

# expobj.target_coords_all = expobj.target_coords

# flu, expobj.spks, expobj.stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)

aoutils.s2pMaskStack(obj=expobj, pkl_list=[pkl_path], s2p_path=s2p_path,
                     parent_folder='/home/pshah/mnt/qnap/Analysis/2020-12-18/')

# %% (quick) plot individual fluorescence traces - see InteractiveMatplotlibExample to make these plots interactively
# # plot raw fluorescence traces
# plt.figure(figsize=(50,3))
# for i in expobj.s2p_cell_targets:
#     plt.plot(expobj.raw[i], linewidth=0.1)
# plt.xlim(0, len(expobj.raw[0]))
# plt.show()

# plotting the distribution of radius and aspect ratios - should this be running before the filtering step which is right below????????

to_plot = plot_cell_radius_aspectr(expobj, expobj.stat, to_plot='radius')
a = [i for i in to_plot if i > 6]
id = to_plot.index(min(a))
# expobj.good_cells[id]

id = expobj.cell_id.index(1937)
expobj.stat[id]

# ###### CODE TO FILTER CELLS THAT ARE ACTIVE AT LEAST ONCE FOR >2.5*std

# pull out needed variables because numba doesn't work with custom classes (such as this all-optical class object)
cell_ids = expobj.cell_id
raws = expobj.raw
# expobj.append_seizure_frames(bad_frames=None)
photostim_frames = expobj.photostim_frames
radiuses = expobj.radius

# initial quick run to allow numba to compile the function - not sure if this is actually creating time savings
_ = aoutils._good_cells(cell_ids=cell_ids[:3], raws=raws, photostim_frames=expobj.photostim_frames, radiuses=radiuses,
                        min_radius_pix=2.5, max_radius_pix=8.5)
expobj.good_cells = aoutils._good_cells(cell_ids=cell_ids, raws=raws, photostim_frames=expobj.photostim_frames,
                                        radiuses=radiuses,
                                        min_radius_pix=2.5, max_radius_pix=8.5)

# filter for GOOD PHOTOSTIM. TARGETED CELLS with responses above threshold

expobj.pre_stim = 15  # specify pre-stim and post-stim periods of analysis and plotting
expobj.post_stim = 150

# function for gathering all good photostim cells who respond on average across all trials to the photostim
# note that the threshold for this is 1 * std of the prestim raw flu (fluorescence trace)

aoutils._good_photostim_cells(expobj=expobj, groups_per_stim=3, pre_stim=expobj.pre_stim, post_stim=expobj.post_stim,
                              dff_threshold=None)
# TODO what does threshold value mean? add more descriptive print output for that

# (full) plot individual cell's flu or dFF trace, with photostim. timings for that cell
cell = 1
# group = [expobj.good_photostim_cells.index(i) for i in expobj.good_photostim_cells if cell in i][0]  # this will determine which slm group's photostim to plot on the flu trace
group = 1

# plot flu trace of selected cell with the std threshold
idx = expobj.cell_id.index(cell)
plot_flu_trace(expobj=expobj, idx=idx, slm_group=group, to_plot='dff');
print(expobj.stat[idx])  # TODO don't keep a vague argument requirement for to_plot

# %%
##### SAVE expobj as PKL
# Pickle the expobject output to save it for analysis
pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)


def save_pkl(expobj, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("pkl saved to %s" % pkl_path)


def find_photostim_frames(expobj):
    '''finds all photostim frames and saves them into the bad_frames.npy file'''
    photostim_frames = []
    for j in expobj.stim_start_frames:
        for i in range(
                expobj.duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
            photostim_frames.append(j + i)

    expobj.photostim_frames = photostim_frames
    # print(photostim_frames)
    print('/// Original # of frames:', expobj.n_frames, 'frames ///')
    print('/// # of Photostim frames:', len(photostim_frames), 'frames ///')
    print('/// Minus photostim. frames total:', expobj.n_frames - len(photostim_frames), 'frames ///')


find_photostim_frames(expobj)

###### CODE TO FILTER TRIALS WHERE PHOTOSTIMULATED CELLS FIRE > 1*std ABOVE PRE-STIM
good_photostim_trials = []
good_photostimtrials_dF_stdF = []
good_photostimtrials_dFF = []
photostimtrials_dF_stdF = []
photostimtrials_dFF = []
for i in stim_timings:
    photostimcells_dF_stdF = []
    photostimcells_dFF = []
    for cell_idx in targets:
        trace = expobj.raw[expobj.cell_id.index(cell_idx)][
                i - pre_stim:i + post_stim]  # !!! cannot use the raw trace <-- avg of from all photostimmed cells
        pre_stim_trace = trace[:pre_stim]
        post_stim_trace = trace[pre_stim + 10:post_stim]
        mean_pre = np.mean(pre_stim_trace)
        std_pre = np.std(pre_stim_trace)
        mean_post = np.mean(post_stim_trace)
        dF_stdF = (trace - mean_pre) / std_pre
        dFF = (trace - mean_pre) / mean_pre
        photostimcells_dF_stdF.append(dF_stdF)
        photostimcells_dFF.append(dFF)

    trial_dF_stdF = np.mean(photostimcells_dF_stdF, axis=0)
    trial_dFF = np.mean(photostimcells_dFF, axis=0)

    photostimtrials_dF_stdF.append(trial_dF_stdF)
    photostimtrials_dFF.append(trial_dFF)

    thresh = np.mean(trial_dFF[pre_stim + 10:pre_stim + 20])
    thresh_ = np.mean(trial_dF_stdF[pre_stim + 10:pre_stim + 20])

    # if thresh > 0.3:
    #     good_photostim_trials.append(i)

    if thresh_ > 1:
        good_photostim_trials.append(i)
        good_photostimtrials_dF_stdF.append(trial_dF_stdF)
        good_photostimtrials_dFF.append(trial_dFF)

# check to see what the new trials' photostim response look like
targets_dff_filtered = []
pre_stim = 10
post_stim = 250
for cell in targets:
    if cell in expobj.cell_id:
        cell_idx = expobj.cell_id.index(cell)
        flu = []
        for stim in good_photostim_trials:
            # frames_to_plot = list(range(stim-8, stim+35))
            flu.append(expobj.raw[cell_idx][stim - pre_stim:stim + post_stim])

        flu_dff = []
        for trace in flu:
            mean = np.mean(trace[0:pre_stim])
            trace_dff = ((trace - mean) / mean) * 100
            flu_dff.append(trace_dff)

        targets_dff_filtered.append(np.mean(flu_dff, axis=0))

        # flu_avg = np.mean(flu_dff, axis=0)
        # std = np.std(flu_dff, axis=0)
        # ci = 1.960 * (std/np.sqrt(len(flu_dff))) # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
        # x = list(range(-pre_stim, post_stim))
        # y = flu_avg
        #
        # fig, ax = plt.subplots()
        # ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1) # plot confidence interval
        # ax.axvspan(0, 10, alpha=0.2, color='red')
        # ax.plot(x, y)
        # fig.suptitle('Cell %s' % cell)
        # plt.show()

aoutils.plot_photostim_avg(dff_array=targets_dff_filtered, pre_stim=pre_stim, post_stim=post_stim, title=title)
aoutils.plot_photostim_(dff_array=targets_dff_filtered, pre_stim=pre_stim, post_stim=post_stim, title=title)

# now plot to see what the dF_stdF trace looks like
plot_photostim_(dff_array=good_photostimtrials_dFF, pre_stim=pre_stim, post_stim=post_stim, title='good trials dFF')
plot_photostim_(dff_array=photostimtrials_dFF, pre_stim=pre_stim, post_stim=post_stim, title='dFF')

fig, ax = plt.subplots()
for i in range(len(photostimtrials_dFF)):
    plt.plot(photostimtrials_dFF[i], linewidth=1.05)
ax.axvspan(10, 20, alpha=0.2, color='red')
plt.show()

fig, ax = plt.subplots()
plt.plot(trial_dFF)
ax.axvspan(10, 20, alpha=0.2, color='red')
plt.show()
