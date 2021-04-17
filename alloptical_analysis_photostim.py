# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
import utils.funcs_pj as pj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import draw

###### IMPORT pkl file containing data in form of expobj
trial = 't-013'
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

if not hasattr(expobj, 's2p_path'):
    expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-1x-alltrials/plane0'

if not hasattr(expobj, 'meanRawFluTrace'):
    expobj.mean_raw_flu_trace(plot=True)
else:
    aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])

aoplot.plotLfpSignal(expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])


#%%#####################################################################################################################

#### -------------------- ALL OPTICAL PHOTOSTIM AND ETC. ANALYSIS STEPS ################################################

########################################################################################################################

# %% PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - PHOTOSTIM TARGETTED suite2p cells

# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
x = np.asarray([i for i in expobj.targets_dfstdF_avg])
# y_label = 'pct. dFF (normalized to prestim period)'
y_label = 'dFstdF (normalized to prestim period)'

aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, stim_duration=expobj.stim_duration_frames, pre_stim=expobj.pre_stim,
                              post_stim=expobj.post_stim,
                              title=(experiment + '- responses of all photostim targets'),
                              y_label=y_label, x_label='Time post-stimulation (seconds)')


# %% PLOT ENTIRE TRIAL - targeted suite2p cells plotted individually as subplots

expobj.raw_s2ptargets = [expobj.raw[expobj.cell_id.index(i)] for i in expobj.s2p_cell_targets if i in expobj.good_photostim_cells_all]
expobj.dff_s2ptargets = aoutils.normalize_dff(np.array(expobj.raw_s2ptargets))
# expobj.targets_dff_base = aoutils.normalize_dff_baseline(
#     arr=expobj.raw_df.loc[[str(x) for x in expobj.s2p_cell_targets]],
#     baseline_array=expobj.baseline_raw_df)
# plot_photostim_subplots(dff_array=SLMTargets_dff,
#                 title=(experiment + '%s responses of responsive cells' % len(expobj.good_photostim_cells_stim_responses_dFF)))
to_plot = expobj.dff_s2ptargets

aoplot.plot_photostim_traces_overlap(array=to_plot, expobj=expobj, exclude_id=[expobj.s2p_cell_targets.index(cell) for cell in [211, 400, 542]],
                                     y_lims=[0, 5000], title=(experiment + '-'))

aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Frames',
                             y_label='Raw Flu', title=experiment)


# # plot the photostim targeted cells as a heatmap
# dff_array = expobj.SLMTargets_dff[:, :]
# w = 10
# dff_array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array]
# dff_array = np.asarray(dff_array)
#
# plt.figure(figsize=(5, 10));
# sns.heatmap(dff_array, cmap='RdBu_r', vmin=0, vmax=500);
# plt.show()



# %% plot SLM photostim individual targets -- full traces, dff normalized

# aoplot.plot_photostim_traces(array=expobj.SLMTargets_stims_raw, expobj=expobj, x_label='Frames',
#                                y_label='Raw Flu',
#                                title=(experiment))

expobj.dff_SLMTargets = aoutils.normalize_dff(np.array(expobj.raw_SLMTargets))


# make rolling average for these plots to smooth out the traces a little more
w = 10
to_plot = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets[:3]]

aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Frames',
                             y_label='dFF Flu', title=experiment)

# len_ = len(array)
# fig, axs = plt.subplots(nrows=len_, sharex=True, figsize=(30, 3 * len_))
# for i in range(len(axs)):
#     axs[i].plot(array[i], linewidth=1, color='black')
#     for j in expobj.stim_start_frames:
#         axs[i].axvline(x=j, c='gray', alpha=0.7, linestyle='--')
#     if len_ == len(expobj.s2p_cell_targets):
#         axs[i].set_title('Cell # %s' % expobj.s2p_cell_targets[i])
# plt.show()

# %% plot peri- photostim traces for individual SLM target individual, incl. individual traces for each stim

pre_stim = expobj.pre_stim
post_stim = expobj.post_stim
expobj.SLMTargets_stims_dff, expobj.SLMTargets_stims_dffAvg, expobj.SLMTargets_stims_dfstdF, \
expobj.SLMTargets_stims_dfstdF_avg, expobj.SLMTargets_stims_raw, expobj.SLMTargets_stims_rawAvg = \
    expobj.get_alltargets_stim_traces_norm(pre_stim=pre_stim, post_stim=post_stim)


# array = (np.convolve(SLMTargets_stims_raw[targets_idx], np.ones(w), 'valid') / w)

# targets_idx = 0
plot = False
for i in range(0, expobj.n_targets_total):
    SLMTargets_stims_raw, SLMTargets_stims_dff, SLMtargets_stims_dfstdF = expobj.get_alltargets_stim_traces_norm(targets_idx=i, pre_stim=pre_stim,
                                                                                               post_stim=post_stim)
    if plot:
        w = 2
        array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in SLMTargets_stims_raw]
        random_sub = np.random.randint(0,100,10)
        aoplot.plot_periphotostim_avg(arr=SLMtargets_stims_dfstdF[random_sub], expobj=expobj, stim_duration=expobj.stim_duration_frames,
                                      title='Cell ' + str(i), pre_stim=pre_stim, post_stim=post_stim, color='steelblue', y_lims=[-0.5, 2.5])
    # plt.show()





# %% photostim. SUCCESS RATE MEASUREMENTS and PLOT - PHOTOSTIM TARGETED CELLS
# measure, for each cell, the pct of trials in which the dFF > 20% post stim (normalized to pre-stim avgF for the trial and cell)
# can plot this as a bar plot for now showing the distribution of the reliability measurement

SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))

expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
    aoutils.calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=expobj.SLMTargets_stims_raw,
                                      dfstdf_threshold=0.3, pre_stim=expobj.pre_stim, sz_filter=False,
                                      verbose=True, plot=False)

expobj.save()

random_sub = np.random.randint(0,expobj.n_targets_total, 5)
w = 3
arr_to_plot = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.raw_SLMTargets[random_sub]]

aoplot.plot_photostim_traces(array=arr_to_plot, expobj=expobj, x_label='Frames',
                             y_label='dFF Flu', title=experiment, scatter=np.array(list(expobj.hits_cells.values()))[random_sub],
                             line_ids=random_sub)



# %% ########## BAR PLOT showing average success rate of photostimulation

pj.bar_with_points(data=[list(expobj.StimSuccessRate_cells.values())], x_tick_labels=['t-013'], ylims=[0, 100], bar=False, y_label ='% success stims.',
                   title='%s success rate of stim responses' % trial, expand_size_x=2)


# plot across different groups
# t009_pre_4ap_reliability = list(expobj.StimSuccessRate_cells.values())
# t011_post_4ap_reliabilty = list(expobj.StimSuccessRate_cells.values())  # reimport another expobj for post4ap trial
# t013_post_4ap_reliabilty = list(expobj.StimSuccessRate_cells.values())  # reimport another expobj for post4ap trial
#
# pj.bar_with_points(data=[t009_pre_4ap_reliability, t011_post_4ap_reliabilty, t013_post_4ap_reliabilty],
#                    x_tick_labels=['t-009', 't-011', 't-013'], colors=['green', 'deeppink'],
#                    ylims=[0, 100], bar=False, title='reliability of stim responses', expand_size_y=1.2, expand_size_x=1.2)


# %% PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - NON - TARGETS
x = np.asarray([i for i in expobj.dfstdF_traces_avg])
# y_label = 'pct. dFF (normalized to prestim period)'
y_label = 'dFstdF (normalized to prestim period)'

aoplot.plot_periphotostim_avg(dff_array=x, expobj=expobj, stim_duration=expobj.stim_duration_frames, pre_stim=expobj.pre_stim,
                              post_stim=expobj.post_stim,
                              title=(experiment + '- responses of all photostim targets'),
                              y_label=y_label, x_label='Time post-stimulation (seconds)')

# %% PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D

x = np.asarray([i for i in expobj.targets_dfstdF_avg])
aoplot.plot_traces_heatmap(x, vmin=-1, vmax=1, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames - 1,
                           title=(experiment + ' - targets only'))

x = np.asarray([i for i in expobj.dfstdF_traces_avg])
aoplot.plot_traces_heatmap(x, vmin=-0.5, vmax=0.5, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames - 1,
                           title=(experiment + ' - nontargets'))

# %% BAR PLOT PHOTOSTIM RESPONSES SIZE - TARGETS vs. NON-TARGETS
# collect photostim timed average dff traces
all_cells_dff = []
good_std_cells = []

# calculate and plot average response of cells in response to all stims as a bar graph


# there's a bunch of very high dFF responses of cells
# remove cells with very high average response values from the dff dataframe
# high_responders = expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500].index.values
# expobj.dff_responses_all_cells.iloc[high_responders[0], 1:]
# list(expobj.dff_responses_all_cells.iloc[high_responders[0], 1:])
# idx = expobj.cell_id.index(1668);
# aoplot.plot_flu_trace(expobj=expobj, idx=idx, to_plot='dff', size_factor=2)


# need to troubleshoot how these scripts are calculating the post stim responses for the non-targets because some of them seem ridiculously off
# --->  this should be okay now since I've moved to df_stdf correct?



## using pj.bar_with_points() for a nice bar graph
group1 = list(expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'photostim target'][
                  'Avg. dF/stdF response'])
group2 = list(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'non-target']['Avg. dF/stdF response'])
pj.bar_with_points(data=[group1, group2], x_tick_labels=['photostim target', 'non-target'], xlims=[0, 0.6],
                   ylims=[0, 1.5], bar=False,
                   colors=['red', 'black'], title=experiment, y_label='Avg dF/stdF response', expand_size_y=1.3,
                   expand_size_x=1.4)

# %% PLOT HEATMAP OF PHOTOSTIM. RESPONSES TO PHOTOSTIM FOR ALL CELLS
# - need to find a way to sort these responses that similar cells are sorted together
# - implement a heirarchical clustering method

stim_timings = [str(i) for i in expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing

# make heatmap of responses across all cells across all stims
df_ = expobj.dfstdf_all_cells[stim_timings]  # select appropriate stim time reponses from the pandas df
df_ = df_[df_.columns].astype(float)

plt.figure(figsize=(5, 15));
sns.heatmap(df_, cmap='seismic', vmin=-5, vmax=5, cbar_kws={"shrink": 0.25});
plt.show()


# %% PLOT imshow() XY locations with COLORS AS average response of ALL cells in FOV
aoplot.xyloc_responses(expobj, to_plot='dfstdf', clim=[-1, +1], plot_target_coords=True)



# %% PLOT seizure period as heatmap

sz = 2
sz_onset, sz_offset = expobj.stims_bf_sz[sz], expobj.stims_af_sz[sz+1]
x = expobj.raw[[expobj.cell_id.index(cell) for cell in expobj.good_cells], sz_onset:sz_offset]

## TODO organize individual cells in array in order of peak firing rate


stims = [(stim - sz_onset) for stim in expobj.stim_start_frames if sz_onset <= stim < sz_offset]
stims_off = [(stim + expobj.stim_duration_frames - 1) for stim in stims]

x_bf = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_bf_sz[sz])[0][0]]
x_af = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_af_sz[sz+1])[0][0]]

lfp_signal = expobj.lfp_signal[x_bf:x_af]

aoplot.plot_traces_heatmap(x, stim_on=stims, stim_off=stims_off, cmap='Spectral_r', figsize=(10,6),
                           title=('%s - seizure %s' % (trial, sz)), xlims=None, vmin=100, vmax=500,
                           lfp_signal=lfp_signal)


# %% plot the target photostim responses for individual targets for each stim over the course of the trial
#    (normalize to each target's overall mean response) and plot over the timecourse of the trial


SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
target_colors = pj.make_random_color_array(SLMtarget_ids)

# --- plot with mean FOV fluorescence signal
fig, ax1 = plt.subplots(figsize=[60, 6])
fig, ax1 = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', figsize=[20, 3], show=False,
                                      fig=fig, ax=ax1)
ax2 = ax1.twinx()
for cell in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[cell])
    # print(mean_response)
    for i in range(len(expobj.stim_start_frames)):
        response = expobj.responses_SLMtargets[cell][i] - mean_response
        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)
        ax2.scatter(x=expobj.stim_start_frames[i] + rand, y=response, color=target_colors[cell], alpha=0.70, s=15, zorder=4)
# for i in expobj.stim_start_frames:
#     plt.axvline(i)
plt.show()


# %% --- plot with LFP signal
fig, ax1 = plt.subplots(figsize=[60, 6])
fig, ax1 = aoplot.plotLfpSignal(expobj, stim_span_color=None, x_axis='frames', show=False, fig=fig, ax=ax1)
ax2 = ax1.twinx()
for cell in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[cell])
    # print(mean_response)
    for i in range(len(expobj.stim_times)):
        response = expobj.responses_SLMtargets[cell][i] - mean_response
        rand = np.random.randint(-10, 30, 1)[0] #* 1/(abs(response)**1/2)
        ax2.scatter(x=expobj.stim_times[i] + rand * 1e3, y=response, color=target_colors[cell], alpha=0.70, s=15, zorder=4)
# for i in expobj.stim_start_frames:
#     plt.axvline(i)
plt.show()

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
# TODO calculate probability of stimulation in 100x100um micron bins around targeted cell

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


# photostimulation of targeted cells before CSD, just after CSD, and a while after CSD


# photostimulation of targeted cells before seizure, just after seizure, and a while after seizure


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


