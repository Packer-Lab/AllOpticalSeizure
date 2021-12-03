# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys; import os
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj

from skimage import draw

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)


expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-013')

"""######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""




# sys.exit()
"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


# %% superceded in aoresults-init-1) lists of trials to analyse for run_pre4ap_trials and run_post4ap_trials trials within experiments

allopticalResults.pre_4ap_trials = [
    ['RL108 t-009'],
    # ['RL108 t-010'],
    # ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],  # - pickle truncated .21/10/18 - analysis func jupyter run on .21/11/12
    ['RL109 t-014'],
    ['PS04 t-012',  # 'PS04 t-014',  - not sure what's wrong with PS04, but the photostim and Flu are falling out of sync .21/10/09
     'PS04 t-017'],
    ['PS05 t-010'],
    ['PS07 t-007'],
    ['PS07 t-009'],
    # ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],  # matching run_post4ap_trials trial cannot be analysed
    ['PS06 t-011'],
    # ['PS06 t-012'],  # matching run_post4ap_trials trial cannot be analysed
    # ['PS11 t-007'],
    ['PS11 t-010'],
    # ['PS17 t-005'],
    # ['PS17 t-006', 'PS17 t-007'],
    # ['PS18 t-006']
]

allopticalResults.post_4ap_trials = [
    ['RL108 t-013'],
    # ['RL108 t-011'],
    # ['RL109 t-020'],
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016', 'RL109 t-017'],
    # ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    ['PS07 t-017'],
    # ['PS06 t-014', 'PS06 t-015'], - missing seizure_lfp_onsets (no paired measurements mat file for trial .21/10/09)
    ['PS06 t-013'],
    # ['PS06 t-016'], - no seizures, missing seizure_lfp_onsets (no paired measurements mat file for trial .21/10/09)
    # ['PS11 t-016'],
    ['PS11 t-011'],
    # ['PS17 t-011'],
    # ['PS17 t-009'],
    # ['PS18 t-008']
]

assert len(allopticalResults.pre_4ap_trials) == len(allopticalResults.post_4ap_trials), print('pre trials %s ' % len(allopticalResults.pre_4ap_trials),
                                                                                              'post trials %s ' % len(allopticalResults.post_4ap_trials))


allopticalResults.trial_maps = {'pre': {}, 'post': {}}
allopticalResults.trial_maps['pre'] = {
    'a': ['RL108 t-009'],
    # 'b': ['RL108 t-010'],
    # 'c': ['RL109 t-007'],
    'd': ['RL109 t-008'],
    'e': ['RL109 t-013'],
    'f': ['RL109 t-014'],
    # 'g': ['PS04 t-012',  # 'PS04 t-014',  # - temp just until PS04 gets reprocessed
    #       'PS04 t-017'],
    'h': ['PS05 t-010'],
    'i': ['PS07 t-007'],
    'j': ['PS07 t-009'],
    # 'k': ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
    'l': ['PS06 t-011'],
    # 'm': ['PS06 t-012'],  # - t-016 missing sz lfp onsets
    # 'n': ['PS11 t-007'],
    'o': ['PS11 t-010'],
    # 'p': ['PS17 t-005'],
    # 'q': ['PS17 t-006', 'PS17 t-007'],
    # 'r': ['PS18 t-006']
}

allopticalResults.trial_maps['post'] = {
    'a': ['RL108 t-013'],
    # 'b': ['RL108 t-011'], -- need to redo sz boundary classifying processing
    # 'c': ['RL109 t-020'], -- need to redo sz boundary classifying processing
    'd': ['RL109 t-021'],
    'e': ['RL109 t-018'],
    'f': ['RL109 t-016', 'RL109 t-017'],
    # 'g': ['PS04 t-018'],  -- need to redo sz boundary classifying processing
    'h': ['PS05 t-012'],
    'i': ['PS07 t-011'],
    'j': ['PS07 t-017'],
    # 'k': ['PS06 t-014', 'PS06 t-015'],  # - missing seizure_lfp_onsets
    'l': ['PS06 t-013'],
    # 'm': ['PS06 t-016'],  # - missing seizure_lfp_onsets - LFP signal not clear, but there is seizures on avg Flu trace
    # 'n': ['PS11 t-016'],
    'o': ['PS11 t-011'],
    # 'p': ['PS17 t-011'],
    # 'q': ['PS17 t-009'],
    # 'r': ['PS18 t-008']
}

assert len(allopticalResults.trial_maps['pre'].keys()) == len(allopticalResults.trial_maps['post'].keys())

allopticalResults.save()









# %% #########################################################################################################################
#### END OF CODE THAT HAS BEEN REVIEWED SO FAR ##########################################################################
#########################################################################################################################

sys.exit()


















































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
    :param all_x: ls of x coords of cells in dataset
    :param all_y: ls of y coords of cells in dataset
    :param responses_of_cells: ls of responses of cells to plots
    :param bins: integer - number of bins to split FOV in (along one axis)
    :return: plot of binned 2d histograms
    """

    all_amps_real = responses_of_cells  # ls of photostim. responses
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
            # frames_to_plot = ls(range(stim-8, stim+35))
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

        # thresh = np.mean(np.mean(flu_std, axis=0)[pre_stim_sec+10:pre_stim_sec+30])
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
        # x = ls(range(-pre_stim_sec, post_stim_sec))
        # y = flu_avg
        #
        # fig, ax = plt.subplots()
        # ax.fill_between(x, (y - ci), (y + ci), edgecolor='b', alpha=.1) # plot confidence interval
        # ax.axvspan(0, 10, alpha=0.2, edgecolor='red')
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


################# - ARCHIVED NOV 11 2021
# %% 4) ###### IMPORT pkl file containing data in form of expobj, and run processing as needed (implemented as a loop currently)


expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre g.1')

plot = 1
if plot:
    aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Time', figsize=[20, 3])
    aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='time', figsize=[8, 2])
    aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered)
    aoplot.plot_lfp_stims(expobj)

for exptype in ['post', 'pre']:
    for key in allopticalResults.trial_maps[exptype].keys():
        if len(allopticalResults.trial_maps[exptype][key]) > 1:
            aoresults_map_id = []
            for i in range(len(allopticalResults.trial_maps[exptype][key])):
                aoresults_map_id.append('%s %s.%s' % (exptype, key, i))
        else:
            aoresults_map_id = ['%s %s' % (exptype, key)]

        for mapid in aoresults_map_id:
            expobj, experiment = aoutils.import_expobj(aoresults_map_id=mapid)

            plot = 0
            if plot:
                aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Time', figsize=[20, 3])
                aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='time', figsize=[8, 2])
                aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered)
                aoplot.plot_lfp_stims(expobj)

        # 5) any data processing -- if needed

        # expobj.paqProcessing()
        # expobj._findTargets()


        # if not hasattr(expobj, 's2p_path'):
        #     expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-1x-alltrials/plane0'

        # expobj.s2pProcessing(s2p_path=expobj.s2p_path, subset_frames=expobj.curr_trial_frames, subtract_neuropil=True,
        #                      baseline_frames=expobj.baseline_frames, force_redo=True)
            expobj._findTargetedS2pROIs(force_redo=True)
        # aoutils.s2pMaskStack(obj=expobj, pkl_list=[expobj.pkl_path], s2p_path=expobj.s2p_path, parent_folder=expobj.analysis_save_path, force_redo=True)
        #



            if not hasattr(expobj, 'meanRawFluTrace'):
                expobj.mean_raw_flu_trace(plot=True)

            # for suite2p detected non-ROIs
            expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
                expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
                aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                                        post_stim=expobj.post_stim)
            # for s2p detected target ROIs
            expobj.targets_dff, expobj.SLMTargets_stims_dffAvg, expobj.targets_dfstdF, \
            expobj.targets_dfstdF_avg, expobj.targets_stims_raw, expobj.targets_stims_raw_avg = \
                aoutils.get_s2ptargets_stim_traces(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                                   post_stim=expobj.post_stim)


            expobj.save()




# %% suite2p ROIs - PHOTOSTIM TARGETS - PLOT AVG PHOTOSTIM PRE- POST- STIM TRACE AVGed OVER ALL PHOTOSTIM. TRIALS

to_plot = 'dFstdF'

if to_plot == 'dFstdF':
    arr = np.asarray([i for i in expobj.targets_dfstdF_avg])
    y_label = 'dFstdF (normalized to prestim period)'
elif to_plot == 'dFF':
    arr = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg])
    y_label = 'dFF (normalized to prestim period)'
aoplot.plot_periphotostim_avg(arr=arr, expobj=expobj, pre_stim_sec=0.5, post_stim_sec=1.0,
                              title=(experiment + '- responses of all photostim targets'), figsize=[5,4],
                              x_label='Time post-stimulation (seconds)')


# %% SUITE2P ROIS - PHOTOSTIM TARGETS - PLOT ENTIRE TRIAL - individual ROIs plotted individually entire Flu trace

to_plot = expobj.dff_SLMTargets
aoplot.plot_photostim_traces_overlap(array=to_plot, expobj=expobj, y_lims=[0, 5000], title=(experiment + '-'))

aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Frames', y_label='dFF Flu',
                             title='%s %s - dFF SLM Targets' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))


# # plot the photostim targeted cells as a heatmap
# dff_array = expobj.SLMTargets_dff[:, :]
# w = 10
# dff_array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array]
# dff_array = np.asarray(dff_array)
#
# plt.figure(figsize=(5, 10));
# sns.heatmap(dff_array, cmap='RdBu_r', vmin=0, vmax=500);
# plt.show()



# %% SLM PHOTOSTIM TARGETS - plot individual, full traces, dff normalized

# make rolling average for these plots to smooth out the traces a little more
w = 3
to_plot = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets])
# to_plot = expobj.dff_SLMTargets

aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Frames',
                             y_label='dFF Flu', title=experiment)

aoplot.plot_photostim_traces_overlap(array=expobj.dff_SLMTargets, expobj=expobj, x_axis='Time (secs.)',
                                     title='%s - dFF Flu photostims' % experiment, figsize=(2*20, 2*len(to_plot)*0.15))

# len_ = len(array)
# fig, axs = plt.subplots(nrows=len_, sharex=True, figsize=(30, 3 * len_))
# for i in range(len(axs)):
#     axs[i].plot(array[i], linewidth=1, edgecolor='black')
#     for j in expobj.stim_start_frames:
#         axs[i].axvline(x=j, c='gray', alpha=0.7, linestyle='--')
#     if len_ == len(expobj.s2p_cell_targets):
#         axs[i].set_title('Cell # %s' % expobj.s2p_cell_targets[i])
# plt.show()

# array = (np.convolve(SLMTargets_stims_raw[targets_idx], np.ones(w), 'valid') / w)

# # targets_idx = 0
# plot = True
# for i in range(0, expobj.n_targets_total):
#     SLMTargets_stims_raw, SLMTargets_stims_dff, SLMtargets_stims_dfstdF = expobj.get_alltargets_stim_traces_norm(targets_idx=i, pre_stim_sec=pre_stim_sec,
#                                                                                                                  post_stim_sec=post_stim_sec)
#     if plot:
#         w = 2
#         array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in SLMTargets_stims_raw]
#         random_sub = np.random.randint(0,100,10)
#         aoplot.plot_periphotostim_avg(arr=SLMtargets_stims_dfstdF[random_sub], expobj=expobj, stim_duration=expobj.stim_duration_frames,
#                                       title='Target ' + str(i), pre_stim_sec=pre_stim_sec, post_stim_sec=post_stim_sec, color='steelblue', y_lims=[-0.5, 2.5])
#     # plt.show()


# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
# x = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])

y_label = 'dF/prestim_stdF'
aoplot.plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dfstdF_avg, expobj=expobj, stim_duration=expobj.stim_duration_frames,
                              figsize=[5, 4], y_lims=[-0.5, 3], pre_stim_sec=0.5, post_stim_sec=1.0,
                              title=(experiment + '- responses of all photostim targets'),
                              y_label=y_label, x_label='post-stimulation (seconds)')


# %% plotting of photostim. success rate

data = [np.mean(expobj.responses_SLMtargets_dfprestimf[i]) for i in range(expobj.n_targets_total)]

pj.plot_hist_density([data], x_label='response magnitude (dF/stdF)')
pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=[expobj.metainfo['trial']], ylims=[0, 100], bar=False, y_label='% success stims.',
                        title='%s success rate of stim responses' % expobj.metainfo['trial'], expand_size_x=2)



# %% SUITE2P NON-TARGETS - PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
x = np.asarray([i for i in expobj.dfstdF_traces_avg])
# y_label = 'pct. dFF (normalized to prestim period)'
y_label = 'dFstdF (normalized to prestim period)'

aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=0.5,
                              post_stim_sec=1.5, title='responses of s2p non-targets', y_label=y_label,
                              x_label='Time post-stimulation (seconds)', y_lims=[-1, 3])


# %% PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D

arr = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg]); vmin = -1; vmax = 1
arr = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg]); vmin = -20; vmax = 20
aoplot.plot_traces_heatmap(arr, expobj=expobj, vmin=-20, vmax=20, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames - 1,
                           title=('peristim avg trace heatmap' + ' - slm targets only'), x_label='Time')

arr = np.asarray([i for i in expobj.dfstdF_traces_avg]); vmin = -1; vmax = 1
arr = np.asarray([i for i in expobj.dff_traces_avg]); vmin = -20; vmax = 20
aoplot.plot_traces_heatmap(arr, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames - 1,
                           title=('peristim avg trace heatmap' + ' - nontargets'), x_label='Time')


# %% BAR PLOT PHOTOSTIM RESPONSES SIZE - TARGETS vs. NON-TARGETS
# collect photostim timed average dff traces
all_cells_dff = []
good_std_cells = []

# calculate and plot average response of cells in response to all stims as a bar graph


# there's a bunch of very high dFF responses of cells
# remove cells with very high average response values from the dff dataframe
# high_responders = expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500].index.values
# expobj.dff_responses_all_cells.iloc[high_responders[0], 1:]
# ls(expobj.dff_responses_all_cells.iloc[high_responders[0], 1:])
# idx = expobj.cell_id.index(1668);
# aoplot.plot_flu_trace(expobj=expobj, idx=idx, to_plot='dff', size_factor=2)


# need to troubleshoot how these scripts are calculating the post stim responses for the non-targets because some of them seem ridiculously off
# --->  this should be okay now since I've moved to df_stdf correct?

group1 = list(expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'photostim target'][
                  'Avg. dF/stdF response'])
group2 = list(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'non-target']['Avg. dF/stdF response'])
pj.plot_bar_with_points(data=[group1, group2], x_tick_labels=['photostim target', 'non-target'], xlims=[0, 0.6],
                        ylims=[0, 3], bar=False, colors=['red', 'black'], title=experiment, y_label='Avg dF/stdF response',
                        expand_size_y=2, expand_size_x=1)


# %% PLOT imshow() XY area locations with COLORS AS average response of ALL cells in FOV

aoplot.xyloc_responses(expobj, to_plot='dfstdf', clim=[-1, +1], plot_target_coords=True)


# %% PLOT INDIVIDUAL WHOLE TRACES AS HEATMAP OF PHOTOSTIM. RESPONSES TO PHOTOSTIM FOR ALL CELLS -- this is just the whole trace for each target, not avg over stims in any way
# - need to find a way to sort these responses that similar cells are sorted together
# - implement a heirarchical clustering method

stim_timings = [str(i) for i in expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing

# make heatmap of responses across all cells across all stims
df_ = expobj.dfstdf_all_cells[stim_timings]  # select appropriate stim time reponses from the pandas df
df_ = df_[df_.columns].astype(float)

plt.figure(figsize=(5, 15));
sns.heatmap(df_, cmap='seismic', vmin=-5, vmax=5, cbar_kws={"shrink": 0.25});
plt.show()