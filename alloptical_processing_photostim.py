## this file is for processing the photostim-experiment alloptical expobj object AFTER suite2p has been run
## the end of the script will update the expobj that was in the original pkl path

import sys; sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import os
import pickle
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils.funcs_pj as pj
import tifffile as tf

###### IMPORT pkl file containing expobj
trial = 't-016'
date = '2020-12-19'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)


expobj.sz_boundary_csv_done = False  # i know this is a rather very precarious thing here...

if not hasattr(expobj, 's2p_path'):
    expobj.s2p_path = input('input the suite2p path for this trial (to the plane0 folder)!!')
    expobj.save()

if not hasattr(expobj, 'meanRawFluTrace'):
    expobj.mean_raw_flu_trace(plot=True)
    expobj.save()

if not hasattr(expobj, 'stims_in_sz'):
    expobj.seizures_lfp_timing_matarray = expobj.seizures_info_array
    expobj.collect_seizures_info()
    expobj.save()

plot = False
if plot:
    aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
    # aoplot.plotLfpSignal(expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
    aoplot.plot_SLMtargets_Locs(expobj)
    aoplot.plot_lfp_stims(expobj)


# %% prep for importing data from suite2p for this whole experiment
# determine which frames to retrieve from the overall total s2p output

if not hasattr(expobj, 'suite2p_trials'):
    to_suite2p = ['t-003', 't-005', 't-010', 't-011', 't-012']  # specify all trials that were used in the suite2p runtotal_frames_stitched = 0
    baseline_trials = ['t-003', 't-005']  # specify which trials to use as spont baseline
    # note ^^^ this only works currently when the spont baseline trials all come first, and also back to back

    expobj.suite2p_trials = to_suite2p
    expobj.baseline_trials = baseline_trials
    expobj.save()

# main function that imports suite2p data and adds attributes to the expobj
expobj.subset_frames_current_trial(trial=trial, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials, force_redo=True)
expobj.s2pProcessing(s2p_path=expobj.s2p_path, subset_frames=expobj.curr_trial_frames, subtract_neuropil=True,
                     baseline_frames=expobj.baseline_frames, force_redo=True)
expobj.target_coords_all = expobj.target_coords
expobj.s2p_targets()
aoutils.s2pMaskStack(obj=expobj, pkl_list=[pkl_path], s2p_path=expobj.s2p_path, parent_folder=expobj.analysis_save_path, force_redo=True)


# %% STA - raw SLM targets processing

# collect raw Flu data from SLM targets
expobj.raw_traces_from_targets(force_redo=True)

plot = True
if plot:
    aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg, title='SLM targets location w/ mean Flu img')
    aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered, title='SLM targets location w/ registered mean Flu img')



#%%#####################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##### ------------------- processing for SLM targets Flu traces ########################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# collect SLM photostim individual targets -- individual, full traces, dff normalized
expobj.dff_SLMTargets = aoutils.normalize_dff(np.array(expobj.raw_SLMTargets))
expobj.save()

# collect and plot peri- photostim traces for individual SLM target, incl. individual traces for each stim
expobj.pre_stim = int(0.5 * expobj.fps)
expobj.post_stim = int(4 * expobj.fps)
expobj.SLMTargets_stims_dff, expobj.SLMTargets_stims_dffAvg, expobj.SLMTargets_stims_dfstdF, \
expobj.SLMTargets_stims_dfstdF_avg, expobj.SLMTargets_stims_raw, expobj.SLMTargets_stims_rawAvg = \
    expobj.get_alltargets_stim_traces_norm(pre_stim=expobj.pre_stim, post_stim=expobj.post_stim)


# %% photostim. SUCCESS RATE MEASUREMENTS and PLOT - SLM PHOTOSTIM TARGETED CELLS
# measure, for each cell, the pct of trials in which the dF_stdF > 20% post stim (normalized to pre-stim avgF for the trial and cell)
# can plot this as a bar plot for now showing the distribution of the reliability measurement

SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))

expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
    aoutils.calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=expobj.SLMTargets_stims_raw, dfstdf_threshold=0.3,
                                      pre_stim=expobj.pre_stim, sz_filter=False,
                                      verbose=True, plot=False)

expobj.save()

# %% SAVE RESULTS TO SUPEROBJECT

results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allOpticalresults = aoutils.import_resultsobj(pkl_path=results_object_path)

# %%
mean_reliability_rate = round(np.mean(list(expobj.StimSuccessRate_SLMtargets.values())), 2)
mean_response_dfstdf = round(np.mean(list(expobj.responses_SLMtargets.values())), 2)
mean_response_dff = np.nan

prep_trial = '%s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'])
stim_setup = '32 cells x 1 groups; 5mW per cell, 100ms stim (prot. #1)'
# stim_setup = '32 cells x 1 groups; 7mW per cell, 250ms stim (prot. #1b)'
if prep_trial not in list(allOpticalresults.slmtargets_stim_responses['prep_trial']):
    allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses.shape[0]+1] = [prep_trial] + ['-'] * (allOpticalresults.slmtargets_stim_responses.shape[1] - 1)
    allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'date'] = expobj.metainfo['date']
    allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'exptype'] = expobj.metainfo['exptype']
    allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'stim_setup'] = stim_setup
allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = mean_response_dfstdf
allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dFF all targets)'] = mean_response_dff
allOpticalresults.slmtargets_stim_responses.loc[allOpticalresults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>0.3 dF/stdF)'] = mean_reliability_rate


# %% save to superobject
allOpticalresults.save()

# %% SAVE THE UPDATED expobj OBJECT IN THE ORIGINAL PKL PATH TO USE NEXT

expobj.save_pkl(pkl_path=pkl_path)


print("\n COMPLETED RUNNING ALL OPTICAL PROCESSING PHOTOSTIM.")


#%%#####################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##### ------------------- processing for suite2p ROIs traces ###########################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# %% (quick) plot individual fluorescence traces - see InteractiveMatplotlibExample to make these plots interactively
# plot raw fluorescence traces
plt.figure(figsize=(50, 3))
for i in expobj.s2p_cell_targets:
    plt.plot(expobj.baseline_raw[expobj.cell_id.index(i)], linewidth=0.1)
plt.xlim(0, len(expobj.baseline_raw[0]))
plt.show()

# %% plotting the distribution of radius and aspect ratios - should this be running before the filtering step which is right below????????

radius_list = aoplot.plot_cell_radius_aspectr(expobj, expobj.stat, to_plot='radius')


# %% FILTER ALL SUITE2P_ROIs THAT ARE ACTIVE AT LEAST ONCE FOR >2.5*std

# pull out needed variables because numba doesn't work with custom classes (such as this all-optical class object)
# expobj.append_seizure_frames(bad_frames=None)
expobj.good_cells, events_loc_cells, flu_events_cells, stds = aoutils._good_cells(cell_ids=expobj.cell_id, raws=expobj.raw, photostim_frames=expobj.photostim_frames, std_thresh=2.5)
expobj.save()

# sort the stds dictionary in order of std
stds_sorted = {}
sorted_keys = sorted(stds, key=stds.get)  # [1, 3, 2]

for w in sorted_keys:
    stds_sorted[w] = stds[w]

# make a plot for the cells with high std. to make sure that they are not being unfairly excluded out

ls = list(stds_sorted.keys())[:5]
for cell in ls:
    aoplot.plot_flu_trace(expobj, to_plot='dff',  cell=cell, show=False)
    plt.scatter(x=events_loc_cells[cell], y=flu_events_cells[cell], s=0.5, c='darkgreen')
    plt.show()

# %% SAVE THE UPDATED expobj OBJECT IN THE ORIGINAL PKL PATH TO USE NEXT

expobj.save()


print("\n COMPLETED RUNNING ALL OPTICAL PROCESSING PHOTOSTIM.")



#%%#####################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##### ------------------- processing steps for SEIZURE TRIALS only!! ###################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

if 'post' in expobj.metainfo['exptype'] and '4ap' in expobj.metainfo['exptype']:
    print('\n MOVING ONTO POST-4AP SZ PROCESSING\n\n')
else:
    sys.exit()


expobj.MeanSeizureImages(baseline_tiff="/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_t-005/2020-12-18_t-005_Cycle00001_Ch3.tif",
                         frames_last=1000)

# counter = 0
# for i in avg_sub_l:
#     plt.imshow(i); plt.suptitle('%s' % counter); plt.show()
#     counter += 1

# MAKE AVG STIM IMAGES AROUND EACH PHOTOSTIM TIMINGS
expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=False, save_img=True, force_redo=True)

for i in range(len(expobj.avg_sub_l)):
    img = pj.rotate_img_avg(expobj.avg_sub_l[i], angle=90)
    # PCA decomposition of the avg_seizure images
    img_compressed = pj.pca_decomp_image(img, components=1, plot_quant=True)



#%%#####################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##### ------------------- processing steps for ALL OPTICAL PHOTOSTIM related stuff #####################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# %% collect ENTIRE TRIAL Flu - PHOTOSTIM targeted suite2p ROIs cells plotted individually entire Flu trace

expobj.raw_s2ptargets = [expobj.raw[expobj.cell_id.index(i)] for i in expobj.s2p_cell_targets if i in expobj.good_photostim_cells_all]
expobj.dff_s2ptargets = aoutils.normalize_dff(np.array(expobj.raw_s2ptargets))
# expobj.targets_dff_base = aoutils.normalize_dff_baseline(
#     arr=expobj.raw_df.loc[[str(x) for x in expobj.s2p_cell_targets]],
#     baseline_array=expobj.baseline_raw_df)
# plot_photostim_subplots(dff_array=SLMTargets_dff,
#                 title=(experiment + '%s responses of responsive cells' % len(expobj.good_photostim_cells_stim_responses_dFF)))



# %% Collect pre to post stim traces for PHOTOSTIM TARGETED CELLS, FILTER FOR GOOD PHOTOSTIM. TARGETED CELLS with responses above threshold = 1 std of the prestim std

expobj.pre_stim = 1*int(expobj.fps)  # specify pre-stim and post-stim periods of analysis and plotting
expobj.post_stim = 3*int(expobj.fps)

# function for gathering all good photostim cells who respond on average across all trials to the photostim
# note that the threshold for this is 1 * std of the prestim raw flu (fluorescence trace)
expobj.targets_dff, expobj.targets_dff_avg, expobj.targets_dfstdF, \
expobj.targets_dfstdF_avg, expobj.targets_stims_raw, expobj.targets_stims_raw_avg = \
    aoutils.get_s2ptargets_stim_traces(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                       post_stim=expobj.post_stim)

aoutils._good_photostim_cells(expobj=expobj, pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, dff_threshold=None)

# what does threshold value mean? add more descriptive print output for that

# %% Collect pre to post stim traces for NON-TARGETS

expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
    expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
    aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                            post_stim=expobj.post_stim)



#%% turn the important cell x time arrays into pandas dataframes

# raw Flu traces of all good cells
columns = [f'{num}' for num in range(expobj.curr_trial_frames[0], expobj.curr_trial_frames[1])]
index = [f'{num}' for num in expobj.good_cells]
idxs = [expobj.cell_id.index(cell) for cell in expobj.good_cells]
expobj.raw_df = pd.DataFrame(expobj.raw[idxs, :], columns=columns, index=index)


# # raw baseline Flu traces of all good cells
# columns = [f'{num}' for num in range(baseline_frames[0], baseline_frames[1])]
# index = [f'{num}' for num in expobj.good_cells]
# idxs = [expobj.cell_id.index(cell) for cell in expobj.good_cells]
# expobj.baseline_raw_df = pd.DataFrame(expobj.baseline_raw[idxs, :], columns=columns, index=index)



# %% 5) calculate dFF responses of all cells to photostimulation trials expobj

# non-targeted cells: calculate response of non-targeted cells in response to photostim. trials
# - make a pandas dataframe that contains the post-stim response of all cells at each stim timepoint
#   give group name 'non_targets' to the non-targetted cells, and the appropriate SLM group number to targetted cells


expobj.dff_responses_all_cells, expobj.risky_cells = aoutils.all_cell_responses_dff(expobj, normalize_to='pre-stim')

# calculate the avg response values for all cells across all stims
average_responses = np.mean(expobj.dff_responses_all_cells[expobj.dff_responses_all_cells.columns[1:]], axis=1)
responses = {'cell_id': [], 'group': [], 'Avg. dFF response': []}
for cell in expobj.good_cells:
    if cell not in expobj.risky_cells:
        if cell in expobj.s2p_cell_targets:
            responses['cell_id'].append(cell)
            responses['group'].append('photostim target')
            responses['Avg. dFF response'].append(average_responses[cell])
        else:
            responses['cell_id'].append(cell)
            responses['group'].append('non-target')
            responses['Avg. dFF response'].append(average_responses[cell])

expobj.average_responses_df = pd.DataFrame(responses)

print('\nThe avg. dF/F responses of photostim targets is: %s' % np.mean(
    expobj.average_responses_df[expobj.average_responses_df.group == 'photostim target'])[1])

print('\nThe avg. dF/F responses of non-targets is: %s' % np.mean(
    expobj.average_responses_df[expobj.average_responses_df.group == 'non-target'])[1])

# %% 5.1) calculate dF_stdF responses of all cells to photostimulation trials expobj

expobj.dfstdf_all_cells = aoutils.all_cell_responses_dFstdF(expobj)

# calculate the avg response values for all cells across all stims
average_responses_dfstdf = np.mean(expobj.dfstdf_all_cells[expobj.dfstdf_all_cells.columns[1:]], axis=1)
responses = {'cell_id': [], 'group': [], 'Avg. dF/stdF response': []}
for cell in expobj.good_cells:
    if cell not in expobj.risky_cells:
        if cell in expobj.s2p_cell_targets:
            responses['cell_id'].append(cell)
            responses['group'].append('photostim target')
            responses['Avg. dF/stdF response'].append(average_responses_dfstdf[cell])
        else:
            responses['cell_id'].append(cell)
            responses['group'].append('non-target')
            responses['Avg. dF/stdF response'].append(average_responses_dfstdf[cell])

expobj.average_responses_dfstdf = pd.DataFrame(responses)

print('\nThe avg. dF/stdF responses of photostim targets is: %s' % np.mean(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf.group == 'photostim target'])[1])

print('\nThe avg. dF/stdF responses of non-targets is: %s' % np.mean(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf.group == 'non-target'])[1])




# %% Convert stim responses TO NAN for cells inside the sz boundary at each of the stim timings

## for post 4ap trials with seizures only
for stim in expobj.dfstdf_all_cells.columns[1:]:
    if stim in expobj.cells_sz_stim.keys():
        cells_toko = expobj.cells_sz_stim[stim]
        expobj.dfstdf_all_cells.loc[cells_toko, str(stim)] = np.nan
        expobj.dff_responses_all_cells.loc[cells_toko, str(stim)] = np.nan

expobj.save()


#%% ---- END --------





########################################################################################################################


# # %%  EXTRA THINGS
# # there's a bunch of very high dFF responses of cells
# expobj.abnormal_high_responders = list(
#     expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500]['cell_id']);
# print(len(expobj.abnormal_high_responders))
# cell = expobj.abnormal_high_responders[0]
# x_ = list(expobj.dff_responses_all_cells.loc[cell][1:]);
# print(x_, '\nAverage:', np.mean(x_))
# [expobj.stim_start_frames[x] for x in range(len(x_)) if x_[x] > 6000]
# idx = expobj.cell_id.index(cell)
#
# # what is the mean baseline fluorescence value of these high responder cells?
# np.mean(expobj.raw[idx, 11281 + expobj.stim_duration_frames:11281 + 2 * expobj.stim_duration_frames])
#
# a = 0
# for trace in expobj.raw:
#     if np.mean(trace) <= 50:
#         a += 1
# print(a)
#
# #
# cell = expobj.abnormal_high_responders[0]
# mean_pre_list = []
# trace_dff_list = []
# trace_raw_list = []
# problem_stims = []
# for stim in expobj.stim_start_frames:
#     cell_idx = expobj.cell_id.index(cell)
#     trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.stim_duration_frames + expobj.post_stim];
#     trace_raw_list.append(trace)
#     mean_pre = np.mean(trace[0:expobj.pre_stim]);
#     mean_pre_list.append(mean_pre)
#     trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100;
#     trace_dff_list.append(trace_dff)
#     response = np.mean(trace_dff[
#                        expobj.pre_stim + expobj.stim_duration_frames:expobj.pre_stim + 3 * expobj.stim_duration_frames])
#     if response > 500:
#         problem_stims.append(list(expobj.stim_start_frames).index(stim))
#
# # del(trace_dff_list[10])
# for trace in trace_raw_list:
#     plt.plot(trace)
# plt.plot(np.mean(trace_raw_list, axis=0), edgecolor='black')
# # plt.ylim([-10,100])
# plt.show()
#
# # plt.plot(trace_raw_list[10]); plt.show()
#
# # %%
#
#
# aoplot.plot_flu_trace(expobj=expobj, x_lims=None, idx=idx, to_plot='dff')
#
# aoplot.plot_flu_trace(expobj=expobj, idx=idx, to_plot='raw', x_lims=[7000, 8000], figsize=(10, 5), linewidth=1)
