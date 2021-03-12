## this file is for processing the photostim-experiment alloptical expobj object AFTER suite2p has been run
## the end of the script will update the expobj that was in the original pkl path

import pickle
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% ###### IMPORT pkl file containing expobj
trial = 't-013'
experiment = 'RL108: photostim-post4ap-%s' % trial
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)


# determine which frames to retrieve from the overall total s2p output
trials = ['t-005', 't-006', 't-008', 't-009', 't-010', 't-011', 't-012',
          't-013']  # specify all trials that were used in the suite2p run
baseline_trials = ['t-005', 't-006', 't-008']  # specify which trials to use as spont baseline
# note ^^^ this only works currently when the spont baseline trials all come first, and also back to back
total_frames_stitched = 0
curr_trial_frames = None
baseline_frames = [0, 0]
for t in trials:
    pkl_path_2 = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, t, date, t)
    with open(pkl_path_2, 'rb') as f:
        expobj = pickle.load(f)
        # import suite2p data
    total_frames_stitched += expobj.n_frames
    if t == trial:
        curr_trial_frames = [total_frames_stitched - expobj.n_frames, total_frames_stitched]
    if t in baseline_trials:
        baseline_frames[1] = total_frames_stitched

with open(pkl_path, 'rb') as f:
    expobj = pickle.load(f)

# %% suite2p processing on expobj; import suite2p data, flu, spks, cell coordinates and make s2p masks images stack

s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-08x/plane0'  # (most recent run for RL108 -- contains all trials including post4ap all optical trials)
# s2p_path = '/Users/prajayshah/Documents/data-to-process/2020-12-18/suite2p/alloptical-2p-pre-4ap-08x/plane0'
# flu, spks, stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)


# s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-18/suite2p/photostim-4ap_stitched/plane0'

# main function that imports suite2p data and adds attributes to the expobj
expobj.s2pProcessing(s2p_path=s2p_path, subset_frames=curr_trial_frames, subtract_neuropil=True,
                     baseline_frames=baseline_frames)
# if needed for pkl expobj generated from older versions of Vape
expobj.target_coords_all = expobj.target_coords
expobj.s2p_targets()

# expobj.target_coords_all = expobj.target_coords

# flu, expobj.spks, expobj.stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)

aoutils.s2pMaskStack(obj=expobj, pkl_list=[pkl_path], s2p_path=s2p_path,
                     parent_folder='/home/pshah/mnt/qnap/Analysis/2020-12-18/')

# %% (quick) plot individual fluorescence traces - see InteractiveMatplotlibExample to make these plots interactively
# plot raw fluorescence traces
plt.figure(figsize=(50, 3))
for i in expobj.s2p_cell_targets:
    plt.plot(expobj.baseline_raw[expobj.cell_id.index(i)], linewidth=0.1)
plt.xlim(0, len(expobj.baseline_raw[0]))
plt.show()

# %% plotting the distribution of radius and aspect ratios - should this be running before the filtering step which is right below????????

to_plot = aoplot.plot_cell_radius_aspectr(expobj, expobj.stat, to_plot='radius')
a = [i for i in to_plot if i > 6]
id = to_plot.index(min(a))
# expobj.good_cells[id]


# %% FILTER CELLS THAT ARE ACTIVE AT LEAST ONCE FOR >2.5*std

# pull out needed variables because numba doesn't work with custom classes (such as this all-optical class object)
cell_ids = expobj.cell_id
raws = expobj.raw
# expobj.append_seizure_frames(bad_frames=None)
photostim_frames = expobj.photostim_frames
radiuses = expobj.radius

# initial quick run to allow numba to compile the function - not sure if this is actually creating time savings
_ = aoutils._good_cells(cell_ids=cell_ids[:3], raws=raws, photostim_frames=expobj.photostim_frames, radiuses=radiuses,
                        std_thresh=2, min_radius_pix=2.5, max_radius_pix=8.5)
expobj.good_cells = aoutils._good_cells(cell_ids=cell_ids, raws=raws, photostim_frames=expobj.photostim_frames,
                                        radiuses=radiuses,
                                        std_thresh=2, min_radius_pix=2.5, max_radius_pix=8.5)

# %% filter for GOOD PHOTOSTIM. TARGETED CELLS with responses above threshold = 1 std of the prestim std

expobj.pre_stim = 15  # specify pre-stim and post-stim periods of analysis and plotting
expobj.post_stim = 150

# function for gathering all good photostim cells who respond on average across all trials to the photostim
# note that the threshold for this is 1 * std of the prestim raw flu (fluorescence trace)
expobj.targets_dff, expobj.targets_dff_avg, expobj.targets_dfstdF, \
    expobj.targets_dfstdF_avg, expobj.targets_raw, expobj.targets_raw_avg = \
    aoutils.get_targets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                         post_stim=expobj.post_stim)

aoutils._good_photostim_cells(expobj=expobj, pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, dff_threshold=None)

# TODO what does threshold value mean? add more descriptive print output for that

# %% (full) plot individual cell's flu or dFF trace, with photostim. timings for that cell

# plot flu trace of selected cell with the std threshold
aoplot.plot_flu_trace(expobj=expobj, cell=0, x_lims=None, to_plot='dff')


#%% turn important cell x time arrays into pandas dataframes

# raw Flu traces of all good cells
columns = [f'{num}' for num in range(curr_trial_frames[0], curr_trial_frames[1])]
index = [f'{num}' for num in expobj.good_cells]
idxs = [expobj.cell_id.index(cell) for cell in expobj.good_cells]
expobj.raw_df = pd.DataFrame(expobj.raw[idxs, :], columns=columns, index=index)


# raw baseline Flu traces of all good cells
columns = [f'{num}' for num in range(baseline_frames[0], baseline_frames[1])]
index = [f'{num}' for num in expobj.good_cells]
idxs = [expobj.cell_id.index(cell) for cell in expobj.good_cells]
expobj.baseline_raw_df = pd.DataFrame(expobj.baseline_raw[idxs, :], columns=columns, index=index)



# %% calculate dFF responses of all cells to photostimulation trials expobj

# non-targeted cells: calculate response of non-targeted cells in response to photostim. trials
# - make a pandas dataframe that contains the post-stim response of all cells at each stim timepoint
#   give group name 'non_targets' to the non-targetted cells, and the appropriate SLM group number to targetted cells


expobj.dff_all_cells = aoutils.all_cell_responses_dff(expobj, normalize_to='pre-stim')

# calculate the avg response values for all cells across all stims
average_responses = np.mean(expobj.dff_all_cells[expobj.dff_all_cells.columns[1:]], axis=1)
responses = {'cell_id': [], 'group': [], 'Avg. dFF response': []}
for cell in expobj.good_cells:
    if cell in expobj.s2p_cell_targets:
        responses['cell_id'].append(cell)
        responses['group'].append('photostim target')
        responses['Avg. dFF response'].append(average_responses[cell])
    else:
        responses['cell_id'].append(cell)
        responses['group'].append('non-target')
        responses['Avg. dFF response'].append(average_responses[cell])

expobj.average_responses_df = pd.DataFrame(responses)

print('The avg. responses of photostim targets is: %s' % np.mean(
    expobj.average_responses_df[expobj.average_responses_df.group == 'photostim target'])[1])

# %% calculate dF_stdF responses of all cells to photostimulation trials expobj

expobj.dfstdf_all_cells = aoutils.all_cell_responses_dFstdF(expobj)

# calculate the avg response values for all cells across all stims
average_responses_dfstdf = np.mean(expobj.dfstdf_all_cells[expobj.dfstdf_all_cells.columns[1:]], axis=1)
responses = {'cell_id': [], 'group': [], 'Avg. dF/stdF response': []}
for cell in expobj.good_cells:
    if cell in expobj.s2p_cell_targets:
        responses['cell_id'].append(cell)
        responses['group'].append('photostim target')
        responses['Avg. dF/stdF response'].append(average_responses_dfstdf[cell])
    else:
        responses['cell_id'].append(cell)
        responses['group'].append('non-target')
        responses['Avg. dF/stdF response'].append(average_responses_dfstdf[cell])

expobj.average_responses_dfstdf = pd.DataFrame(responses)

print('The avg. responses of photostim targets is: %s' % np.mean(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf.group == 'photostim target'])[1])


# %% SAVE THE UPDATED expobj OBJECT IN THE ORIGINAL PKL PATH TO USE NEXT

expobj.save_pkl(pkl_path=pkl_path)





# %%  EXTRA THINGS
# there's a bunch of very high dFF responses of cells
expobj.abnormal_high_responders = list(
    expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500]['cell_id']);
print(len(expobj.abnormal_high_responders))
cell = expobj.abnormal_high_responders[0]
x_ = list(expobj.dff_all_cells.loc[cell][1:]);
print(x_, '\nAverage:', np.mean(x_))
[expobj.stim_start_frames[x] for x in range(len(x_)) if x_[x] > 6000]
idx = expobj.cell_id.index(cell)

# what is the mean baseline fluorescence value of these high responder cells?
np.mean(expobj.raw[idx, 11281 + expobj.duration_frames:11281 + 2 * expobj.duration_frames])

a = 0
for trace in expobj.raw:
    if np.mean(trace) <= 50:
        a += 1
print(a)

#
cell = expobj.abnormal_high_responders[0]
mean_pre_list = []
trace_dff_list = []
trace_raw_list = []
problem_stims = []
for stim in expobj.stim_start_frames:
    cell_idx = expobj.cell_id.index(cell)
    trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim];
    trace_raw_list.append(trace)
    mean_pre = np.mean(trace[0:expobj.pre_stim]);
    mean_pre_list.append(mean_pre)
    trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100;
    trace_dff_list.append(trace_dff)
    response = np.mean(trace_dff[
                       expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 3 * expobj.duration_frames])
    if response > 500:
        problem_stims.append(list(expobj.stim_start_frames).index(stim))

# del(trace_dff_list[10])
for trace in trace_raw_list:
    plt.plot(trace)
plt.plot(np.mean(trace_raw_list, axis=0), color='black')
# plt.ylim([-10,100])
plt.show()

# plt.plot(trace_raw_list[10]); plt.show()

# %%


aoplot.plot_flu_trace(expobj=expobj, x_lims=None, idx=idx, to_plot='dff')

aoplot.plot_flu_trace(expobj=expobj, idx=idx, to_plot='raw', x_lims=[7000, 8000], figsize=(10, 5), linewidth=1)
