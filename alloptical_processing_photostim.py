## this file is for processing the photostim-experiment alloptical expobj object AFTER suite2p has been run
## the end of the script will update the expobj that was in the original pkl path

import sys
import os
import pickle
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils.funcs_pj as pj

###### IMPORT pkl file containing expobj
trial = 't-010'
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

cont_inue=True  # i know this is a rather very precarious thing here...


#%% prep for importing data from suite2p for this whole experiment
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
        _expobj = pickle.load(f)
        # import suite2p data
    total_frames_stitched += _expobj.n_frames
    if t == trial:
        expobj.curr_trial_frames = [total_frames_stitched - _expobj.n_frames, total_frames_stitched]
    if t in baseline_trials:
        baseline_frames[1] = total_frames_stitched



# suite2p processing on expobj; import suite2p data, flu, spks, cell coordinates and make s2p masks images stack

s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-08x/plane0'  # (most recent run for RL108 -- contains all trials including post4ap all optical trials)
# s2p_path = '/Users/prajayshah/Documents/data-to-process/2020-12-18/suite2p/alloptical-2p-pre-4ap-08x/plane0'
# flu, spks, stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)


# s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-18/suite2p/photostim-4ap_stitched/plane0'

# main function that imports suite2p data and adds attributes to the expobj
expobj.s2pProcessing(s2p_path=s2p_path, subset_frames=expobj.curr_trial_frames, subtract_neuropil=True,
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


# %% FILTER ALL CELLS THAT ARE ACTIVE AT LEAST ONCE FOR >2.5*std

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

# %% SAVE THE UPDATED expobj OBJECT IN THE ORIGINAL PKL PATH TO USE NEXT

# make the necessary Analysis saving subfolder as well
expobj.analysis_save_path = expobj.tiff_path[:21] + 'Analysis/' + expobj.tiff_path_dir[26:]
if os.path.exists(expobj.analysis_save_path):
    pass
elif os.path.exists(expobj.analysis_save_path[:-17]):
    os.mkdir(expobj.analysis_save_path)
elif os.path.exists(expobj.analysis_save_path[:-27]):
    os.mkdir(expobj.analysis_save_path[:-17])


expobj.save_pkl(pkl_path=pkl_path)




#%%#####################################################################################################################

##### ------------------- processing steps for SEIZURE TRIALS only!! ###################################################

########################################################################################################################

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


# MAKE SUBSELECTED TIFFS OF INVIDUAL SEIZURES BASED ON THEIR START AND STOP FRAMES
on_ = [expobj.stim_start_frames[0]]
on_.extend(expobj.stims_bf_sz)
expobj._subselect_sz_tiffs(onsets=on, offsets=expobj.stims_af_sz)


# %% classifying cells as in or out of the current seizure location in the FOV

# FRIST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv
if cont_inue:
    pass
else:
    sys.exit()

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
expobj.flip_stims = [328, 476, 624, 772, 921, 1069, 1217, 1365, 1514, 1662, 1810, 1958,
                     4775, 4923, 5071, 5219, 5368,
                     6702, 6850, 6998, 7295,
                     8777, 8925, 9074, 9370, 9518,
                     12038, 12186, 12335, 12483, 12631, 13669]  # specify here the stims where the flip=False leads to incorrect assignment

print('working on classifying cells for stims start frames:')
expobj.cells_sz_stim = {}
for on, off in zip(on_, expobj.stims_af_sz):
    stims_of_interest = [stim for stim in expobj.stim_start_frames if on <= stim <= off]
    print('|-', stims_of_interest)

    for stim in stims_of_interest:
        sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path, trial, stim)
        if stim in expobj.flip_stims:
            flip = True
        else:
            flip = False

        in_sz = expobj.classify_cells_sz(sz_border_path, to_plot=True, title='%s' % stim, flip=flip)
        expobj.cells_sz_stim[stim] = in_sz  # for each stim, there will be a list of cells that will be classified as in seizure or out of seizure
expobj.save()


#%%#####################################################################################################################

##### ------------------- processing steps for ALL OPTICAL PHOTOSTIM related stuff #####################################

########################################################################################################################

# Collect pre to post stim traces for PHOTOSTIM TARGETED CELLS, FILTER FOR GOOD PHOTOSTIM. TARGETED CELLS with responses above threshold = 1 std of the prestim std

expobj.pre_stim = 15  # specify pre-stim and post-stim periods of analysis and plotting
expobj.post_stim = 150

# function for gathering all good photostim cells who respond on average across all trials to the photostim
# note that the threshold for this is 1 * std of the prestim raw flu (fluorescence trace)
expobj.targets_dff, expobj.targets_dff_avg, expobj.targets_dfstdF, \
    expobj.targets_dfstdF_avg, expobj.targets_raw, expobj.targets_raw_avg = \
    aoutils.get_targets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                         post_stim=expobj.post_stim)

aoutils._good_photostim_cells(expobj=expobj, pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, dff_threshold=None)

# what does threshold value mean? add more descriptive print output for that

# Collect pre to post stim traces for NON-TARGETS

expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
    expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
    aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                            post_stim=expobj.post_stim)




# %% (full) plot individual cell's flu or dFF trace, with photostim. timings for that cell

# plot flu trace of selected cell with the std threshold
# aoplot.plot_flu_trace(expobj=expobj, cell=0, x_lims=None, to_plot='dff')


#%% turn important cell x time arrays into pandas dataframes

# raw Flu traces of all good cells
columns = [f'{num}' for num in range(expobj.curr_trial_frames[0], expobj.curr_trial_frames[1])]
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

print('\nThe avg. dF/F responses of photostim targets is: %s' % np.mean(
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

print('\nThe avg. dF/stdF responses of photostim targets is: %s' % np.mean(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf.group == 'photostim target'])[1])




# %% Convert stim responses TO NAN for cells inside the sz boundary at each of the stim timings


for stim in expobj.dfstdf_all_cells.columns[1:]:
    if stim in expobj.cells_sz_stim.keys():
        cells_toko = expobj.cells_sz_stim[stim]
        expobj.dfstdf_all_cells.loc[cells_toko, str(stim)] = np.nan
        expobj.dff_all_cells.loc[cells_toko, str(stim)] = np.nan

expobj.save()






#%% ---- END --------





########################################################################################################################


# # %%  EXTRA THINGS
# # there's a bunch of very high dFF responses of cells
# expobj.abnormal_high_responders = list(
#     expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500]['cell_id']);
# print(len(expobj.abnormal_high_responders))
# cell = expobj.abnormal_high_responders[0]
# x_ = list(expobj.dff_all_cells.loc[cell][1:]);
# print(x_, '\nAverage:', np.mean(x_))
# [expobj.stim_start_frames[x] for x in range(len(x_)) if x_[x] > 6000]
# idx = expobj.cell_id.index(cell)
#
# # what is the mean baseline fluorescence value of these high responder cells?
# np.mean(expobj.raw[idx, 11281 + expobj.duration_frames:11281 + 2 * expobj.duration_frames])
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
#     trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim];
#     trace_raw_list.append(trace)
#     mean_pre = np.mean(trace[0:expobj.pre_stim]);
#     mean_pre_list.append(mean_pre)
#     trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100;
#     trace_dff_list.append(trace_dff)
#     response = np.mean(trace_dff[
#                        expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 3 * expobj.duration_frames])
#     if response > 500:
#         problem_stims.append(list(expobj.stim_start_frames).index(stim))
#
# # del(trace_dff_list[10])
# for trace in trace_raw_list:
#     plt.plot(trace)
# plt.plot(np.mean(trace_raw_list, axis=0), color='black')
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
