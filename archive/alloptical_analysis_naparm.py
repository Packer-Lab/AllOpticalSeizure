# #%% start on remote machine
#
# # imports general modules, runs ipython magic commands
# # change path in this notebook to point to repo locally
# # n.b. sometimes need to run this cell twice to init the plotting paramters
# import sys; sys.path.append('/home/pshah/Documents/code/Vape/utils/')
# import alloptical_utils_pj as ao
# import numpy as np
# import utils_funcs as uf
# import matplotlib.pyplot as plt
# print(sys.path)
# import funcs_pj as pj
# import pickle
# import pandas as pd
#
# plt.rcParams['figure.figsize'] = [20.0, 3.0]
#
# trial = 't-017'

#%% start on local machine
import alloptical_utils_pj as ao
import numpy as np
import pandas as pd
import utils.utils_funcs as uf #from Vape
import matplotlib.pyplot as plt
import pickle
import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/Vape')
import math
from numba import jit, njit

trial = 't-017'
experiment = 'J063: find_resp-4ap-t017'
# trials = ['t-019', 't-020']

# save_path = "/home/pshah/mnt/qnap/Data/2020-03-18/J063/2020-03-18_J063_%s/2020-03-18_%s.pkl" % (trial, trial)
# save_path = '/Users/prajayshah/Documents/data-to-process/2020-03-18/2020-03-18_t-019_t-020.pkl'
# save_path = "/Volumes/Extreme SSD/oxford-data/2020-03-19/2020-03-19_%s.pkl" % trial
pkl_path = "/Volumes/Extreme SSD/oxford-data/2020-03-18/2020-03-18_%s.pkl" % trial

#%% IMPORT pkl file containing exp_obj
with open(pkl_path, 'rb') as f:
    exp_obj = pickle.load(f)

#%% suite2p processing  on exp_obj
# s2p_path = '/home/pshah/mnt/qnap/Data/2020-03-18/J063/suite2p/find_resp-4ap-t017/plane0'
# s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-19/suite2p/find_resp-baseline-t002/plane0'
s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-18/suite2p/find_resp-4ap-t017/plane0'
exp_obj.s2pProcessing(s2p_path=s2p_path)
exp_obj.find_s2p_targets_naparm()

flu, exp_obj.spks, exp_obj.stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)

# plot average of all FOV response

def plot_overall_flu(exp_obj=exp_obj):
    dff = ao.normalize_dff(exp_obj.raw)
    dff_mean = np.mean(dff, axis=0)

    fig, ax = plt.subplots(figsize=(20, 4))
    color = 'black'
    ax.plot(dff_mean, color=color, alpha=0.4, linewidth=0.5)
    ax.tick_params(axis='y', labelcolor=color)
    if type(exp_obj.seizure_frames) == list:
        ax.axvline(x=exp_obj.seizure_frames[0], color='black')
        ax.axvline(x=exp_obj.seizure_frames[-1], color='black')
    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = int(round(item/exp_obj.fps))
    ax.set_xticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(experiment + ' - mean population Ca2+ signal of FOV', horizontalalignment='center', verticalalignment='top', pad=20, fontsize=15)
    plt.show()
plot_overall_flu(exp_obj)


#%% FILTERING SUITE2p CELLS BASED ON THEIR RESPONSE
# filter cells that are active at least once for >2.5*std across 4 frames (defined as 1 Ca event)
import time
start = time.time()

cell_ids = exp_obj.cell_id
raws = exp_obj.raw
photostim_frames = exp_obj.photostim_frames
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
        raw = raws[i]
        raw_ = np.delete(raw, exp_obj.photostim_frames, None)
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




exp_obj.good_cells = _good_cells(cell_ids=cell_ids, raws=raws, photostim_frames=photostim_frames, radiuses=radiuses,
                                 min_radius_pix=3.5, max_radius_pix=8.5)
end = time.time()
print((end-start))


@njit
def _good_cells(cell_ids, raws, photostim_frames, radiuses, min_radius_pix, max_radius_pix):
    good_cells = []
    for i in range(len(cell_ids)):
        print(i, " out of ", len(cell_ids), " cells")
        raw = raws[i]
        raw_ = np.delete(raw, photostim_frames)
        raw_dff = ao.normalize_dff_jit(raw_)
        std_ = raw_dff.std()

        a = []
        y = []
        for j in np.arange(len(raw_dff), step=4):
            avg = np.mean(raw_dff[j:j+4])
            if avg > np.mean(raw_dff)+2.5*std_: # if the avg of 4 frames is greater than the threshold then save the result
                a.append(j)
                y.append(avg)

        radius = radiuses[i]

        if len(a) > 0 and radius > min_radius_pix and radius < max_radius_pix:
            good_cells.append(cell_ids[i])
    print('# of good cells found: ', len(good_cells), ' (out of ', len(cell_ids), ' ROIs)')
    return good_cells

#%% filter for good photostim. targeted cell with responses above threshold
# specify pre-stim and post-stim periods of analysis and plotting
pre_stim = 10
post_stim = 100

# function for gathering all good photostim cells who respond reliably to photostim
def _good_photostim_cells(exp_obj, std_thresh=1, dff_threshold=None, pre_stim=10, post_stim=200):
    '''
    make sure to specify std threshold to use for filtering
    the pre-stim and post-stim args specify which pre-stim and post-stim frames to consider for filtering
    '''
    exp_obj.good_naparm_cells = []
    exp_obj.good_naparm_cells_responses = []
    exp_obj.good_photostim_cells_stim_responses_dF_stdF = []
    exp_obj.good_photostim_cells_stim_responses_dFF = []
    total = 0 # use to tally up how many cells across all groups are filtered in
    total_considered = 0 # use to tally up how many cells were looked at for their photostim response.
    for group in range(len(exp_obj.s2p_targets_naparm)):
        print('\nGroup %s' % group)
        stim_timings = exp_obj.stim_start_frames[0][group::exp_obj.n_groups]
        title = 'SLM naparm Group #%s' % group
        # is_target = exp_obj.s2p_targets_naparm[group]
        targeted_cells = [cell for cell in exp_obj.s2p_targets_naparm[group] if cell in exp_obj.good_cells]

        # collect photostim timed average dff traces of photostim targets
        targets_dff = []
        pre_stim = pre_stim
        post_stim = post_stim
        for cell in targeted_cells:
            # print('considering cell # %s' % cell)
            if cell in exp_obj.cell_id:
                cell_idx = exp_obj.cell_id.index(cell)
                flu = [exp_obj.raw[cell_idx][stim - pre_stim:stim + post_stim] for stim in stim_timings
                       if stim not in exp_obj.seizure_frames]

                flu_dff = []
                for trace in flu:
                    mean = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean) / mean) * 100
                    flu_dff.append(trace_dff)

                targets_dff.append(np.mean(flu_dff, axis=0))

        # FILTER CELLS WHERE PHOTOSTIMULATED TARGETS FIRE > 10*std ABOVE PRE-STIM
        good_photostim_responses = {}
        good_photostim_cells = []
        good_targets_dF_stdF = []
        good_targets_dff = []
        std_thresh = std_thresh
        for cell in targeted_cells:
            trace = targets_dff[
                targeted_cells.index(cell)]  # trace = averaged dff trace across all photostims. for this cell
            pre_stim_trace = trace[:pre_stim]
            # post_stim_trace = trace[pre_stim_sec + exp_obj.stim_duration_frames:post_stim_sec]
            mean_pre = np.mean(pre_stim_trace)
            std_pre = np.std(pre_stim_trace)
            # mean_post = np.mean(post_stim_trace[:10])
            dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
            # response = np.mean(dF_stdF[pre_stim_sec + exp_obj.stim_duration_frames:pre_stim_sec + 3*exp_obj.stim_duration_frames])
            response = np.mean(trace[pre_stim + exp_obj.stim_duration_frames:pre_stim + 3 * exp_obj.stim_duration_frames]) # calculate the dF over pre-stim mean F response within the response window
            if dff_threshold is None:
                thresh_ = mean_pre + std_thresh * std_pre
            else:
                thresh_ = dff_threshold
            if response > thresh_:  # test if the response passes threshold
                good_photostim_responses[cell] = response
                good_photostim_cells.append(cell)
                good_targets_dF_stdF.append(dF_stdF)
                good_targets_dff.append(trace)
                print('Cell #%s - dFF post-stim: %s (threshold value = %s)' % (cell, response, thresh_))

        exp_obj.good_naparm_cells.append(good_photostim_cells)
        exp_obj.good_naparm_cells_responses.append(good_photostim_responses)
        exp_obj.good_photostim_cells_stim_responses_dF_stdF.append(good_targets_dF_stdF)
        exp_obj.good_photostim_cells_stim_responses_dFF.append(good_targets_dff)

        print('%s cells filtered out of %s s2p target cells' % (len(good_photostim_cells), len(targeted_cells)))
        total += len(good_photostim_cells)
        total_considered += len(targeted_cells)

    exp_obj.good_naparm_cells_all = [y for x in exp_obj.good_naparm_cells for y in x]
    print('\nTotal number of good photostim responsive cells found: %s (out of %s)' % (total, total_considered))
_good_photostim_cells(exp_obj, pre_stim=pre_stim, post_stim=post_stim, dff_threshold=20)


#%% image of all target coords - for demonstration purposes
# - add random colored circles around the target coords that were good photostim targets (same edgecolor for cells in the same SLM group)

arr = np.zeros((exp_obj.frame_x, exp_obj.frame_x), dtype='uint8')
for group in range(exp_obj.n_groups):
    target_areas_ = exp_obj.target_areas[group]
    coords = [(y,x) for cell in target_areas_ for (y,x) in cell]
    for coord in coords:
        arr[coord[0], coord[1]] = 100

import random
plt.imshow(arr)
# add plt.scatter circles over good stim. cells
for group in range(exp_obj.n_groups):
    cell_ids = exp_obj.good_naparm_cells[group]
    if len(cell_ids) > 0:
        rgb = (random.random(), random.random(), random.random())
        for cell in cell_ids:
            idx = exp_obj.cell_id.index(cell)
            x = exp_obj.stat[idx]['med'][1]
            y = exp_obj.stat[idx]['med'][0]
            plt.scatter(x=x, y=y, edgecolors=rgb, facecolors=rgb, linestyle='--', linewidths=1)
plt.suptitle((experiment + '- all targets fired during classic_naparm - responsive cells highlighted'), fontsize=10, y=1)
plt.show()



#%% PLOTTING individual photostim. targeted cells' photostim. response traces
# plot avg photostim response of all responsive cells (i.e. good cells from above filtering steps)
r_array = [j for i in exp_obj.good_photostim_cells_stim_responses_dFF for j in i]; y_label='% dFF (normalized to prestim period)'

# make plots of photostim targeted trials
def plot_photostim_avg(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None,
                       x_label='Time post-stim. (seconds)', y_label=None):
    len_ = len(dff_array)
    flu_avg = np.median(dff_array, axis=0)
    std = np.std(dff_array, axis=0)
    ci = 1.960 * (std / np.sqrt(len_))  # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
    x = list(range(-pre_stim, post_stim))
    y = flu_avg

    fig, ax = plt.subplots()
    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)  # plot confidence interval
    ax.axvspan(0, stim_duration, alpha=0.2, color='green')
    if post_stim > int(exp_obj.fps):
        for i in range(int(exp_obj.fps), post_stim, int(exp_obj.fps)): # make sure to change this based on the fps of the naparm trial imaging
            ax.axvspan(i, i+stim_duration, color='gray', alpha=0.1)
    ax.plot(x, y)
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    ax.set_title((title+' - %s' % len_+' cells'), horizontalalignment='center', verticalalignment='top', pad=20, fontsize=10)

    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = round(item/exp_obj.fps, 1)
    ax.set_xticklabels(labels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()
def plot_photostim_(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None,
                    x_label=None, y_label=None):
    len_ = len(dff_array)
    x = list(range(-pre_stim, post_stim))
    fig, ax = plt.subplots()
    ax.axvspan(0, stim_duration, alpha=0.2, color='green')
    for cell_trace in dff_array:
        ax.plot(x, cell_trace, linewidth=2, alpha=0.5)
    if post_stim > int(exp_obj.fps):
        for i in range(int(exp_obj.fps), post_stim,
                       int(exp_obj.fps)):  # make sure to change this based on the fps of the naparm trial imaging
            ax.axvspan(i, i + stim_duration, color='gray', alpha=0.1)
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    ax.set_title((title + ' - %s' % len_ + ' cells'), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()

plot_photostim_avg(dff_array=r_array, stim_duration=exp_obj.stim_duration_frames, pre_stim=pre_stim, post_stim=post_stim, title=(experiment + ' - Avg photostim. response of all responsive cells'), y_label=y_label)
plot_photostim_(dff_array=r_array, stim_duration=exp_obj.stim_duration_frames, pre_stim=pre_stim, post_stim=post_stim, title=(experiment + ' - Photostim. response of all responsive cells'), y_label=y_label)

def plot_single_stim_trial(cell, stim_number, title, stim_frame=None, pre_stim=pre_stim, post_stim=post_stim, x_label=None, y_label=None):
    '''plot a single cell's photostim response on a single stim trial, along with the 1*std pre-stim line'''
    if stim_frame is None:
        group = [exp_obj.good_naparm_cells.index(i) for i in exp_obj.good_naparm_cells if cell in i][0]
        stim_frame = exp_obj.stim_start_frames[0][group::exp_obj.n_groups][stim_number]

    cell_idx = exp_obj.cell_id.index(cell)
    trace = exp_obj.raw[cell_idx][stim_frame-pre_stim:stim_frame + exp_obj.stim_duration_frames + post_stim]
    std_pre = np.std(trace[0:pre_stim])
    pre_stim_mean = np.mean(trace[0:pre_stim])
    response = np.mean(trace[pre_stim+exp_obj.stim_duration_frames: pre_stim + 1 + 2 * exp_obj.stim_duration_frames])

    # make plot
    fig, ax = plt.subplots()
    ax.plot(trace)
    x = range(pre_stim + exp_obj.stim_duration_frames, pre_stim + 1 + 2 * exp_obj.stim_duration_frames)
    y = [response]*len(x)
    ax.scatter(x=x, y=y, marker='_', alpha=0.4, color='purple')
    ax.axhline(y=pre_stim_mean+2*std_pre, linestyle='-', color = 'purple', alpha=0.4)
    ax.axvspan(pre_stim, pre_stim + exp_obj.stim_duration_frames, color='green', alpha=0.1)
    ax.axvspan(pre_stim + exp_obj.stim_duration_frames, pre_stim + 1 + 2 * exp_obj.stim_duration_frames, color='gray', alpha=0.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title((title + ' - Cell %s, stim. # %s' % (cell, stim_number)), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    plt.show()
plot_single_stim_trial(cell=53, stim_number=3, stim_frame=None, pre_stim=pre_stim, post_stim=100, title=experiment)




#%% Plotting of quantified responses of PHOTOSTIM TARGETED CELLS by individual stim. trials
# uses a metric to measure the max. response of the targeted cell in the post-stim period
# creates a pandas dataframe to handle collecting this response data from the targeted cells
    # rows: cells
    # columns: group #, all stim_frame_starts (cells not targeted on that stim will have nan as response metric)

pre_stim = 10
post_stim = 100

# create empty dataframe with the correct rows and columns
d = {}
#d['cells'] = exp_obj.targeted_cells_all
d['groups'] = [int(exp_obj.good_naparm_cells.index(x)) for x in exp_obj.good_naparm_cells for y in x]
for stim in exp_obj.stim_start_frames[0]:
    d['%s' % stim] = [None]*len(exp_obj.good_naparm_cells_all)
df = pd.DataFrame(d, index=exp_obj.good_naparm_cells_all)
# population dataframe
for group in df['groups'].unique():
    stims = exp_obj.stim_start_frames[0][group::exp_obj.n_groups]
    cells = exp_obj.good_naparm_cells[group]
    for stim in stims:
        for cell in cells:
            cell_idx = exp_obj.cell_id.index(cell)
            trace = exp_obj.raw[cell_idx][stim - pre_stim:stim + exp_obj.stim_duration_frames + post_stim]
            mean = np.mean(trace[0:pre_stim])
            trace_dff = ((trace - mean) / abs(mean)) * 100
            std_pre = np.std(trace[0:pre_stim])
            # response = np.mean(trace_dff[pre_stim_sec + exp_obj.stim_duration_frames:pre_stim_sec + 3*exp_obj.stim_duration_frames])
            dF_stdF = (trace - mean) / std_pre  # make dF divided by std of pre-stim F trace
            # response = np.mean(dF_stdF[pre_stim_sec + exp_obj.stim_duration_frames:pre_stim_sec + 1 + 2 * exp_obj.stim_duration_frames])
            response = np.mean(trace_dff[pre_stim + exp_obj.stim_duration_frames:pre_stim + 1 + 2 * exp_obj.stim_duration_frames])
            df.at[cell, '%s' % stim] = response

exp_obj.cell_responses_dff = df # save responses to exp_obj

# average stim response of all cells in that stim group at each stim. timepoint
def plot_stim_responses_stimgroups(title=''):
    '''plot average stim response of each cell at each stim. timepoint, along with the average dFF across all cells at each frame'''
    dff = ao.normalize_dff(exp_obj.raw)
    dff_mean = np.mean(dff, axis=0)

    # # use for troubleshooting gathering responses from pandas df
    # keys = df.keys()[1:]
    # response = [df["%s" % stim].mean() for stim in df.keys()[1:]]
    # x = [int(i) for i in keys[pd.notna(response)]]
    # y = [a for a in response if str(a) != 'nan']
    # #

    fig, ax1 = plt.subplots(figsize=(20,3))
    color = 'gray'
    ax1.plot(dff_mean, color=color, alpha=0.4, linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    if type(exp_obj.seizure_frames) == list:
        ax1.axvline(x=exp_obj.seizure_frames[0], color='black')
        ax1.axvline(x=exp_obj.seizure_frames[-1], color='black')

    ax2 = ax1.twinx()
    color = 'black'
    for stim_group in range(exp_obj.n_groups):
        stims = exp_obj.stim_start_frames[0][stim_group::exp_obj.n_groups]
        response = [df["%s" % stim].mean() for stim in stims]
        x = [int(i) for i in stims[pd.notna(response)]]
        y = [a for a in response if str(a) != 'nan']
        ax2.scatter(x, y, s=5)
        ax2.plot(x, y, linewidth=0.5)

    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title((title), horizontalalignment='center', verticalalignment='top', pad=20, fontsize=15)
    fig.show()
plot_stim_responses_stimgroups(title=(experiment + 'average stim. response - by stims groups - %s groups' % exp_obj.n_groups))

# individual cell responses at each stim. timepoint - with line connecting each stim for each cell, but turned off
def plot_stim_responses_cells(title='', y_label=y_label, y_axis_range=[]):
    '''plot average stim response of each cell at each stim. timepoint, along with the average dFF across all cells at each frame'''
    dff = ao.normalize_dff(exp_obj.raw)
    dff_mean = np.mean(dff, axis=0)

    # keys = df.keys()[1:]
    # response = [df["%s" % stim].mean() for stim in df.keys()[1:]]
    # x = [int(i) for i in keys[pd.notna(response)]]
    # y = [a for a in response if str(a) != 'nan']


    fig, ax1 = plt.subplots(figsize=(20,3))
    color = 'gray'
    ax1.plot(dff_mean, color=color, alpha=0.4, linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    if type(exp_obj.seizure_frames) == list:
        ax1.axvline(x=exp_obj.seizure_frames[0], color='black')
        ax1.axvline(x=exp_obj.seizure_frames[-1], color='black')

    ax2 = ax1.twinx()
    color = 'black'
    x_all = []
    y_all = []
    for cell in exp_obj.good_naparm_cells_all:
        stim_group = df.loc[cell, 'groups']
        x = exp_obj.stim_start_frames[0][stim_group::exp_obj.n_groups]
        y = [df.loc[cell, '%s' % i] for i in x]
        ax2.scatter(x,y,s=5)
        # ax2.plot(x, y, linewidth=0.5)
        x_all.extend(x)
        y_all.extend(y)

    # ax2.scatter(x_all, y_all, s=5)
    #ax2.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x))) # plot line of best fit through scatter plot  # this best fit line is linear so not useful

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel(y_label)
    ax2.set_ylim(y_axis_range)

    ax1.set_title(title, horizontalalignment='center', verticalalignment='top', pad=20, fontsize=15)

    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # change x axis ticks to seconds
    labels = [item for item in ax1.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = int(round(item/exp_obj.fps))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Time (seconds)')

    fig.show()
plot_stim_responses_cells(y_axis_range=[0,1000], y_label='df/pref',
                          title=(experiment + 'stim. responses - by individual cells - %s cells' % len(exp_obj.good_naparm_cells_all)))

# average stim response of all cells in that stim group at each stim. timepoint - normalized
def plot_stim_responses_norm_cells(title='', norm_start=0, norm_end=exp_obj.n_frames, y_axis_range=[]):
    '''plot average stim response of each cell at each stim. timepoint, along with the average dFF across all cells at each frame'''
    dff = ao.normalize_dff(exp_obj.raw)
    dff_mean = np.mean(dff, axis=0)

    keys = df.keys()[1:]
    response = [df["%s" % stim].mean() for stim in df.keys()[1:]]
    x = [int(i) for i in keys[pd.notna(response)]]
    y = [a for a in response if str(a) != 'nan']


    fig, ax1 = plt.subplots(figsize=(20, 3))
    color = 'gray'
    ax1.plot(dff_mean, color=color, alpha=0.4, linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    if type(exp_obj.seizure_frames) == list:
        ax1.axvline(x=exp_obj.seizure_frames[0], color='black')
        ax1.axvline(x=exp_obj.seizure_frames[-1], color='black')

    ax2 = ax1.twinx()
    color = 'black'

    for cell in exp_obj.good_naparm_cells_all:
        stim_group = df.loc[cell, 'groups']
        x = exp_obj.stim_start_frames[0][stim_group::exp_obj.n_groups]
        y = [df.loc[cell, '%s' % i] for i in x]
        mean_ = np.mean([y[i] for i in range(len(x)) if norm_end > x[i] > norm_start])

        y_ = [i/mean_ for i in y] # normalized response of this cell

        ax2.scatter(x, y_, s=5)
        ax2.plot(x, y_, linewidth=0.5)

    ax2.set_ylim(y_axis_range)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title((title), horizontalalignment='center', verticalalignment='top', pad=20, fontsize=15)
    fig.show()
plot_stim_responses_norm_cells(title='stim. responses by individual cells - normalized - 4ap', y_axis_range=[0,3],
                               norm_start=0, norm_end=exp_obj.seizure_frames[0])

#%% Quantify before CSD/seizure and after/seizure CSD stim. responses
df = exp_obj.cell_responses_dff

x_all = []
y_all = []
for cell in exp_obj.good_naparm_cells_all:
    stim_group = df.loc[cell, 'groups']
    x = exp_obj.stim_start_frames[0][stim_group::exp_obj.n_groups]
    y = [df.loc[cell, '%s' % i] for i in x]
    x_all.extend(x)
    y_all.extend(y)


stim_pre_csd = []
resp_pre_csd = []
stim_post_csd = []
resp_post_csd = []
data = {}

for cell in exp_obj.good_naparm_cells_all:
    stim_group = df.loc[cell, 'groups']
    x = exp_obj.stim_start_frames[0][stim_group::exp_obj.n_groups]
    y = np.array([df.loc[cell, '%s' % i] for i in x])
    stim_pre_csd.extend(list(x[np.where(x < exp_obj.seizure_frames[0])]))
    resp_pre_csd.extend(list(y[np.where(x < exp_obj.seizure_frames[0])]))
    stim_post_csd.extend(list(x[np.where(x > exp_obj.seizure_frames[0])]))
    resp_post_csd.extend(list(y[np.where(x > exp_obj.seizure_frames[0])]))

data['group'] = ['Pre-Sz'] * len(resp_pre_csd) + ['Post-Sz'] * len(resp_post_csd)
data['response'] = resp_pre_csd + resp_post_csd

data=pd.DataFrame(data)

import seaborn as sns
sns.catplot(x='group', y='response', data=data)
plt.ylim(0, 1000)
plt.show()

# TODO add t-test comparison


#%% PLOT individual cell's flu trace, with photostim. timings for that cell
cell = 59
group = [exp_obj.good_naparm_cells.index(i) for i in exp_obj.good_naparm_cells if cell in i][0]  # this will determine which slm group's photostim to plot on the flu trace

# plot flu trace of selected cell with the std threshold
idx = exp_obj.cell_id.index(cell)


def plot_flu_trace(exp_obj=exp_obj, idx=idx, slm_group=group, to_plot='raw'):
    raw = exp_obj.raw[idx]
    raw_ = np.delete(raw, exp_obj.photostim_frames)
    raw_dff = ao.normalize_dff(raw_)
    std_dff = np.std(raw_dff, axis=0)
    std = np.std(raw_, axis=0)

    if to_plot == 'raw':
        to_plot_ = raw
        to_thresh = std
    else:
        to_plot_ = raw_dff
        to_thresh = std_dff

    plt.figure(figsize=(20, 3))
    plt.plot(to_plot_, linewidth=0.1)
    # if to_plot == 'raw':
    #     plt.suptitle(('raw flu for cell #%s' % exp_obj.cell_id[idx]), horizontalalignment='center',
    #                  verticalalignment='top',
    #                  fontsize=15, y=1.00)
    # else:
    #     plt.scatter(x, y=[0] * len(x), c='r', linewidth=0.10)
    #     plt.axhline(y=np.mean(to_plot_) + 2.5 * to_thresh, c='green')
    #     plt.suptitle(('%s flu for cell #%s' % (to_plot, exp_obj.cell_id[idx])), horizontalalignment='center',
    #                  verticalalignment='top',
    #                  fontsize=15, y=1.00)

    for i in exp_obj.stim_start_frames[0][slm_group::exp_obj.n_groups]:
        plt.axvline(x=i - 1, c='gray', alpha=0.1)

    if exp_obj.seizure_frames:
        plt.scatter(exp_obj.seizure_frames, y=[-20] * len(exp_obj.seizure_frames), c='g', linewidth=0.10)

    # plt.ylim(0, 300)
    plt.show()
plot_flu_trace(); stat[idx]

#%%
#  plotting the distribution of radius and aspect ratios
radius = []
aspect_ratio = []
for cell in range(len(stat)):
    #if exp_obj.cell_id[cell] in exp_obj.good_cells:
    if exp_obj.cell_id[cell] in exp_obj.cell_id:
        radius.append(stat[cell]['radius'])
        aspect_ratio.append(stat[cell]['aspect_ratio'])

to_plot=radius
n, bins, patches = plt.hist(to_plot, 100)
plt.axvline(3.5)
plt.axvline(8.5)
plt.suptitle('All cells', y=0.95)
plt.show()


#%% SAVE exp_obj as PKL
# Pickle the expobject output to save it for analysis
# save_path = "/home/pshah/mnt/qnap/Data/2020-03-18/J063/2020-03-18_J063_%s/2020-03-18_%s.pkl" % (trial, trial)
with open(pkl_path, 'wb') as f:
        pickle.dump(exp_obj, f)
print("pkl saved to %s" % pkl_path)
















#%%
###### photostim analysis - select SLM group to analyse
# select SLM group of cells to analyze
group = 31
stim_timings = exp_obj.stim_start_frames[0][group::exp_obj.stim_duration_frames]
title = 'SLM naparm Group #%s' % group
#is_target = exp_obj.s2p_targets_naparm[group]
targeted_cells = [cell for cell in exp_obj.s2p_targets_naparm[group] if cell in exp_obj.good_cells]
print(targeted_cells)

#%%
###### plot avg traces from PHOTOSTIM TARGETTED cells
# collect photostim timed average dff traces of photostim targets
targets_dff = []
pre_stim = 10
post_stim = 200
for cell in targeted_cells:
    if cell in exp_obj.cell_id:
        cell_idx = exp_obj.cell_id.index(cell)
        flu = []
        for stim in stim_timings:
            #frames_to_plot = ls(range(stim-8, stim+35))
            flu.append(exp_obj.raw[cell_idx][stim-pre_stim:stim+post_stim])

        flu_dff = []
        for trace in flu:
            mean = np.mean(trace[0:pre_stim])
            trace_dff = ((trace - mean) / mean) * 100
            flu_dff.append(trace_dff)

        targets_dff.append(np.mean(flu_dff, axis=0))

###### filter cells where photostimulated targets fire >thresh*std above pre-stim
good_photostim_cells = []
good_targets_dF_stdF = []
good_targets_dff = []
std_thresh=2
for cell in targeted_cells:
    trace = targets_dff[targeted_cells.index(cell)] # trace = averaged dff trace across all photostims. for this cell
    pre_stim_trace = trace[:pre_stim]
    post_stim_trace = trace[pre_stim + 4:post_stim] # 4 = stim duration photostim frames
    mean_pre = np.mean(pre_stim_trace)
    std_pre = np.std(pre_stim_trace)
    mean_post = np.mean(post_stim_trace[:10])
    dF_stdF = (trace - mean_pre) / std_pre # make dF divided by std of pre-stim F trace
    response = np.mean(dF_stdF[pre_stim + 4:pre_stim + 10])
    if response > std_thresh:
        good_photostim_cells.append(cell)
        good_targets_dF_stdF.append(dF_stdF)
        good_targets_dff.append(trace)
        print('Cell #%s - dF_stdF post-stim: %s' % (cell, response))
print('%s cells filtered out of %s targeted cells' % (len(good_photostim_cells), len(targeted_cells)))

#%% PLOTTING
# make plots of photostim targeted trials
def plot_photostim_avg(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None):
    len_ = len(dff_array)
    flu_avg = np.median(dff_array, axis=0)
    std = np.std(dff_array, axis=0)
    ci = 1.960 * (std / np.sqrt(len_))  # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
    x = list(range(-pre_stim, post_stim))
    y = flu_avg

    fig, ax = plt.subplots()
    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)  # plot confidence interval
    ax.axvspan(0, stim_duration, alpha=0.2, color='green')
    for i in range(15, post_stim, 15): # make sure to change this based on the fps of the naparm trial imaging
        ax.axvspan(i, i+stim_duration, color='gray', alpha=0.1)
    ax.plot(x, y)
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    fig.suptitle((title+' - %s' % len_+' cells'), y=0.95)
    plt.show()
plot_photostim_avg(dff_array=targets_dff, stim_duration=4, pre_stim=pre_stim, post_stim=post_stim, title=title)

def plot_photostim_(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None):
    len_ = len(dff_array)
    x = list(range(-pre_stim, post_stim))
    fig, ax = plt.subplots()
    ax.axvspan(0, stim_duration, alpha=0.2, color='green')
    for cell_trace in dff_array:
        ax.plot(x, cell_trace, linewidth='0.5', alpha=0.5)
    for i in range(15, post_stim, 15): # make sure to change this based on the fps of the naparm trial imaging
        ax.axvspan(i, i+stim_duration, color='gray', alpha=0.1)
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    fig.suptitle((title + ' - %s' % len_ + ' cells'), y=0.95)
    plt.show()
# plot_photostim_(dff_array=targets_dff,pre_stim_sec=pre_stim_sec, post_stim_sec=post_stim_sec, title=title)

# check to see what the filtered cells' photostim response look like
plot_photostim_avg(dff_array=good_targets_dff, stim_duration=4, pre_stim=pre_stim, post_stim=post_stim, title=title + ' - filtered - avg of all cells')
plot_photostim_(dff_array=good_targets_dff, stim_duration=4, pre_stim=pre_stim, post_stim=post_stim, title=title + ' - filtered - individual cells')



# TODO photostimulation of targeted cells before CSD, just after CSD, and a while after CSD

# TODO photostimulation of targeted cells before seizure, just after seizure, and a while after seizure


################
# TODO perform analysis for Naparm trials


