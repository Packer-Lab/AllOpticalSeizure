# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import draw

# %% analysis for SLM targets responses

###### IMPORT pkl file containing data in form of expobj
trial = 't-016'
date = '2020-12-19'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

def slm_targets_responses(expobj):
    # plot SLM photostim individual targets -- individual, full traces, dff normalized

    # make rolling average for these plots to smooth out the traces a little more
    w = 3
    to_plot = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets])
    # to_plot = expobj.dff_SLMTargets

    # aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Time (secs.)',
    #                              y_label='dFF Flu', title=experiment)

    aoplot.plot_photostim_traces_overlap(array=expobj.dff_SLMTargets, expobj=expobj, x_axis='Time (secs.)',
                                         y_spacing_factor=2,
                                         title='%s - dFF Flu photostims' % experiment,
                                         figsize=(2 * 20, 2 * len(to_plot) * 0.15))

    y_label = 'dF/prestim_stdF'
    aoplot.plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dfstdF_avg, expobj=expobj,
                                  stim_duration=expobj.stim_duration_frames,
                                  figsize=[5, 4], y_lims=[-0.5, 1.5],
                                  title=('%s - responses of all photostim targets' % expobj.metainfo['trial']),
                                  y_label=y_label, x_label='Time post-stimulation (seconds)')

    if hasattr(expobj, 'stims_in_sz'):
        # stims out sz
        data = [[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]]
        pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_out_sz - ')
        pj.plot_bar_with_points(data=[list(expobj.outsz_StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                                ylims=[0, 100], bar=False,
                                y_label='% success stims.',
                                title='%s success rate of stim responses (stims out sz)' % trial, expand_size_x=2)

        # stims in sz
        data = [[np.mean(expobj.insz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]]
        pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ')
        pj.plot_bar_with_points(data=[list(expobj.insz_StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                                ylims=[0, 100], bar=False,
                                y_label='% success stims.',
                                title='%s success rate of stim responses (stims in sz)' % trial, expand_size_x=2)

    else:
        # no sz
        data = [[np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]]
        pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='no sz')
        pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                                ylims=[0, 100], bar=False,
                                y_label='% success stims.', title='%s success rate of stim responses (no sz)' % trial,
                                expand_size_x=2)




# %%
###### IMPORT pkl file containing data in form of expobj
trial = 't-010'
date = '2021-01-08'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date)

# if not hasattr(expobj, 's2p_path'):
#     expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-1x-alltrials/plane0'

if not hasattr(expobj, 'meanRawFluTrace'):
    expobj.mean_raw_flu_trace(plot=True)

plot = True
if plot:
    aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Time', figsize=[20, 3])
    aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='frames', figsize=[20, 3])
    aoplot.plotSLMtargetsLocs(expobj, background=expobj.meanFluImg_registered)
    aoplot.plot_lfp_stims(expobj)









#%% ####################################################################################################################

#### -------------------- ALL OPTICAL PHOTOSTIM AND ETC. ANALYSIS STEPS ################################################

########################################################################################################################

# %% PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - PHOTOSTIM TARGETTED suite2p ROIs cells

# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
x = np.asarray([i for i in expobj.targets_dfstdF_avg])
# y_label = 'pct. dFF (normalized to prestim period)'
y_label = 'dFstdF (normalized to prestim period)'

aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, stim_duration=expobj.stim_duration_frames, pre_stim=expobj.pre_stim,
                              post_stim=expobj.post_stim,
                              title=(experiment + '- responses of all photostim targets'),
                              y_label=y_label, x_label='Time post-stimulation (seconds)')

# %% PLOT ENTIRE TRIAL - PHOTOSTIM targeted suite2p ROIs cells plotted individually entire Flu trace

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



# %% plot SLM photostim individual targets -- individual, full traces, dff normalized

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
#     SLMTargets_stims_raw, SLMTargets_stims_dff, SLMtargets_stims_dfstdF = expobj.get_alltargets_stim_traces_norm(targets_idx=i, pre_stim=pre_stim,
#                                                                                                                  post_stim=post_stim)
#     if plot:
#         w = 2
#         array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in SLMTargets_stims_raw]
#         random_sub = np.random.randint(0,100,10)
#         aoplot.plot_periphotostim_avg(arr=SLMtargets_stims_dfstdF[random_sub], expobj=expobj, stim_duration=expobj.stim_duration_frames,
#                                       title='Target ' + str(i), pre_stim=pre_stim, post_stim=post_stim, color='steelblue', y_lims=[-0.5, 2.5])
#     # plt.show()


# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
# x = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])

y_label = 'dF/prestim_stdF'
aoplot.plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dfstdF_avg, expobj=expobj, stim_duration=expobj.stim_duration_frames,
                              figsize=[5, 4], y_lims=[-0.5, 3],
                              title=(experiment + '- responses of all photostim targets'),
                              y_label=y_label, x_label='Time post-stimulation (seconds)')

# %% plotting of photostim. success rate

data = [np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]

pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)')
pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=['t-010'], ylims=[0, 100], bar=False, y_label='% success stims.',
                        title='%s success rate of stim responses' % trial, expand_size_x=2)




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
pj.plot_bar_with_points(data=[group1, group2], x_tick_labels=['photostim target', 'non-target'], xlims=[0, 0.6],
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


# %% PLOT imshow() XY area locations with COLORS AS average response of ALL cells in FOV
aoplot.xyloc_responses(expobj, to_plot='dfstdf', clim=[-1, +1], plot_target_coords=True)



# %% PLOT HEATMAP of SEIZURE EVENTS

sz = 2
sz_onset, sz_offset = expobj.stims_bf_sz[sz], expobj.stims_af_sz[sz+1]

# -- approach of dFF normalize to the mean of the Flu data 2 seconds before the seizure
pre_sz = 2*int(expobj.fps)
sz_flu = expobj.raw[[expobj.cell_id.index(cell) for cell in expobj.good_cells], sz_onset - pre_sz: sz_offset]
sz_flu_smooth = np.array([pj.smooth_signal(signal, w=5) for signal in sz_flu])  # grouped average of the raw signal
x_norm = np.array([pj.dff(flu[pre_sz:], np.mean(flu[:pre_sz])) * 100 for flu in sz_flu_smooth])


stims = [(stim - sz_onset) for stim in expobj.stim_start_frames if sz_onset <= stim < sz_offset]
stims_off = [(stim + expobj.stim_duration_frames - 1) for stim in stims]

x_bf = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_bf_sz[sz])[0][0]]
x_af = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_af_sz[sz+1])[0][0]]

lfp_signal = expobj.lfp_signal[x_bf:x_af]

# -- ordering cells based on their order of reaching top 5% signal threshold
x_95 = [np.percentile(trace, 95) for trace in x_norm]

x_peak = [np.min(np.where(x_norm[i] > x_95[i])) for i in range(len(x_norm))]
new_order = np.argsort(x_peak)
x_ordered = x_norm[new_order]

# plot heatmap of dFF processed Flu signals for all cells for selected sz and ordered as determined above
aoplot.plot_traces_heatmap(x_ordered, stim_on=stims, stim_off=stims_off, cmap='Spectral_r', figsize=(10, 6),
                           title=('%s - seizure %s - sz flu smooth - %s to %s' % (trial, sz, sz_onset, sz_offset)),
                           xlims=None, vmin=100, vmax=500, lfp_signal=lfp_signal)

# just the bottom half cells that seems to show more of an order
x_ordered = x_norm[new_order[250:]]
aoplot.plot_traces_heatmap(x_ordered, stim_on=stims, stim_off=stims_off, cmap='Spectral_r', figsize=(10, 6),
                           title=('%s - seizure %s - sz flu smooth - %s to %s' % (trial, sz, sz_onset, sz_offset)),
                           xlims=None, vmin=100, vmax=500, lfp_signal=lfp_signal)


# %% PLOT cell location with cmap based on their order of reaching top 5% signal during sz event

cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
aoplot.plot_cell_loc(expobj, cells=cell_ids_ordered, show_s2p_targets=False, color_float_list=list(range(len(cell_ids_ordered))),
                     title='cell locations ordered by recruitment in sz # %s' % sz, invert_y=True, cmap='Purples')

# just the bottom half cells that seems to show more of an order
cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
aoplot.plot_cell_loc(expobj, cells=cell_ids_ordered[250:], show_s2p_targets=False, color_float_list=list(np.array(x_peak)[new_order][250:]),
                     title='cell locations ordered by recruitment in sz # %s' % sz, invert_y=True, cmap='Purples')

# %% plot the target photostim responses for individual targets for each stim over the course of the trial
#    (normalize to each target's overall mean response) and plot over the timecourse of the trial


SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
target_colors = pj.make_random_color_array(SLMtarget_ids)

# --- plot with mean FOV fluorescence signal
fig, ax1 = plt.subplots(figsize=[20, 3])
fig, ax1 = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', figsize=[20, 3], show=False,
                                      fig=fig, ax=ax1)
ax2 = ax1.twinx()
for target in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[target])
    # print(mean_response)
    for i in range(len(expobj.stim_start_frames)):
        response = expobj.responses_SLMtargets[target][i] - mean_response
        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)
        ax2.scatter(x=expobj.stim_start_frames[i] + rand, y=response, color=target_colors[target], alpha=0.70, s=15, zorder=4)
# for i in expobj.stim_start_frames:
#     plt.axvline(i)
plt.show()


# %% --- plot with LFP signal
fig1, ax1 = plt.subplots(figsize=[20, 3])
fig1, ax1 = aoplot.plotLfpSignal(expobj, color='black', stim_span_color='', x_axis='time', show=False, fig=fig1, ax=ax1, alpha=0.5)
ax2 = ax1.twinx()
for target in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[target])
    target_coord = expobj.target_coords_all[target]
    # print(mean_response)
    x = []
    y = []
    distance_to_sz = []

    # calculate response magnitude at each stim time for selected target
    for i in range(len(expobj.stim_times)):
        # the response magnitude of the current SLM target at the current stim time (relative to the mean of the responses of the target over this trial)
        response = expobj.responses_SLMtargets[target][i] - mean_response  # changed to division by mean response instead of substracting
        min_distance = pj.calc_distance_2points((0, 0), (expobj.frame_x,
                                                         expobj.frame_y))  # maximum distance possible between two points within the FOV, used as the starting point when the sz has not invaded FOV yet

        if hasattr(expobj, 'cells_sz_stim') and expobj.stim_start_frames[i] in list(expobj.cells_sz_stim.keys()):  # calculate distance to sz only for stims where cell locations in or out of sz boundary are defined in the seizures
            if expobj.stim_start_frames[i] in expobj.stims_in_sz:
            # if (expobj.stim_start_frames[i] not in expobj.stims_bf_sz) and (expobj.stim_start_frames[i] not in expobj.stims_af_sz):  # also calculate distance for stims before and after
                # collect cells from this stim that are in sz
                s2pcells_sz = expobj.cells_sz_stim[expobj.stim_start_frames[i]]

                # classify the SLM target as in or out of sz, if out then continue with mesauring distance to seizure wavefront,
                # if in sz then assign negative value for distance to sz wavefront
                sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (
                    expobj.analysis_save_path, trial, expobj.stim_start_frames[i])

                in_sz_bool = expobj._InOutSz(cell_med=[target_coord[1], target_coord[0]],
                                             sz_border_path=sz_border_path)

                if expobj.stim_start_frames[i] in expobj.not_flip_stims:
                    flip = False
                else:
                    flip = True
                    in_sz_bool = not in_sz_bool

                if in_sz_bool is True:
                    min_distance = -1

                else:
                    ## working on add feature for edgecolor of scatter plot based on calculated distance to seizure
                    ## -- thinking about doing this as comparing distances between all targets and all suite2p ROIs,
                    #     and the shortest distance that is found for each SLM target is that target's distance to seizure wavefront
                    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
                    if len(s2pcells_sz) > 0:
                        for j in range(len(s2pcells_sz)):
                            s2p_idx = expobj.cell_id.index(s2pcells_sz[j])
                            dist = pj.calc_distance_2points(target_coord, tuple(
                                [expobj.stat[s2p_idx]['med'][1], expobj.stat[s2p_idx]['med'][0]]))  # distance in pixels
                            if dist < min_distance:
                                min_distance = dist

                            # if j < 5:
                            #     fig, ax = pj.plot_cell_loc(expobj, cells=[s2pcells_sz[j]], show=False, fig=fig, ax=ax,
                            #                                background=expobj.meanFluImg_registered)
                            #     ax.scatter(x=target_coord[0], y=target_coord[1])

                # if 10 < min_distance < 40:
                #     fig, ax = plt.subplots()
                #     fig, ax = alloptical_plotting.plot_cell_loc(expobj, cells=s2pcells_sz, show=False, fig=fig, ax=ax,
                #                                                 background=expobj.meanFluImg_registered)
                #     ax.scatter(x=target_coord[0], y=target_coord[1])
                #     plt.title('stim %s' % expobj.stim_start_frames[i])
                #     fig.show()
                #     print(min_distance)

        # min_distance.append((np.random.rand(1) * 1000)[0])  # just for testing
        distance_to_sz.append(min_distance)
        # plot the response magnitude of the current SLM target at the current stim time
        rand = np.random.randint(-10, 30, 1)[
            0]  # * 1/(abs(response)**1/2)  # used for adding random jitter to the x loc scatter point
        x.append(expobj.stim_times[i] - expobj.frame_start_time_actual + rand * 1e3)
        y.append(response)
    if np.std(distance_to_sz) > 0.1:
        ax2.scatter(x=x, y=y, c=distance_to_sz, cmap='RdYlBu_r',
                    alpha=0.5, s=10,
                    zorder=4)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
    else:
        ax2.scatter(x=x, y=y, facecolor='#BA1C32', alpha=0.5, s=10,
                    zorder=4)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
    # fig1.show()
    # ax2.scatter(x=x, y=y, c=distance_to_sz, cmap='RdYlBu_r',
    #             alpha=0.70, s=15,
    #             zorder=4)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
    # ax2.scatter(x=expobj.stim_times[i] + rand * 1e3, y=response, edgecolor=target_colors[target], alpha=0.70, s=15, zorder=4)  # use same edgecolor for each target at all stim times
# for i in expobj.stim_start_frames:
#     plt.axvline(i)
plt.xlim([expobj.stim_times[0] - 4e5, expobj.stim_times[-1] + 1e5])
fig1.show()

# %% plot response magnitude vs. distance

fig1, ax1 = plt.subplots(figsize=[5, 5])
responses = []
distance_to_sz = []
responses_ = []
distance_to_sz_ = []
for target in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[target])
    target_coord = expobj.target_coords_all[target]
    # print(mean_response)

    # calculate response magnitude at each stim time for selected target
    for i in range(len(expobj.stim_times)):
        # the response magnitude of the current SLM target at the current stim time (relative to the mean of the responses of the target over this trial)
        response = expobj.responses_SLMtargets[target][i] / mean_response  # changed to division by mean response instead of substracting
        min_distance = pj.calc_distance_2points((0, 0), (expobj.frame_x,
                                                         expobj.frame_y))  # maximum distance possible between two points within the FOV, used as the starting point when the sz has not invaded FOV yet

        if hasattr(expobj, 'cells_sz_stim') and expobj.stim_start_frames[i] in list(expobj.cells_sz_stim.keys()):  # calculate distance to sz only for stims where cell locations in or out of sz boundary are defined in the seizures
            if expobj.stim_start_frames[i] in expobj.stims_in_sz:
                # collect cells from this stim that are in sz
                s2pcells_sz = expobj.cells_sz_stim[expobj.stim_start_frames[i]]

                # classify the SLM target as in or out of sz, if out then continue with mesauring distance to seizure wavefront,
                # if in sz then assign negative value for distance to sz wavefront
                sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (
                    expobj.analysis_save_path, trial, expobj.stim_start_frames[i])

                in_sz_bool = expobj._InOutSz(cell_med=[target_coord[1], target_coord[0]],
                                             sz_border_path=sz_border_path)

                if expobj.stim_start_frames[i] in expobj.not_flip_stims:
                    flip = False
                else:
                    flip = True
                    in_sz_bool = not in_sz_bool

                if in_sz_bool is True:
                    min_distance = -1

                else:
                    ## working on add feature for edgecolor of scatter plot based on calculated distance to seizure
                    ## -- thinking about doing this as comparing distances between all targets and all suite2p ROIs,
                    #     and the shortest distance that is found for each SLM target is that target's distance to seizure wavefront
                    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
                    if len(s2pcells_sz) > 0:
                        for j in range(len(s2pcells_sz)):
                            s2p_idx = expobj.cell_id.index(s2pcells_sz[j])
                            dist = pj.calc_distance_2points(target_coord, tuple(
                                [expobj.stat[s2p_idx]['med'][1], expobj.stat[s2p_idx]['med'][0]]))  # distance in pixels
                            if dist < min_distance:
                                min_distance = dist

        if min_distance > 600:
            distance_to_sz_.append(min_distance + np.random.randint(-10, 10, 1)[0] - 165)
            responses_.append(response)
        elif min_distance > 0:
            distance_to_sz.append(min_distance)
            responses.append(response)

# calculate linear regression line
ax1.plot(range(int(min(distance_to_sz)), int(max(distance_to_sz))), np.poly1d(np.polyfit(distance_to_sz, responses, 1))(range(int(min(distance_to_sz)), int(max(distance_to_sz)))),
         color='black')

ax1.scatter(x=distance_to_sz, y=responses, color='cornflowerblue',
            alpha=0.5, s=16, zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
ax1.scatter(x=distance_to_sz_, y=responses_, color='firebrick',
            alpha=0.5, s=16, zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
ax1.set_xlabel('distance to seizure front (pixels)')
ax1.set_ylabel('response magnitude')
ax1.set_title('')
fig1.show()

#%%
for i in range(len(expobj.stim_times)):
    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
    s2pcells = expobj.cells_sz_stim[expobj.stim_start_frames[i]]
    target_coord = expobj.target_coords_all[target]
    min_distance = 1000
    for j in range(len(s2pcells)):
        dist = pj.calc_distance_2points(target_coord, tuple(expobj.stat[j]['med']))  # distance in pixels
        if dist < min_distance:
            min_distance = dist


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


