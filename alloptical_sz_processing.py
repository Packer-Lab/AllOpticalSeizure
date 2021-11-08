# %% 0) IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os
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

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = aoutils.import_resultsobj(pkl_path=results_object_path)


# %% 4.2) TODO measure seizure legnths across all imaging trials (including any spont imaging you might have)

for key in list(allopticalResults.trial_maps['post'].keys()):
    # import initial expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.0' % key, verbose=False)
    prep = expobj.metainfo['animal prep.']
    # look at all post4ap trials in expobj
    if 'post-4ap trials' in expobj.metainfo.keys():
        a = 'post-4ap trials'
    elif 'post4ap_trials' in expobj.metainfo.keys():
        a = 'post4ap_trials'
    # for loop over all of those post4ap trials
    for trial in expobj.metainfo['%s' % a]:
        # import expobj
        expobj, experiment = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
        # count the average length of each seizure
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
            counter = 0
            sz_lengths_total = 0
            if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                    if sz_onset != 0:
                        sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] - expobj.frame_clock_actual[sz_onset])/expobj.paq_rate
                        counter += 1
                avg_len = sz_lengths_total / counter
                expobj.avg_sz_len = avg_len

                print('Avg. seizure length (secs) for %s, %s, %s: ' % (prep, trial, expobj.metainfo['exptype']),
                      np.round(expobj.avg_sz_len, 2))

        else:
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
            n_seizures = 0
            print('no sz events for %s, %s, %s ' % (prep, trial, expobj.metainfo['exptype']))

# 4.2.1) measure seizure incidence across onePstim trials
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path, verbose=False)
        # count the average length of each seizure
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
            counter = 0
            sz_lengths_total = 0
            if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                    if sz_onset != 0:
                        sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] - expobj.frame_clock_actual[sz_onset])/expobj.paq_rate
                        counter += 1
                avg_len = sz_lengths_total / counter
                expobj.avg_sz_len = avg_len
                print('Avg. seizure length (secs) for %s, %s, %s: ' % (
                expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
                      np.round(expobj.avg_sz_len, 2))

        else:
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
            n_seizures = 0
            print('Avg. seizure length (secs) for %s, %s, %s ' % (
                expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']))


# 4.2.2) plot seizure length across onePstim and twoPstim trials
twop_trials = [24.0, 93.73, 38.86, 84.77, 17.16, 83.78, 15.78, 36.88]
onep_trials = [30.02, 34.25, 114.53, 35.57]

pj.plot_bar_with_points(data=[twop_trials, onep_trials], x_tick_labels=['2p stim', '1p stim'], colors=['purple', 'green'], y_label='seizure length (secs)',
                        title='Avg. length of sz', expand_size_x=0.4, expand_size_y=1, ylims=[0, 120], title_pad=15, shrink_text=0.8)



# %% 4.1) counting seizure incidence across all imaging trials

for key in list(allopticalResults.trial_maps['post'].keys()):
    # import initial expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.0' % key, verbose=False)
    prep = expobj.metainfo['animal prep.']
    # look at all post4ap trials in expobj
    if 'post-4ap trials' in expobj.metainfo.keys():
        a = 'post-4ap trials'
    elif 'post4ap_trials' in expobj.metainfo.keys():
        a = 'post4ap_trials'
    # for loop over all of those post4ap trials
    for trial in expobj.metainfo['%s' % a]:
        # import expobj
        expobj, experiment = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
        total_time_recording = np.round((expobj.n_frames/expobj.fps) / 60., 2)  # return time in mins

        # count seizure incidence (avg. over mins) for each experiment (animal)
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
        else:
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
            n_seizures = 0

        print('Seizure incidence for %s, %s, %s: ' % (prep, trial, expobj.metainfo['exptype']), np.round(n_seizures / total_time_recording, 2))

# 4.1.1) measure seizure incidence across onePstim trials
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path, verbose=False)
        total_time_recording = np.round((expobj.n_frames/expobj.fps) / 60., 2)  # return time in mins

        # count seizure incidence (avg. over mins) for each experiment (animal)
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
        else:
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
            n_seizures = 0

        print('Seizure incidence for %s, %s, %s: ' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']), np.round(n_seizures / total_time_recording, 2))

# 4.1.2) plot seizure incidence across onePstim and twoPstim trials
twop_trials = [0.35, 0.251666667, 0.91, 0.33, 0.553333333, 0.0875, 0.47, 0.33, 0.52]
onep_trials = [0.38, 0.26, 0.19, 0.436666667, 0.685]

pj.plot_bar_with_points(data=[twop_trials, onep_trials], x_tick_labels=['2p stim', '1p stim'], colors=['purple', 'green'], y_label='sz incidence (events/min)',
                        title='rate of seizures during exp', expand_size_x=0.4, expand_size_y=1, ylims=[0, 1], shrink_text=0.8)



#%% 1) defining trials to run for analysis

ls = [
    # ['RL108 t-013'],
    # # ['RL108 t-011'], - problem with pickle data being truncated
    # ['RL109 t-020'], - problem with pickle data being truncated
    # ['RL109 t-021'],
    # ['RL109 t-018'],
    # ['RL109 t-016'], #, 'RL109 t-017'], - need to redo paired measurements
    # ['PS04 t-018'], - problem with pickle data being truncated
    ['PS05 t-012'],
    ['PS07 t-011'],
    # ['PS07 t-017'], - lots of little tiny seizures, not sure if they are propagating through likely not
    # ['PS06 t-014', 'PS06 t-015'], - missing seizure_lfp_onsets
    ['PS06 t-013'],
    # ['PS06 t-016'], - missing seizure_lfp_onsets
    ['PS11 t-016'],
    ['PS11 t-011'],
    # ['PS17 t-011'],
    ['PS17 t-009'],
    # ['PS18 t-008']
]

ls2 = [
    ['PS07 t-011'],
]

# %% 2.1) classification of cells in/out of sz boundary

# using post4ap experiments from allopticalResults attr. in for loop for processing:

# for i in allopticalResults.post_4ap_trials:
for i in ls2:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[-1]
        # j = -1
        # prep = 'RL109'
        # trial = 't-016'
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial)
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
        aoplot.plot_lfp_stims(expobj)

# matlab_pairedmeasurements_path = '%s/paired_measurements/%s_%s_%s.mat' % (expobj.analysis_save_path[:-23], expobj.metainfo['date'], expobj.metainfo['animal prep.'], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
# expobj.paqProcessing()
# expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path)
# expobj.save()

# aoplot.plotSLMtargetsLocs(expobj, background=None)

# ######## CLASSIFY SLM PHOTOSTIM TARGETS AS IN OR OUT OF current SZ location in the FOV
# -- FIRST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv

        if not hasattr(expobj, 'sz_boundary_csv_done'):
            expobj.sz_boundary_csv_done = True
        else:
            AssertionError('confirm that sz boundary csv creation has been completed')
            # sys.exit()

        # specify stims for classifying cells
        on_ = []
        if 0 in expobj.seizure_lfp_onsets:  # this is used to check if 2p imaging is starting mid-seizure (which should be signified by the first lfp onset being set at frame # 0)
            on_ = on_ + [expobj.stim_start_frames[0]]
        on_.extend(expobj.stims_bf_sz)
        if len(expobj.stims_af_sz) != len(on_):
            end = expobj.stims_af_sz + [expobj.stim_start_frames[-1]]
        else:
            end = expobj.stims_af_sz
        print('seizure start frames: ', on_)
        print('seizure end frames: ', end)

        ##### import the CSV file in and classify cells by their location in or out of seizure

        if not hasattr(expobj, 'not_flip_stims'):
        # if hasattr(expobj, 'not_flip_stims'):
            expobj.not_flip_stims = []

        print('working on classifying cells for stims start frames...')
        expobj.slmtargets_sz_stim = {}

#  2.1.1) ######## - all stims in sz are classified, with individual sz events labelled

        stims_of_interest = [stim for stim in expobj.stims_in_sz[1:]]
        print('\n all stims in seizures: \n|-', stims_of_interest)
        nrows = len(stims_of_interest) // 4 + 1
        if nrows == 1:
            nrows += 1
        ncols = 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
        counter = 0

        sz_num = 0
        for on, off in zip(on_, end):
            stims_of_interest = [stim for stim in expobj.stim_start_frames if on < stim < off if stim != expobj.stims_in_sz[0]]
            print('\n working sz # %s with stims: \n|-' % (sz_num), stims_of_interest)

            for stim in stims_of_interest:
                ax = axs[counter // ncols, counter % ncols]

                sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (
                expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
                if not os.path.exists(sz_border_path):
                    print(sz_border_path)
                # first round of classifying (dont flip any cells over) - do this in the second round
                if stim not in expobj.not_flip_stims:
                    flip = False
                else:
                    flip = True

                # classification of suite2p ROIs relative to sz boundary
                in_sz, out_sz, fig, ax = expobj.classify_cells_sz_bound(sz_border_path, stim=stim, to_plot=True,
                                                                        flip=flip, fig=fig, ax=ax, text='sz %s stim %s' % (sz_num, stim))
                # classification of SLM targets relative to sz boundary
                # in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(sz_border_path, stim=stim, to_plot=True, title='%s' % stim, flip=flip, fig=fig, ax=ax)
                expobj.slmtargets_sz_stim[
                    stim] = in_sz  # for each stim, there will be a list of cells that will be classified as in seizure or out of seizure

                axs[counter // ncols, counter % ncols] = ax
                counter += 1
            sz_num += 1

        fig.suptitle('%s %s - Avg img around stims during- all stims' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']), y=0.995)
        fig.show()

        # expobj.save()
print('end end end.')

# %% 2.2) need to repeat the above code
# to correct for mis-assignment of cells (look at results and then find out which stims need to be flipped)

 # ['RL109 t-020'],
    # ['RL109 t-021'],
    # ['RL109 t-018'],
# prep = 'PS11'
# trial = 't-011'
# expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=True)


# expobj.not_flip_stims = [expobj.stims_in_sz[1:]]  # specify here the stims where the flip=False leads to incorrect assignment
expobj.not_flip_stims = expobj.stims_in_sz[1:]  # specify here the stims where the flip=False leads to incorrect assignment

expobj.save()

# %% 2.3) re-run with new flip stims
expobj.slmtargets_sz_stim = {}
sz_num = 0
for on, off in zip(on_, end):
    stims_of_interest = [stim for stim in expobj.stim_start_frames if on < stim < off if stim != expobj.stims_in_sz[0]]
    print('|-', stims_of_interest)

    nrows = len(stims_of_interest) // 4 + 1
    if nrows == 1:
        nrows += 1
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    counter = 0

    for stim in stims_of_interest:
        ax = axs[counter // ncols, counter % ncols]

        sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
        if not os.path.exists(sz_border_path):
            print(sz_border_path)
        # first round of classifying (dont flip any cells over) - do this in the second round
        if stim not in expobj.not_flip_stims:
            flip = False
        else:
            flip = True

        in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(sz_border_path, stim=stim, to_plot=True, title='%s' % stim, flip=flip, fig=fig, ax=ax)
        expobj.slmtargets_sz_stim[stim] = in_sz  # for each stim, there will be a list of cells that will be classified as in seizure or out of seizure

        axs[counter // ncols, counter % ncols] = ax
        counter += 1
    fig.suptitle('%s %s - Avg img around stims during sz - seizure # %s' % (
    expobj.metainfo['animal prep.'], expobj.metainfo['trial'], sz_num + 1), y=0.995)
    fig.show()
    sz_num += 1


# %% 3) responses of targets during seizures, but EXCLUDE STIMS WHERE THE CELL IS INSIDE THE SZ BOUND -- should this be in alloptical_results_photostim.py????

if hasattr(expobj, 'stims_in_sz'):

    # stims during sz
    stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
    if len(stims) > 0:
        expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
            aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=0.15, stims_to_use=stims)

    # stims outside sz
    stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
    if len(stims) > 0:
        expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
            aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=0.15, stims_to_use=stims)


else:
    expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
        aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=0.15, stims_to_use=None)

# collect stim responses with stims excluded as necessary
targets_avgresponses_exclude_stims_sz = {}
for row in expobj.insz_responses_SLMtargets.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
    responses = [expobj.insz_responses_SLMtargets.loc[row][stim] for stim in expobj.stims_in_sz if row not in expobj.slmtargets_sz_stim[stim]]
    targets_avgresponses_exclude_stims_sz[row] = np.mean(responses)


targets_avgresponses_stims_outsz = {}
for row in expobj.insz_responses_SLMtargets.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    responses = [expobj.outsz_responses_SLMtargets.loc[row][stim] for stim in expobj.stims_out_sz]
    targets_avgresponses_stims_outsz[row] = np.mean(responses)


targets_avgresponses_stims_presz = {}
for row in expobj.responses_SLMtargets.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    responses = [expobj.responses_SLMtargets.loc[row][stim] for stim in expobj.stim_start_frames]
    targets_avgresponses_stims_presz[row] = np.mean(responses)


data = [list(targets_avgresponses_exclude_stims_sz.values()), list(targets_avgresponses_stims_outsz.values()), list(targets_avgresponses_stims_presz.values())]
pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ', figsize=(5,5), fill_color=['orange', 'skyblue', 'green'],
                     colors=['black'] * 3, legend_labels=[None] * 3)


# %% 6)
################################# SEIZURE EVENTS PLOTTING ##############################################################
########################################################################################################################

# PLOT HEATMAP of SEIZURE EVENTS

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


# PLOT cell location with cmap based on their order of reaching top 5% signal during sz event

cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
aoplot.plot_cells_loc(expobj, cells=cell_ids_ordered, show_s2p_targets=False, color_float_list=list(range(len(cell_ids_ordered))),
                      title='cell locations ordered by recruitment in sz # %s' % sz, invert_y=True, cmap='Purples')

# just the bottom half cells that seems to show more of an order
cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
aoplot.plot_cells_loc(expobj, cells=cell_ids_ordered[250:], show_s2p_targets=False, color_float_list=list(np.array(x_peak)[new_order][250:]),
                      title='cell locations ordered by recruitment in sz # %s' % sz, invert_y=True, cmap='Purples')


















































# %%
# plots for SLM targets responses
def slm_targets_responses(expobj, experiment, trial, y_spacing_factor=2, figsize=[20, 20], smooth_overlap_traces=5, linewidth_overlap_traces=0.2, dff_threshold=0.15,
                          y_lims_periphotostim_trace=[-0.5, 2.0], v_lims_periphotostim_heatmap=[-5, 5], save_results=True, force_redo=False, cmap=None):
    # plot SLM photostim individual targets -- individual, full traces, dff normalized

    # make rolling average for these plots to smooth out the traces a little more

    # force_redo = False
    if force_redo:
        # expobj._findTargets()
        # expobj.raw_traces_from_targets(force_redo=force_redo, save=True)
        # expobj.save()

        if hasattr(expobj, 'stims_in_sz'):
            seizure_filter = True

            stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
            # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
            if len(stims) > 0:
                expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=dff_threshold, stims_to_use=stims)

                # expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
                #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=raw_traces_stims,
                #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
                #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
                #                               verbose=True, plot=False)

            stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
            # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
            if len(stims) > 0:
                expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=dff_threshold, stims_to_use=stims)

                # expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
                #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=raw_traces_stims,
                #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
                #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
                #                               verbose=True, plot=False)

        else:
            seizure_filter = False
            print('\n Calculating stim success rates and response magnitudes ***********')
            expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
                aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=dff_threshold, stims_to_use=expobj.stim_start_frames)

            # expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
            #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=expobj.SLMTargets_stims_raw,
            #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
            #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
            #                               verbose=True, plot=False)

        expobj.save()


    ####################################################################################################################
    w = smooth_overlap_traces
    to_plot = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets])
    # to_plot = expobj.dff_SLMTargets
    # aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Time (secs.)',
    #                              y_label='dFF Flu', title=experiment)


    # initialize figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(4, 8)

    ax9 = fig.add_subplot(gs[0, 0])
    fig, ax9 = aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered, title=None, fig=fig, ax=ax9, show=False)


    ax0 = fig.add_subplot(gs[0, 1:])
    ax0 = aoplot.plot_lfp_stims(expobj, fig=fig, ax=ax0, show=False, x_axis='Time (secs.)')

    ax1 = fig.add_subplot(gs[1:3, :])
    aoplot.plot_photostim_traces_overlap(array=expobj.dff_SLMTargets, expobj=expobj, x_axis='Time (secs.)',
                                         y_spacing_factor=y_spacing_factor, fig=fig, ax=ax1, show=False,
                                         title='%s - dFF Flu photostims' % experiment, linewidth=linewidth_overlap_traces,
                                         figsize=(2 * 20, 2 * len(to_plot) * 0.15))

    ax2 = fig.add_subplot(gs[-1, 0:2])
    y_label = 'dF/F'
    cell_avg_stim_traces = expobj.SLMTargets_stims_dffAvg
    aoplot.plot_periphotostim_avg(arr=cell_avg_stim_traces, expobj=expobj,
                                  stim_duration=expobj.stim_duration_frames,
                                  figsize=[5, 4], y_lims=y_lims_periphotostim_trace, fig=fig, ax=ax2, show=False,
                                  title=('responses of all photostim targets'),
                                  y_label=y_label, x_label='Time post-stimulation (seconds)')

    # fig.show()

    if hasattr(expobj, 'stims_in_sz'):

        # make response magnitude and response success rate figure
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=((5 * 4), 5), nrows=1, ncols=4)
        # stims out sz
        ax3 = fig.add_subplot(gs[-1, 2:4])
        data = [[np.mean(expobj.outsz_responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        fig, ax3 = pj.plot_hist_density(data, x_label='response magnitude (dF/F)', title='stims_out_sz - ',
                                     fig=fig, ax=ax3, show=False)
        ax4 = fig.add_subplot(gs[-1, 4])
        fig, ax4 = pj.plot_bar_with_points(data=[list(expobj.outsz_StimSuccessRate_SLMtargets.values())],
                                           x_tick_labels=[trial],
                                           ylims=[0, 100], bar=False, y_label='% success stims.',
                                           title='target success rate (stims out sz)', expand_size_x=2,
                                           show=False, fig=fig, ax=ax4)
        # stims in sz
        ax5 = fig.add_subplot(gs[-1, 5:7])
        data = [[np.mean(expobj.insz_responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        fig, ax5 = pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ',
                                        fig=fig, ax=ax5, show=False)
        ax6 = fig.add_subplot(gs[-1, 7])
        fig, ax6 = pj.plot_bar_with_points(data=[list(expobj.insz_StimSuccessRate_SLMtargets.values())],
                                        x_tick_labels=[trial],
                                        ylims=[0, 100], bar=False, y_label='% success stims.',
                                        title='target success rate (stims in sz)', expand_size_x=2,
                                        show=False, fig=fig, ax=ax6)
        fig.tight_layout()
        if save_results:
            save = expobj.analysis_save_path[:-17] + '/results/' + '%s_%s_slm_targets_responses_dFF' % (expobj.metainfo['animal prep.'], trial)
            if not os.path.exists(expobj.analysis_save_path[:-17] + '/results'):
                os.makedirs(expobj.analysis_save_path[:-17] + '/results')
            print('saving png and svg to: %s' % save)
            fig.savefig(fname=save + '.png', transparent=True, format='png')
            fig.savefig(fname=save + '.svg', transparent=True, format='svg')

        fig.show()

    else:
        # no sz
        # fig, (ax1, ax2) = plt.subplots(figsize=((5 * 2), 5), nrows=1, ncols=2)
        data = [[np.mean(expobj.responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        ax3 = fig.add_subplot(gs[-1, 2:4])
        fig, ax3 = pj.plot_hist_density(data, x_label='response magnitude (dF/F)', title='no sz', show=False, fig=fig, ax=ax3)
        ax4 = fig.add_subplot(gs[-1, 4])
        fig, ax4 = pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                                           ylims=[0, 100], bar=False, show=False, fig=fig, ax=ax4,
                                           y_label='% success stims.', title='target success rate (pre4ap)',
                                           expand_size_x=2)

        zero_point = abs(v_lims_periphotostim_heatmap[0]/v_lims_periphotostim_heatmap[1])
        c = ColorConverter().to_rgb
        if cmap is None:
            cmap = pj.make_colormap([c('blue'), c('white'), zero_point - 0.20, c('white'), c('red')])
        ax5 = fig.add_subplot(gs[-1, 5:])
        fig, ax5 = aoplot.plot_traces_heatmap(data=cell_avg_stim_traces, expobj=expobj, vmin=v_lims_periphotostim_heatmap[0], vmax=v_lims_periphotostim_heatmap[1],
                                              stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames + 1, cbar=False,
                                              title=(expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo[
                                              'trial'] + ' - SLM targets raw Flu'), show=False, fig=fig, ax=ax5, x_label='Frames', y_label='Neurons',
                                              xlims=(0, expobj.pre_stim + expobj.stim_duration_frames + expobj.post_stim),
                                              cmap=cmap)

        fig.tight_layout()
        if save_results:
            save = expobj.analysis_save_path[:-17] + '/results/' + '%s_%s_slm_targets_responses_dFF' % (expobj.metainfo['animal prep.'], trial)
            if not os.path.exists(expobj.analysis_save_path[:-17] + '/results'):
                os.makedirs(expobj.analysis_save_path[:-17] + '/results')
            print('saving png and svg to: %s' % save)
            fig.savefig(fname=save+'.png', transparent=True, format='png')
            fig.savefig(fname=save+'.svg', transparent=True,  format='svg')

        fig.show()
