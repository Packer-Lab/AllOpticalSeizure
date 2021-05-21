#%% DATA ANALYSIS FOR ONE-P PHOTOSTIM EXPERIMENTS
import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot



# %% ###### IMPORT pkl file containing data in form of expobj
trial = 't-012'
date = '2021-01-19'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date,
                                           pkl_path='/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl' % (date, date, trial, date, trial))


# %% # look at the average Ca Flu trace pre and post stim, just calculate the average of the whole frame and plot as continuous timeseries
# - this approach should also allow to look at the stims that give rise to extended seizure events where the Ca Flu stays up

# # EXCLUDE CERTAIN STIM START FRAMES
# expobj.stim_start_frames = [frame for frame in expobj.stim_start_frames if 4000 > frame or frame > 5000]
# expobj.stim_end_frames = [frame for frame in expobj.stim_end_frames if 4000 > frame or frame > 5000]
# expobj.stim_start_times = [time for time in expobj.stim_start_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_end_times = [time for time in expobj.stim_end_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_duration_frames = int(np.mean(
#     [expobj.stim_end_frames[idx] - expobj.stim_start_frames[idx] for idx in range(len(expobj.stim_start_frames))]))

aoplot.plotMeanRawFluTrace(expobj, stim_span_color='white', x_axis='frames', xlims=[0, 3000])
aoplot.plotLfpSignal(expobj, x_axis='time')

aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff', quantify=True)

aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75,
                                 optoloopback=True)


# %% classifying stims as in or out of seizures

seizures_lfp_timing_matarray = '/home/pshah/mnt/qnap/Analysis/%s/paired_measurements/%s_%s_%s.mat' % (date, date, expobj.metainfo['animal prep.'], trial[-3:])

expobj.collect_seizures_info(seizures_lfp_timing_matarray=seizures_lfp_timing_matarray,
                             discard_all=False)


# %% ANALYSIS/PLOTTING STUFF
# create onePstim superobject that will collect analyses from various individual experiments

results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = aoutils.import_resultsobj(pkl_path=results_object_path)


# %% collection plots of many trials sub divided as specified


# pre-4ap trials plot
nrows = 4
ncols = 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
counter = 0; write_full_text=True
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'pre-4ap response'])[0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
        ax = axs[counter//ncols, counter % ncols]

        fig, ax, flu_list, mean_response, decay_constant = aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff', quantify=True,
                                                                                            show=False, fig=fig, ax=ax, write_full_text=write_full_text, shrink_text=1.25)
        # fig, ax = aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, optoloopback=True, show=False)

        axs[counter // ncols, counter % ncols] = ax

        counter += 1
        write_full_text = False  # switch off write full text option after the first plot

fig.suptitle('Pre-4ap trials only, avg flu trace for 1p stim', y=0.995)
fig.show()


# post-4ap stims out of sz trials plot
nrows = 4
ncols = 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
counter = 0; write_full_text=True
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (outside sz)'])[0] != '-':
        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
        ax = axs[counter//ncols, counter % ncols]

        title = 'Avg. trace - stims out of sz -'

        fig, ax, flu_list, mean_response, decay_constant = aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff', quantify=True,
                                                                                            show=False, fig=fig, ax=ax, write_full_text=write_full_text, shrink_text=1.25, stims_to_analyze=expobj.stims_out_sz,
                                                                                            title=title)
        # fig, ax = aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, optoloopback=True, show=False)

        axs[counter // ncols, counter % ncols] = ax

        counter += 1
        write_full_text = False  # switch off write full text option after the first plot

fig.suptitle('Post-4ap trials, stims out of sz, avg flu trace for 1p stim', y=0.995)
fig.show()



# post-4ap stims during sz trials plot
nrows = 4
ncols = 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
counter = 0; write_full_text = True
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[0] != '-':
        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
        ax = axs[counter//ncols, counter % ncols]

        title = 'Avg. trace - stims in sz -'

        fig, ax, flu_list, mean_response, decay_constant = aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff', quantify=True,
                                                                                            show=False, fig=fig, ax=ax, write_full_text=write_full_text, shrink_text=1.25, stims_to_analyze=expobj.stims_in_sz,
                                                                                            title=title)
        # fig, ax = aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, optoloopback=True, show=False)

        axs[counter // ncols, counter % ncols] = ax

        counter += 1
        write_full_text = False  # switch off write full text option after the first plot

fig.suptitle('Post-4ap trials, stims out of sz, avg flu trace for 1p stim', y=0.995)
fig.show()

# %% ADD DECAY CONSTANTS TO THE mean_stim_responses dataframe

# pre-4ap trials plot
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'pre-4ap response'])[0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)

        flu_list, mean_response, decay_constant = aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', y_axis='dff', show=False, quantify=True)
        onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses[
                                                'pkl_list'] == expobj.pkl_path, 'Decay constant (secs.)'] = decay_constant
onePresults.save()




# post-4ap stims out of sz trials plot
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (outside sz)'])[0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)

        flu_list, mean_response, decay_constant = aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', y_axis='dff', stims_to_analyze=expobj.stims_out_sz, show=False, quantify=True)
        onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses[
                                                'pkl_list'] == expobj.pkl_path, 'Decay constant (secs.)'] = decay_constant
onePresults.save()



# post-4ap stims during sz trials plot
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)

        flu_list, mean_response, decay_constant = aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', y_axis='dff', stims_to_analyze=expobj.stims_in_sz, show=False, quantify=True)
        onePresults.mean_stim_responses.loc[onePresults.mean_stim_responses[
                                                'pkl_list'] == expobj.pkl_path, 'Decay constant (secs.)'] = decay_constant
onePresults.save()

