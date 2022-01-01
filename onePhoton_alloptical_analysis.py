#%% DATA ANALYSIS FOR ONE-P PHOTOSTIM EXPERIMENTS
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot



#  ###### IMPORT pkl file containing data in form of expobj
trial = 't-012'
prep = 'PS11'
date = '2021-01-24'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
if os.path.exists(pkl_path):
    expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
# expobj, experiment = aoutils.import_expobj(prep=prep, trial=trial)


# %% # look at the average Ca Flu trace pre and post stim, just calculate the average of the whole frame and plot as continuous timeseries
# - this approach should also allow to look at the stims that give rise to extended seizure events where the Ca Flu stays up

# # EXCLUDE CERTAIN STIM START FRAMES
# expobj.stim_start_frames = [frame for frame in expobj.stim_start_frames if 4000 > frame or frame > 5000]
# expobj.stim_end_frames = [frame for frame in expobj.stim_end_frames if 4000 > frame or frame > 5000]
# expobj.stim_start_times = [time for time in expobj.stim_start_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_end_times = [time for time in expobj.stim_end_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_duration_frames = int(np.mean(
#     [expobj.stim_end_frames[idx] - expobj.stim_start_frames[idx] for idx in range(len(expobj.stim_start_frames))]))

aoplot.plotMeanRawFluTrace(expobj, stim_span_color='white', x_axis='frames')
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(10,2), linewidth=1.2, downsample=True, sz_markings=False, ylims=[-1,5], color='slategray')

aoplot.plot_lfp_stims(expobj, x_axis='Time', figsize=(10,2), ylims=[-1,5])

aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff', quantify=True)

if 'pre' in expobj.metainfo['exptype']:
    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3,3), shrink_text=0.8, stims_to_analyze=expobj.stim_start_frames,
                                     title='Avg. run_pre4ap_trials stims LFP')

if 'post' in expobj.metainfo['exptype']:
    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3,3), shrink_text=0.8, stims_to_analyze=expobj.stims_out_sz,
                                     title='Avg. out sz stims LFP')

    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3,3), shrink_text=0.8, stims_to_analyze=expobj.stims_in_sz,
                                     title='Avg. in sz stims LFP')

# %% classifying stims as in or out of seizures

seizures_lfp_timing_matarray = '/home/pshah/mnt/qnap/Analysis/%s/%s/paired_measurements/%s_%s_%s.mat' % (date, prep, date, expobj.metainfo['animal prep.'], trial[-3:])

expobj.collect_seizures_info(seizures_lfp_timing_matarray=seizures_lfp_timing_matarray,
                             discard_all=False)
