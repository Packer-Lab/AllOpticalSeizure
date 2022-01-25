#%% DATA ANALYSIS FOR ONE-P PHOTOSTIM EXPERIMENTS
import os

import numpy as np

import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot



#  ###### IMPORT pkl file containing data in form of expobj
trial = 't-012'
prep = 'PS11'
date = '2021-01-24'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
if os.path.exists(pkl_path):
    expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)


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
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5,3), linewidth=0.5, downsample=True, sz_markings=False, color='black', ylims=[0,5])
# aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(10,2), linewidth=1.2, downsample=True, sz_markings=False, ylims=[-1,5], color='slategray')

aoplot.plot_lfp_stims(expobj, x_axis='Time', figsize=(10,2), ylims=[-1,5])




# %%
date = '2021-01-24'

expobj, _ = aoutils.import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5,3), linewidth=0.5, downsample=True, sz_markings=False, color='black',
                     ylims=[-4,1], xlims=[110*expobj.paq_rate, 210*expobj.paq_rate])

if 'pre' in expobj.metainfo['exptype']:
    aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color='skyblue',
                                     y_axis='dff', quantify=False, figsize=[3, 3])

    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=expobj.stim_start_frames,
                                     title='Avg. run_pre4ap_trials stims LFP')

expobj, _ = aoutils.import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5/100*150,3), linewidth=0.5, downsample=True, sz_markings=False, color='black',
                     ylims=[0,5], xlims=[10*expobj.paq_rate, 160*expobj.paq_rate])

if 'post' in expobj.metainfo['exptype']:
    aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color='skyblue', stims_to_analyze=expobj.stims_out_sz,
                                     y_axis='dff', quantify=False, figsize=[3, 3], title='Avg. interictal stims LFP')

    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=expobj.stims_out_sz,
                                     title='Avg. out sz stims LFP')

    aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color='skyblue', stims_to_analyze=expobj.stims_in_sz,
                                     y_axis='dff', quantify=False, figsize=[3, 3], title='Avg. ictal stims LFP')

    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=expobj.stims_in_sz,
                                     title='Avg. in sz stims LFP')

# %% classifying stims as in or out of seizures

seizures_lfp_timing_matarray = '/home/pshah/mnt/qnap/Analysis/%s/%s/paired_measurements/%s_%s_%s.mat' % (date, prep, date, expobj.metainfo['animal prep.'], trial[-3:])

expobj.collect_seizures_info(seizures_lfp_timing_matarray=seizures_lfp_timing_matarray,
                             discard_all=False)

# %% MEASURING DFF RESPONSE MAGNITUDE VS. PRE-STIM CA2+ FLU OF THE FOV

from OnePhotonStimMain import OnePhotonStimFuncs as onepfuncs
from OnePhotonStimMain import OnePhotonStimPlots as onepplots


onepfuncs._collectPhotostimResponses()
onepfuncs._collectPreStimFluAvgs()

onepplots.plotPrestimF_photostimFlu()

# for trial in ['t-006', 't-008', 't-009', 't-010']:
#     expobj, _ = OnePhotonStimFuncs.import_1pexobj(pkl_path=f'/home/pshah/mnt/qnap/Analysis/2021-02-01/2021-02-01_{trial}/2021-02-01_{trial}.pkl')
#     expobj.analysis_save_path = f'/home/pshah/mnt/qnap/Analysis/2021-02-01/PS16/2021-02-01_{trial}/'
#     expobj.save()


# expobj, _ = OnePhotonStimFuncs.import_1pexobj(prep='PS11', trial='t-017')

# pj.make_general_scatter([expobj.pre_stim_flu_list], [expobj.decay_constants], s=50, colors=['red'], alpha=0.8,
#                         x_label='photostim_flu_responses', y_label='decay_constants',
#                         ax_titles=['Decay constants vs. Pre-stim Flu'])


