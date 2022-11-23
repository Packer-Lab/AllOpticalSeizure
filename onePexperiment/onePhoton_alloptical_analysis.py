#%% DATA ANALYSIS FOR ONE-P PHOTOSTIM EXPERIMENTS - trying to mirror this code with the jupyter notebook for one P stim analysis
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

# import alloptical_utils_pj as aoutils
from _utils_ import alloptical_plotting as aoplot
from _utils_.io import import_expobj, import_1pexobj
from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs, OnePhotonStimResults

from onePexperiment.OnePhotonStimMain import OnePhotonStimPlots as onepplots, OnePhotonStim

Results: OnePhotonStimResults = OnePhotonStimResults.load()

# #  ###### IMPORT pkl file containing data in form of expobj
# trial = 't-008'
# prep = 'PS17'
# date = '2021-01-24'
# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
# if os.path.exists(pkl_path):
#     expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)

date = '2021-01-24'
# pre4ap
# expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
# aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5,3), linewidth=0.5, downsample=True, sz_markings=False, color='black',
#                      ylims=[-4,1], xlims=[110*expobj.paq_rate, 210*expobj.paq_rate])

# post4ap trial
# expobj, _ = import_1pexobj(prep='PS07', trial='t-012', verbose=False)
# aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(15,3), linewidth=0.5, downsample=True, sz_markings=True, color='black')
# aoplot.plotMeanRawFluTrace(expobj, stim_span_color='white', x_axis='Time')


# print(expobj.fov_trace_shutter_blanked_dff_pre_norm)

# self = expobj

# %% 1.0) ## measuring PRE-STIM CA2+ AVG FLU vs. DFF RESPONSE MAGNITUDE, DECAY CONSTANT of the fov

# collection of: photostims relative to seizure timing
OnePhotonStimAnalysisFuncs.collectPhotostimResponses_PrePostSz(resultsobj=Results, ignore_cache=True)


# seizure excluded responses collection
OnePhotonStimAnalysisFuncs.collectPhotostimResponses_szexcluded(resultsobj=Results, ignore_cache=True)


OnePhotonStimAnalysisFuncs.collectPhotostimResponseIndivual(resultsobj=Results, run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False)


# %% 2.1) PLOT - time to seizure onset vs. pre-stim Flu

fig, ax = plt.subplots(figsize=(2.5,4))
onepplots.plotTimeToOnset_preStimFlu(fig=fig, ax=ax, run_pre4ap_trials=True, run_post4ap_trials=False, x_lim=[-4, 5])
fig.show()

fig, ax = plt.subplots(figsize=(4,4))
onepplots.plotTimeToOnset_preStimFlu(fig=fig, ax=ax, run_pre4ap_trials=False, run_post4ap_trials=True)
fig.show()

# %% 2.2) PLOT - time to seizure onset vs. photostim response Flu

fig, ax = plt.subplots(figsize=(2.5,4))
onepplots.plotTimeToOnset_photostimResponse(fig=fig, ax=ax, run_pre4ap_trials=True, run_post4ap_trials=False, x_lim=[-4, 5])
fig.show()

fig, ax = plt.subplots(figsize=(4,4))
onepplots.plotTimeToOnset_photostimResponse(fig=fig, ax=ax, run_pre4ap_trials=False, run_post4ap_trials=True, x_lim=[-80, 80])
fig.show()




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

expobj = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5,3), linewidth=0.5, downsample=True, sz_markings=False, color='black',
                     ylims=[-4,1], xlims=[110*expobj.paq_rate, 210*expobj.paq_rate])

if 'pre' in expobj.metainfo['exptype']:
    aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color='skyblue',
                                     y_axis='dff', quantify=False, figsize=[3, 3])

    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=expobj.stim_start_frames,
                                     title='Avg. run_pre4ap_trials stims LFP')

expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
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

# %% 0)

"""
photostim results are being collected in the .photostim_results dataframe.

"""

# %% 1.0) ## measuring PRE-STIM CA2+ AVG FLU vs. DFF RESPONSE MAGNITUDE, DECAY CONSTANT of the fov

OnePhotonStimAnalysisFuncs.collectPhotostimResponses(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False)
OnePhotonStimAnalysisFuncs.collectPreStimFluAvgs(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False)


# %% 1.1) ## plotting PRE-STIM CA2+ AVG FLU VS. DFF RESPOSNSE MAGNITUDE, DECAY CONSTANT of the fov
fig, ax = plt.subplots(figsize=[4.5,4])
onepplots.plotPrestimF_photostimFlu(fig=fig, ax=ax, run_pre4ap_trials=True, run_post4ap_trials=False, ignore_cache=True)
ax.set_title('(baseline: gray)', wrap=True)
fig.show()


fig, ax = plt.subplots(figsize=[4.5,4])
onepplots.plotPrestimF_photostimFlu(fig=fig, ax=ax, run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
ax.set_title('(ictal: purple, inter-ictal: green)', wrap=True)
fig.show()



fig, ax = plt.subplots(figsize=[4.5,4])
onepplots.plotPrestimF_decayconstant(fig=fig, ax=ax, run_pre4ap_trials=True, run_post4ap_trials=False, ignore_cache=True,
                                     run_trials=[], skip_trials=[])
ax.set_title('(baseline: gray)', wrap=True)
fig.show()

fig, ax = plt.subplots(figsize=[4.5,4])
onepplots.plotPrestimF_decayconstant(fig=fig, ax=ax, run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True,
                                     run_trials=[], skip_trials=[])
ax.set_title('(ictal: purple, inter-ictal: green)', wrap=True)
fig.show()


# %% 2.0) collect time to seizure onset - add to .photostim_results dataframe

OnePhotonStimAnalysisFuncs.collectTimeToSzOnset(ignore_cache=False)




# %% 3.0) radial plot of photostim Flu response binned by 1sec post stimulation, where 0 = photostim time

exp_sz_occurrence = OnePhotonStimAnalysisFuncs.collectSzOccurrenceRelativeStim()

bin_width = int(0.5 * expobj.fps)
period = len(np.arange(0, (expobj.stim_interval_fr / bin_width))[:-1])
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width))[:-1] / period

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
for exp in exp_sz_occurrence:
    plot = exp
    ax.bar(theta, plot, width=(2 * np.pi) / period , bottom=0.0, alpha=0.5)
    # ax.set_rmax(1.1)
    # ax.set_rticks([0.5, 1])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_title("sz occurrence", va='bottom')
fig.show()





















# %% developing code below....

# plot the meanRawFluTrace (maybe dFF normalize later too) along the radial axis...
# self = expobj

frames_lengths = np.arange(0 - self.shutter_interval_fr, (self.stim_start_frames[-1] - self.stim_start_frames[0] + self.stim_interval_fr - self.shutter_interval_fr))  #: frame indexes zero'd at the first photostim frame
# frames_lengths = np.arange(0, (self.stim_start_frames[-1] - self.stim_start_frames[0] + self.stim_interval_fr))  #: frame indexes zero'd at the first photostim frame
theta = (2 * np.pi) * (frames_lengths / self.stim_interval_fr)

print((frames_lengths/self.stim_interval_fr)[:100])
print(len(expobj.experiment_frames))

print(expobj.sz_occurrence_stim_intervals)
# fig, ax = plt.subplots(figsize = (5, 5))
# ax.plot(frames_lengths, expobj.fov_trace_shutter_blanked_dff[expobj.experiment_frames])
# fig.show()

# time_length = np.arange(0, ((self.stim_start_frames[-1] / self.fps) + self.stim_interval), 1 / self.fps)
# time_length = np.linspace(0, ((self.stim_start_frames[-1] / self.fps) + self.stim_interval), len(self.experiment_frames))



# %% linear plot
fig, ax = plt.subplots(figsize = (10,3))
# ax.scatter(theta, expobj.fov_trace_shutter_blanked_dff[expobj.experiment_frames], c=theta)
ax.plot(theta, expobj.sz_liklihood_fr[expobj.experiment_frames], color='red')
fig.show()


# %% radial plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, expobj.sz_liklihood_fr[expobj.experiment_frames], alpha=0.8)
ax.set_rmax(1.1)
ax.set_rticks([0.5, 1])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_title("fov_trace_shutter_blanked", va='bottom')
fig.show()



# %% plotting sz occurrence

bin_width = int(0.5 * self.fps)
period = len(np.arange(0, (self.stim_interval_fr / bin_width))[:-1])
theta = (2 * np.pi) * np.arange(0, (self.stim_interval_fr / bin_width))[:-1] / period
plot = expobj.sz_occurrence_stim_intervals

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(theta, plot, alpha=0.8)
ax.bar(theta, plot, width=(2 * np.pi) / period , bottom=0.0, alpha=0.5)
ax.set_rmax(1.1)
ax.set_rticks([0.5, 1])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_title("sz occurrence", va='bottom')
fig.show()



# %% plot
plt.plot(expobj.meanRawFluTrace[expobj.interictal_fr])
plt.show()


# %% normalizing stim plots

slic = [self.shutter_start_frames[0][0] - self.stim_interval_fr,
        (self.shutter_start_frames[0][-1] + self.stim_interval_fr)]
if slic[0] < 0: slic[0] = self.shutter_start_frames[0][0]
new_trace = self.fov_trace_shutter_blanked[slic[0]: slic[1]]

norm_mean = np.mean(self.meanRawFluTrace[self.interictal_fr])

for i, fr in enumerate(self.shutter_start_frames[0]):
    pre_slice = [fr - self.stim_interval_fr, fr]
    if pre_slice[0] < 0: pre_slice[0] = 0
    pre_mean = np.mean(self.fov_trace_shutter_blanked[pre_slice[0]: pre_slice[1]])
    new_trace[fr - self.stim_interval_fr: fr + self.stim_interval_fr] -= norm_mean
    # new_trace[fr - self.stim_interval_fr: fr + self.stim_interval_fr] /= pre_mean

plt.figure(figsize=(30, 3))
plt.plot(new_trace)
plt.show()








# %% trial plotting
frames_lengths = np.arange(0 - self.shutter_interval_fr, (self.stim_start_frames[-1] - self.stim_start_frames[0] + self.stim_interval_fr - self.shutter_interval_fr))  #: frame indexes zero'd at the first photostim frame
# frames_lengths = np.arange(0, (self.stim_start_frames[-1] - self.stim_start_frames[0] + self.stim_interval_fr))  #: frame indexes zero'd at the first photostim frame
theta = (2 * np.pi) * (frames_lengths / self.stim_interval_fr)


fake_signal = np.array([0] * len(frames_lengths), dtype='float')
for i, num in enumerate(np.round(frames_lengths,1)):
    # if round(num % self.stim_interval_fr, 1) == int(self.stim_interval_fr / 2):
    if round(num % self.stim_interval_fr, 1) == 0:
        fake_signal[i] = 5
    if round(num % self.stim_interval_fr, 1) == 0.5 * self.stim_interval_fr:
        fake_signal[i] = np.random.randint(0, 5, 1)

fig, ax = plt.subplots(figsize=(30, 2))
ax.plot(frames_lengths, fake_signal)
fig.show()


theta = (2 * np.pi) * (frames_lengths / self.stim_interval_fr)


# %% polar plot with bars!
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.bar(theta, fake_signal, width=(2 * np.pi) * (0.5 / self.stim_interval), bottom=0.0, alpha=0.5)
# ax.plot(theta, fake_signal)
ax.set_rmax(5)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_title("fake signal", va='bottom')
plt.show()

# %%
fake_signal = np.arange(0, 1, 0.01)
theta = 2 * np.pi * fake_signal

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, fake_signal)
ax.set_rmax(3)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

