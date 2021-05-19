#%% DATA ANALYSIS FOR ONE-P PHOTOSTIM EXPERIMENTS
import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
from utils.paq_utils import *

###### IMPORT pkl file containing data in form of expobj
trial = 't-008'
date = '2021-02-02'
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date,
                                           pkl_path='/home/pshah/mnt/qnap/Analysis/2021-02-02/PS17/%s_%s/%s_%s.pkl' % (date, trial, date, trial))


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
aoplot.plotLfpSignal(expobj, x_axis='paq')

aoplot.plot_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff')

# %%

import matplotlib.ticker as mticker

def plot_lfp_1pstim_avg_trace(expobj, title='Average LFP peri- stims', individual_traces=False, x_axis='time', pre_stim=1.0, post_stim=5.0,
                              optoloopback: bool = False):
    stim_duration = int(np.mean([expobj.stim_end_times[idx] - expobj.stim_start_times[idx] for idx in range(len(expobj.stim_start_times))]) + 0.01*expobj.paq_rate)
    pre_stim = pre_stim  # seconds
    post_stim = post_stim  # seconds
    fig, ax = plt.subplots()
    x = [expobj.lfp_signal[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim in expobj.stim_start_times]
    x_ = np.mean(x, axis=0)
    ax.plot(x_, color='black', zorder=3, linewidth=1.75)

    if individual_traces:
        # individual traces
        for trace in x:
            ax.plot(trace, color='steelblue', zorder=1, alpha=0.25)
            # ax.axvspan(int(pre_stim * expobj.paq_rate),
            #            int(pre_stim * expobj.paq_rate) + stim_duration,
            #            edgecolor='powderblue', zorder=1, alpha=0.3)
        ax.axvspan(int(pre_stim * expobj.paq_rate),
                   int(pre_stim * expobj.paq_rate) + stim_duration,
                   color='skyblue', zorder=1, alpha=0.7)

    else:
        # plot standard deviation of the traces array as a span above and below the mean
        std_ = np.std(x, axis=0)
        ax.fill_between(x=range(len(x_)), y1=x_ + std_, y2=x_ - std_, alpha=0.3, zorder=2, color='steelblue')
        ax.axvspan(int(pre_stim * expobj.paq_rate),
                   int(pre_stim * expobj.paq_rate) + stim_duration, color='skyblue', zorder=1, alpha=0.7)


    if optoloopback:
        ax2 = ax.twinx()
        if not hasattr(expobj, 'opto_loopback'):
            print('loading', expobj.paq_path)

            paq, _ = paq_read(expobj.paq_path, plot=True)
            expobj.paq_rate = paq['rate']

            # find voltage channel and save as lfp_signal attribute
            voltage_idx = paq['chan_names'].index('opto_loopback')
            expobj.opto_loopback = paq['data'][voltage_idx]
        else:
            pass
        x = [expobj.opto_loopback[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim
             in expobj.stim_start_times]
        y_avg = np.mean(x, axis=0)
        ax2.plot(y_avg, color='lightgray', zorder=3, linewidth=1.75)

    if x_axis == 'time':
        # change x axis ticks to seconds
        label_format = '{:,.2f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = round(item / expobj.paq_rate, 2)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(x) for x in labels])
        ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('paq clock')
    ax.set_ylabel('Voltage')
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    plt.show()

plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75,
                          optoloopback=True)



# %% make downsampled tiff for viewing raw data

trial = 't-012'
date = '2021-01-19'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

