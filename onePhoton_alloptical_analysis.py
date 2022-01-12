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
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(10,2), linewidth=1.2, downsample=True, sz_markings=False, color='slategray')
# aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(10,2), linewidth=1.2, downsample=True, sz_markings=False, ylims=[-1,5], color='slategray')

aoplot.plot_lfp_stims(expobj, x_axis='Time', figsize=(10,2), ylims=[-1,5])

aoplot.plot_flu_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff', quantify=True)

if 'pre' in expobj.metainfo['exptype']:
    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3,3), shrink_text=0.8, stims_to_analyze=expobj.stim_start_frames,
                                     title='Avg. run_pre4ap_trials stims LFP')

    aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=True,
                                     optoloopback=True, figsize=(3,3), shrink_text=0.8,
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

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_lfp_1pstim_avg_trace(expobj, title='Average LFP peri- stims', individual_traces=False, x_axis='time', pre_stim=1.0, post_stim=5.0,
                              optoloopback: bool = False, stims_to_analyze: list = None,  write_full_text: bool = False,
                              fig=None, ax=None, **kwargs):
    # fig, ax = plt.subplots()
    # if there is a fig and ax provided in the function call then use those, otherwise start anew
    # if 'fig' in kwargs.keys():
    #     fig = kwargs['fig']
    #     ax = kwargs['ax']
    # else:
    #     if 'figsize' in kwargs.keys():
    #         fig, ax = plt.subplots(figsize=kwargs['figsize'])
    #     else:
    #         fig, ax = plt.subplots()

    stims_to_analyze = expobj.stims_out_sz
    # stims_to_analyze = expobj.stim_start_frames

    stim_duration = int(np.mean([expobj.stim_end_times[idx] - expobj.stim_start_times[idx] for idx in range(len(expobj.stim_start_times))]) + 0.01*expobj.paq_rate)
    pre_stim = 1.0  # seconds
    post_stim = 1.0  # seconds


    if stims_to_analyze is None:
        # stims_to_analyze = expobj.stim_start_frames
        stims_to_analyze_paq = expobj.stim_start_times
    else:
        # stims_to_analyze_paq = [expobj.convert_frames_to_paqclock(frame) for frame in stims_to_analyze]
        stims_to_analyze_paq = [expobj.stim_start_times[expobj.stim_start_frames.index(stim_frame)] for stim_frame in stims_to_analyze]
        # stims_to_analyze_paq = [expobj.frame_clock_actual[frame] for frame in stims_to_analyze]

    f, ax = plt.subplots()
    stim = stims_to_analyze_paq[0]
    x = expobj.lfp_signal[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)]
    ax.plot(x, color='black', zorder=3, linewidth=1.75)
    f.show()

    x = [expobj.lfp_signal[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim in stims_to_analyze_paq]
    x_ = np.mean(x, axis=0)
    ax.plot(x_, color='black', zorder=3, linewidth=1.75)

    if 'ylims' in kwargs.keys() and kwargs['ylims'] is not None:
        ax.set_ylim([kwargs['ylims'][0], kwargs['ylims'][1]])
    else:
        ax.set_ylim([np.mean(x_) - 2.5, np.mean(x_) + 2.5])
    ax.margins(0)

    if individual_traces:
        # individual traces
        for trace in x:
            ax.plot(trace, color='steelblue', zorder=1, alpha=0.25)
            # ax.axvspan(int(pre_stim_sec * expobj.paq_rate),
            #            int(pre_stim_sec * expobj.paq_rate) + stim_duration,
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

    if 'shrink_text' in kwargs.keys():
        shrink_text = kwargs['shrink_text']
        print(shrink_text)
    else:
        shrink_text = 0.7
        print(shrink_text)


    if optoloopback:
        ax2 = ax.twinx()
        if not hasattr(expobj, 'opto_loopback'):
            print('loading', expobj.paq_path)

            paq, _ = paq_read(expobj.paq_path, plot=False)
            expobj.paq_rate = paq['rate']

            # find voltage channel and save as lfp_signal attribute
            voltage_idx = paq['chan_names'].index('opto_loopback')
            expobj.opto_loopback = paq['data'][voltage_idx]
            #expobj.save()
        else:
            pass
        x = [expobj.opto_loopback[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim
             in expobj.stim_start_times]
        y_avg = np.mean(x, axis=0)
        ax2.plot(y_avg, color='royalblue', zorder=3, linewidth=1.75)
        if write_full_text:
            ax2.text(0.98, 0.12, 'Widefield LED TTL',
                     transform=ax.transAxes, fontweight='bold', horizontalalignment='right',
                     color='royalblue', fontsize=10*shrink_text)
        # ax2.set_ylabel('Widefield LED TTL', color='royalblue', fontweight='bold')
        ax2.yaxis.set_tick_params(right=False,
                                  labelright=False)
        ax2.set_ylim([-3, 30])
        ax2.margins(0)



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

    # add title
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']),
    fontsize=10*shrink_text)

    ax.text(0.98, 0.97, '%s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']),
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, fontweight='bold',
            color='black', fontsize=10 * shrink_text)