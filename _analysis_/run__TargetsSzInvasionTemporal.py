
import sys


print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

"""TODO/GOALS:
x) plotting average traces around time of seizure invasion for all targets across all exps
    - plot also the mean FOV Flu at the bottom

3) plot average stim response vs. (possibly binned?) time to sz invasion for all targets across all exps

4) plot average stim response before and after time of seizure invasion for all targets across all exps - leave for now.

5) plot average stim response of all targets before sz LFP start (-60 to 0secs)


"""
import funcsforprajay.funcs as pj
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

import _alloptical_utils as Utils
from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal as main, \
    TargetsSzInvasionTemporalResults
from _main_.Post4apMain import Post4ap

results: TargetsSzInvasionTemporalResults = TargetsSzInvasionTemporalResults.load()

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
# SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"


# expobj: Post4ap = Utils.import_expobj(prep='PS06', trial='t-013')
# expobj.PhotostimResponsesSLMTargets.adata.var[['stim_start_frame', 'wvfront in sz', 'seizure_num']]


# from _utils_.io import import_expobj
# expobj: Post4ap = import_expobj(prep='PS11', trial='t-011')
#
# plot__exp_sz_lfp_fov(expobj=expobj)


# %% code development zone
# expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
# expobj.TargetsSzInvasionTemporal.plot_szinvasiontime_vs_photostimresponses(fig=None, ax=None)
## binning and plotting zscored photostim responses vs. time ###########################################################

# %% testing the whole analysis pipeline on a singular experiments here

# 0)
# expobj.TargetsSzInvasionTemporal = main(expobj=expobj)


# # 1) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME
# expobj.TargetsSzInvasionTemporal.collect_time_delay_sz_stims(expobj=expobj)
# self = expobj.TargetsSzInvasionTemporal
#
#
# # 2)
# expobj.TargetsSzInvasionTemporal.collect_targets_sz_invasion_traces(expobj=expobj)
#
# # 2.1) plot for targets sz invasion
# # run plot_targets_sz_invasion_meantraces() for this expobj. - see code further below in this file.
#
#
#
#
# # 3) collect photostim responses vs sz invasion time
# expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
# self.plot_szinvasiontime_vs_photostimresponses()
#
# # 5) collect photostim responses vs sz invasion time - zscored
# self.collect_szinvasiontime_vs_photostimresponses_zscored_df(expobj=expobj)
# self.plot_szinvasiontime_vs_photostimresponseszscored()
# data_expobj = self.retrieve__responses_vs_time_to_seizure_SLMTargets_plot2ddensity(expobj=expobj)
#
# # 6) convert sz invasion times to percentile space, sort all data in new percentile space
# data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times = main.convert_responses_sztimes_percentile_space(data=data_expobj)
# main.plot_density_responses_sztimes(data_all, times_to_sz_sorted, scale_percentile_times)
#
# main.plot_lineplot_responses_pctsztimes(percentiles=percentiles, responses_sorted=responses_sorted, scale_percentile_times=scale_percentile_times,
#                                         bin = 10)

# expobj.save()

# %% code deployment zone


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0, skip_trials=main.EXCLUDE_TRIAL)
def run__initTargetsSzInvasionTemporal(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal = main(expobj=expobj)
    expobj.save()


# run__initTargetsSzInvasionTemporal()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=1, allow_rerun=0, skip_trials=main.EXCLUDE_TRIAL)
def run__collect_targets_sz_invasion_traces(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_targets_sz_invasion_traces(expobj=expobj)
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0, skip_trials=main.EXCLUDE_TRIAL)
def run_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_time_delay_sz_stims(expobj=expobj)
    expobj.save()


# run_collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=False, skip_trials=main.EXCLUDE_TRIAL)
def run_check_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_num_pos_neg_szinvasion_stims(expobj=expobj)
    expobj.save()
    expobj.TargetsSzInvasionTemporal.plot_num_pos_neg_szinvasion_stims(**kwargs)

## 1.2) plot mean of seizure invasion Flu traces from all targets for each experiment ##############################
def plot__targets_sz_invasion_meantraces():
    fig, ax = plt.subplots(figsize=[3, 5])

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=1, set_cache=False, skip_trials=main.EXCLUDE_TRIAL)
    def plot_targets_sz_invasion_meantraces(**kwargs):
        expobj: Post4ap = kwargs['expobj']

        fov_mean_trace, to_plot, pre, post = expobj.TargetsSzInvasionTemporal.mean_targets_szinvasion_trace[
                                                 'fov_mean_trace'], \
                                             expobj.TargetsSzInvasionTemporal.mean_targets_szinvasion_trace[
                                                 'mean_trace'], \
                                             expobj.TargetsSzInvasionTemporal.mean_targets_szinvasion_trace[
                                                 'pre_sec'], \
                                             expobj.TargetsSzInvasionTemporal.mean_targets_szinvasion_trace[
                                                 'post_sec']
        invasion_spot = int(pre * expobj.fps)
        to_plot = pj.moving_average(to_plot, n=6)
        fov_mean_trace = pj.moving_average(fov_mean_trace, n=6)
        to_plot_normalize = (to_plot - to_plot[0]) / to_plot[0]
        fov_mean_normalize = (fov_mean_trace - fov_mean_trace[0]) / fov_mean_trace[0]

        x_time = np.linspace(-pre, post, len(to_plot))

        ax.plot(x_time, to_plot_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3,
                label=expobj.t_series_name)
        # ax2.plot(x_time, fov_mean_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3, alpha=0.5, linestyle='--')
        # ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
        ax.scatter(x=0, y=to_plot_normalize[invasion_spot], color='crimson', s=30, zorder=5)

        xticks = [-pre, 0, post]
        # xticks_loc = [xtick*expobj.fps for xtick in [0, pre, pre+post]]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Time (secs) to sz invasion')
        ax.set_ylabel('Flu change (norm.)')
        # ax.set_xlim([np.min(xticks_loc), np.max(xticks_loc)])
        # ax.set_title(f"{expobj.t_series_name}")

    plot_targets_sz_invasion_meantraces()
    fig.suptitle('avg Flu at sz invasion', wrap=True, y=0.96)
    fig.tight_layout(pad=2)
    fig.show()

# run_check_collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=False, skip_trials=main.EXCLUDE_TRIAL)
def run__collect_szinvasiontime_vs_photostimresponses(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
    expobj.save()


# run__collect_szinvasiontime_vs_photostimresponses()


def plot__szinvasiontime_vs_photostimresponses():
    """collects and plots all exps datapoints for szinvasion time vs. photostim responses on one plot"""
    x = []
    y = []

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True, skip_trials=main.EXCLUDE_TRIAL)
    def _return_szinvasiontime_vs_photostimresponses(x, y, **kwargs):
        expobj: Post4ap = kwargs['expobj']
        _x, _y = expobj.TargetsSzInvasionTemporal.return_szinvasiontime_vs_photostimresponses()
        x += _x
        y += _y
        return x, y

    result = _return_szinvasiontime_vs_photostimresponses(x=x, y=y)[-1]
    x, y = result[0], result[1]

    fig, ax = plt.subplots(figsize=[6, 3])
    ax.grid(zorder=0)
    ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.05, ec='white')
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)
    ax.set_title(f"All exps targets time to sz invasion vs. photostim response", wrap=True)
    ax.set_xlabel(f"Time to sz invasion for target (secs)")
    ax.set_ylabel(f"Photostim response for target (dFF)", wrap=True)
    ax.set_ylim([-100, 100])
    fig.tight_layout(pad=2.5)
    fig.show()


# plot__szinvasiontime_vs_photostimresponses()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=False, skip_trials=main.EXCLUDE_TRIAL)
def plot__szinvasiontime_vs_photostimresponses_indivexp(**kwargs):
    """plot all exps datapoints for szinvasion time vs. photostim responses on one plot"""
    expobj: Post4ap = kwargs['expobj']

    fig, ax = plt.subplots(figsize=[3, 3])
    ax.grid(zorder=0)
    expobj.TargetsSzInvasionTemporal.plot_szinvasiontime_vs_photostimresponses(fig=fig, ax=ax)
    ax.set_title(f"{expobj.t_series_name}")
    ax.set_xlabel(f"Time to sz invasion for target")
    ax.set_ylabel(f"Photostim response for target")
    fig.show()


# plot__szinvasiontime_vs_photostimresponses_indivexp()


# zscored photostim responses vs time #################################################################################
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=False, skip_trials=main.EXCLUDE_TRIAL)
def run__collect_szinvasiontime_vs_photostimresponses_zscored(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses_zscored(expobj=expobj)
    expobj.save()


# run__collect_szinvasiontime_vs_photostimresponses_zscored()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True, skip_trials=main.EXCLUDE_TRIAL)
def run__collect_szinvasiontime_vs_photostimresponses_zscored_df(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses_zscored_df(expobj=expobj)
    expobj.save()

# %%
def plot__szinvasiontime_vs_photostimresponseszscored(dataplot='full'):
    """collects and plots all exps datapoints for szinvasion time vs. photostim responses on one plot"""

    # TODO add KDE plot here.

    x = []
    y = []

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False, skip_trials=main.EXCLUDE_TRIAL)
    def _return_szinvasiontime_vs_photostimresponseszscored(x, y, **kwargs):
        expobj: Post4ap = kwargs['expobj']
        _x, _y = expobj.TargetsSzInvasionTemporal.return_szinvasiontime_vs_photostimresponses_zscored()
        x += _x
        y += _y
        return x, y

    # collect data for plotting
    result = _return_szinvasiontime_vs_photostimresponseszscored(x=x, y=y)[-1]

    # positive x (time_to_sz_invasion) values indexes
    x_pos_idx = [idx for idx, val in enumerate(x) if val > 0]
    x_pos = [x[idx] for idx in x_pos_idx]
    y_pos = [y[idx] for idx in x_pos_idx]

    # select data to plot:
    if dataplot == 'full':
        x, y = result[0], result[1]
    elif dataplot == 'pos_only':
        x, y = x_pos, y_pos


    # make scatter plot
    fig, ax = plt.subplots(figsize=[10, 6])
    ax.grid(zorder=0)
    ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.05, ec='white')
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=2.8)
    ax.set_title(f"All exps targets time to sz invasion vs. photostim response", wrap=True)
    ax.set_xlabel(f"Time to sz invasion for target (secs)")
    ax.set_ylabel(f"Photostim response for target ({main.photostim_responses_zscore_type})", wrap=True)
    ax.set_ylim([-4, 4])
    fig.tight_layout(pad=2.5)
    fig.show()

    # make KDE plot
    fig, ax = plt.subplots(figsize=(10,6))
    sns.kdeplot(x=x, y=y, color='red', alpha=0.4, fill=True, fig=fig, ax=ax)
    ax.set_ylim([-4, 4])
    ax.set_title(f"All exps targets time to sz invasion vs. photostim response", wrap=True)
    ax.set_xlabel(f"Time to sz invasion for target (secs)")
    ax.set_ylabel(f"Photostim response for target ({main.photostim_responses_zscore_type})", wrap=True)
    fig.show()


# plot__szinvasiontime_vs_photostimresponseszscored(dataplot='pos_only')


# %%
# binning and plotting zscored photostim responses vs. time ###########################################################
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False, skip_trials=main.EXCLUDE_TRIAL)
def run__retrieve__responses_vs_time_to_seizure_SLMTargets_plot2ddensity(**kwargs):
    """run plotting of responses vs. time as 2d density plot (which will bin the data automatically) for individual experiments"""
    expobj: Post4ap = kwargs['expobj']
    data = expobj.TargetsSzInvasionTemporal.retrieve__responses_vs_time_to_seizure_SLMTargets_plot2ddensity(plot=False,
                                                                                                            expobj=expobj)
    expobj.save()
    return data


# def convert_responses_sztimes_percentile_space(data):
#     """converts sz invasion times to percentile space"""
#     data_all = np.array([[], []]).T
#     for data_ in data:
#         data_all = np.vstack((data_, data_all))
#
#     from scipy.stats import percentileofscore
#
#     times_to_sz = data_all[:, 0]
#     idx_sorted = np.argsort(times_to_sz)
#     times_to_sz_sorted = times_to_sz[idx_sorted]
#     responses_sorted = data_all[:, 1][idx_sorted]
#     s = pd.Series(times_to_sz_sorted)
#     percentiles = s.apply(lambda x: percentileofscore(times_to_sz_sorted, x))
#     scale_percentile_times = {}
#     for pct in range(0, 100):
#         scale_percentile_times[int(pct + 1)] = np.round(np.percentile(times_to_sz_sorted, pct), 0)
#     data_all = np.array([percentiles, responses_sorted]).T
#
#     return data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times


def run__convert_responses_sztimes_percentile_space(data):
    """run converting responses to percentile space"""
    data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times = main.convert_responses_sztimes_percentile_space(
        data=data)
    return data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times


def plot_density_responses_sztimes(data_all, times_to_sz_sorted, scale_percentile_times):
    # plotting density plot for all exps, in percentile space (to normalize for excess of data at times which are closer to zero) - TODO any smoothing?

    bin_size = 2  # secs
    # bins_num = int((max(times_to_sz) - min(times_to_sz)) / bin_size)
    bins_num = [100, 500]

    fig, ax = plt.subplots(figsize=(6, 3))
    pj.plot_hist2d(data=data_all, bins=bins_num, y_label='dFF (zscored)', figsize=(6, 3),
                   x_label='time to seizure (%tile space)',
                   title=f"2d density plot, all exps, 50%tile = {np.percentile(times_to_sz_sorted, 50)}um",
                   y_lim=[-3, 3], fig=fig, ax=ax, show=False)
    ax.axhline(0, ls='--', c='white', lw=1)
    xticks = [1, 25, 50, 57, 75, 100]  # percentile space
    ax.set_xticks(ticks=xticks)
    labels = [scale_percentile_times[x_] for x_ in xticks]
    ax.set_xticklabels(labels)
    ax.set_ylim([-1, 1])
    ax.set_xlabel('time to seizure (secs)')

    fig.show()


# plotting line plot for all datapoints for responses vs. time to seizure
def plot_lineplot_responses_pctsztimes(percentiles, responses_sorted, response_type, scale_percentile_times):
    percentiles_binned = np.round(percentiles)

    bin = 10
    # change to pct% binning
    percentiles_binned = (percentiles_binned // bin) * bin

    d = {'time to seizure (%tile space)': percentiles_binned,
         response_type: responses_sorted}

    df = pd.DataFrame(d)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x='time to seizure (%tile space)', y=response_type, ax=ax)
    ax.set_title(f'responses over time to sz, all exps, normalized to percentile space ({bin}% bins)',
                 wrap=True)
    ax.margins(0.02)
    ax.axhline(0, ls='--', c='orange', lw=1)

    xticks = [1, 25, 50, 57, 75, 100]  # percentile space
    ax.set_xticks(ticks=xticks)
    labels = [scale_percentile_times[x_] for x_ in xticks]
    ax.set_xticklabels(labels)
    ax.set_xlabel('time to seizure (secs)')

    fig.tight_layout(pad=2)
    plt.show()


# %%
if __name__ == '__main__':
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1, skip_trials=main.EXCLUDE_TRIAL)
    def run_misc(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        # expobj.TargetsSzInvasionTemporal.add_slmtargets_time_delay_sz_data(expobj=expobj)
        expobj.TargetsSzInvasionTemporal.collect_targets_sz_invasion_traces(expobj=expobj)
        expobj.save()
    # run_misc()

    # run__initTargetsSzInvasionTemporal()
    # run_collect_time_delay_sz_stims()
    #
    # run__collect_targets_sz_invasion_traces()
    #
    # # plot__targets_sz_invasion_meantraces()
    #
    #
    # # RUNNING BELOW FOR QUANTIFICATION OF PHOTOSTIM RESPONSES VS. TIME DELAY TO SZ INVASION CURRENTLY
    # # fig, ax = plt.subplots(figsize=[3, 3])
    # # run_check_collect_time_delay_sz_stims(fig=fig, ax=ax)
    # # fig.show()
    #
    #
    # run__collect_szinvasiontime_vs_photostimresponses()
    # # plot__szinvasiontime_vs_photostimresponses()
    # plot__szinvasiontime_vs_photostimresponses_indivexp()
    #
    # # run collecting and plotting zscored photostim responses vs. time delay to sz invasion
    # run__collect_szinvasiontime_vs_photostimresponses_zscored()
    # run__collect_szinvasiontime_vs_photostimresponses_zscored_df()
    #
    # plot__szinvasiontime_vs_photostimresponseszscored()
    #
    # results.data = run__retrieve__responses_vs_time_to_seizure_SLMTargets_plot2ddensity()
    # # results.save_results()
    # results.data_all, results.percentiles, results.responses_sorted, results.times_to_sz_sorted, \
    #     results.scale_percentile_times = run__convert_responses_sztimes_percentile_space(data=results.data)
    # results.save_results()
    #
    #
    # main.plot_density_responses_sztimes(results.data_all, results.times_to_sz_sorted, results.scale_percentile_times,
    #                                     photostim_responses_zscore_type=main.photostim_responses_zscore_type)
    # plot_lineplot_responses_pctsztimes(percentiles=results.percentiles, responses_sorted=results.responses_sorted,
    #                                    response_type=main.photostim_responses_zscore_type,
    #                                    scale_percentile_times=results.scale_percentile_times)


    # plot average stim response vs. (possibly binned?) time to sz invasion for all targets across all exps
    # bin_width, sztemporalinv, num, avg_responses, conf_int = main.collect__binned__szinvtime_v_responses()  # binsize = 3 secs
    # results.binned__time_vs_photostimresponses = {'bin_width_sec': bin_width, 'sztemporal_bins': sztemporalinv,
    #                                               'num_points_in_bin': num,
    #                                               'avg_photostim_response_in_bin': avg_responses,
    #                                               '95conf_int': conf_int}
    #
    # results.save_results()
    main.plot__responses_v_szinvtemporal_no_normalization(results=results)

    # results.range_of_sz_invasion_time = [-1, -1, -1]
    # main.saveclass()
    pass

# # %%
#
# import sys
#
#
# print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
#
#
#
# import funcsforprajay.funcs as pj
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
#
# import _alloptical_utils as Utils
# from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal as main, \
#     TargetsSzInvasionTemporalResults
# from _main_.Post4apMain import Post4ap
#
# results: TargetsSzInvasionTemporalResults = TargetsSzInvasionTemporalResults.load()

# %%
# collect and plot time vs photostim responses data
# expobj: Post4ap = Utils.import_expobj(prep='PS06', trial='t-013')



#
# # %%
#
# import funcsforprajay.funcs as pj
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats
#
# """plotting of binned responses over time to sz invasion for each target+stim as a step function, with heatmap showing # of datapoints"""
# # sztemporalinv_bins = results.binned__distance_vs_photostimresponses['sztemporal_bins']
# sztemporalinv = results.binned__time_vs_photostimresponses['sztemporal_bins']
# avg_responses = results.binned__time_vs_photostimresponses['avg_photostim_response_in_bin']
# conf_int = results.binned__time_vs_photostimresponses['95conf_int']
# num2 = results.binned__time_vs_photostimresponses['num_points_in_bin']
#
# conf_int_sztemporalinv = pj.flattenOnce([[sztemporalinv[i], sztemporalinv[i + 1]] for i in range(len(sztemporalinv) - 1)])
# conf_int_values_neg = pj.flattenOnce([[val, val] for val in conf_int[1:, 0]])
# conf_int_values_pos = pj.flattenOnce([[val, val] for val in conf_int[1:, 1]])
#
# fig, axs = plt.subplots(figsize=(6, 6), nrows=2, ncols=1)
# # ax.plot(sztemporalinv[:-1], avg_responses, c='cornflowerblue', zorder=1)
# ax = axs[0]
# ax2 = axs[1]
# ax.step(sztemporalinv, avg_responses, c='cornflowerblue', zorder=2)
# # ax.fill_between(x=(sztemporalinv-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
# ax.fill_between(x=conf_int_sztemporalinv, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
#                 zorder=0)
# # ax.scatter(sztemporalinv[:-1], avg_responses, c='orange', zorder=4)
# ax.set_ylim([-0.5, 0.8])
# ax.set_title(
#     f'photostim responses vs. distance to sz wavefront (binned every {results.binned__time_vs_photostimresponses["bin_width_sec"]}sec)',
#     wrap=True)
# ax.set_xlabel('time to sz inv (secs)')
# ax.set_ylabel(main.photostim_responses_zscore_type)
# ax.margins(0)
#
# pixels = [np.array(num2)] * 10
# ax2.imshow(pixels, cmap='Greys', vmin=-5, vmax=150, aspect=0.1)
# # ax.show()
#
# fig.tight_layout(pad=1)
# fig.show()


