import sys;

from _analysis_._ClassExpSeizureAnalysis import plot__exp_sz_lfp_fov

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

"""TODO/GOALS:
x) plotting average traces around time of seizure invasion for all targets across all exps
    - plot also the mean FOV Flu at the bottom
3) plot average stim response before and after time of seizure invasion for all targets across all exps
4) plot average stim response vs. time to sz invasion for all targets across all exps

"""
import funcsforprajay.funcs as pj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import _alloptical_utils as Utils
from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal as MAIN, \
    TargetsSzInvasionTemporalResults
from _main_.Post4apMain import Post4ap

RESULTS: TargetsSzInvasionTemporalResults = TargetsSzInvasionTemporalResults.load()

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
# SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"


# expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
# expobj = Utils.import_expobj(load_backup_path='/home/pshah/mnt/qnap/Analysis/2020-12-18/RL108/2020-12-18_t-013/backups/2020-12-18_RL108_t-013.pkl')

# expobj.PhotostimResponsesSLMTargets.adata.var[['stim_start_frame', 'wvfront in sz', 'seizure_num']]

# %%
from _utils_.io import import_expobj


# expobj = import_expobj(prep='PS11', trial='t-011')
#
# plot__exp_sz_lfp_fov(expobj=expobj)


# %% code development zone
# expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
# expobj.TargetsSzInvasionTemporal.plot_szinvasiontime_vs_photostimresponses(fig=None, ax=None)
## binning and plotting zscored photostim responses vs. time ###########################################################
# TODO - MAIN WORK RIGHT NOW:


# .22/03/07
# - run through the whole temporal sz invasion pipeline to come up with the final plots after updating sz invasions for the two experiments


# .22/03/07
# - bring back the seizure invasion marking by certain amounts for two experiments
#    - need to determine which exps first: PS04 t-018 move forward by 2secs and PS11 t-011 move forward by 5secs
#    - then, bring back the marking for those and replot avg invasion trace
#           - this looks great.


# %% code deployment zone


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
def run__initTargetsSzInvasionTemporal(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal = MAIN(expobj=expobj)
    expobj.save()


# run__initTargetsSzInvasionTemporal()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=1, allow_rerun=0)
def run__collect_targets_sz_invasion_traces(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_targets_sz_invasion_traces(expobj=expobj)
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1)
def run_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_time_delay_sz_stims(expobj=expobj)
    expobj.save()


# run_collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def run_check_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_num_pos_neg_szinvasion_stims(expobj=expobj)
    expobj.save()
    expobj.TargetsSzInvasionTemporal.plot_num_pos_neg_szinvasion_stims(**kwargs)


# run_check_collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
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

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
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


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
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
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def run__collect_szinvasiontime_vs_photostimresponses_zscored(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses_zscored(expobj=expobj,
                                                                                          zscore_type='dFF (zscored) (interictal)')
    expobj.save()


# run__collect_szinvasiontime_vs_photostimresponses_zscored()


def plot__szinvasiontime_vs_photostimresponseszscored():
    """collects and plots all exps datapoints for szinvasion time vs. photostim responses on one plot"""
    x = []
    y = []

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False)
    def _return_szinvasiontime_vs_photostimresponseszscored(x, y, **kwargs):
        expobj: Post4ap = kwargs['expobj']
        _x, _y = expobj.TargetsSzInvasionTemporal.return_szinvasiontime_vs_photostimresponses_zscored()
        x += _x
        y += _y
        return x, y

    result = _return_szinvasiontime_vs_photostimresponseszscored(x=x, y=y)[-1]
    x, y = result[0], result[1]

    fig, ax = plt.subplots(figsize=[6, 3])
    ax.grid(zorder=0)
    ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.05, ec='white')
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=2.8)
    ax.set_title(f"All exps targets time to sz invasion vs. photostim response", wrap=True)
    ax.set_xlabel(f"Time to sz invasion for target (secs)")
    ax.set_ylabel(f"Photostim response for target (dFF zscored)", wrap=True)
    ax.set_ylim([-1.5, 1.5])
    fig.tight_layout(pad=2.5)
    fig.show()


# plot__szinvasiontime_vs_photostimresponseszscored()


# binning and plotting zscored photostim responses vs. time ###########################################################

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def run__collect_szinvasiontime_vs_photostimresponses_zscored_df(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses_zscored_df(expobj=expobj,
                                                                                             zscore_type='dFF (zscored) (interictal)')
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False)
def run__plot_responses_vs_time_to_seizure_SLMTargets_2ddensity(**kwargs):
    """run plotting of responses vs. time as 2d density plot (which will bin the data automatically) for individual experiments"""
    expobj: Post4ap = kwargs['expobj']
    data = expobj.TargetsSzInvasionTemporal.plot_responses_vs_time_to_seizure_SLMTargets_2ddensity(plot=False,
                                                                                                   expobj=expobj)
    expobj.save()
    return data




def convert_responses_sztimes_percentile_space(data):
    """converts sz invasion times to percentile space"""
    data_all = np.array([[], []]).T
    for data_ in data:
        data_all = np.vstack((data_, data_all))

    from scipy.stats import percentileofscore

    times_to_sz = data_all[:, 0]
    idx_sorted = np.argsort(times_to_sz)
    times_to_sz_sorted = times_to_sz[idx_sorted]
    responses_sorted = data_all[:, 1][idx_sorted]
    s = pd.Series(times_to_sz_sorted)
    percentiles = s.apply(lambda x: percentileofscore(times_to_sz_sorted, x))
    scale_percentile_times = {}
    for pct in range(0, 100):
        scale_percentile_times[int(pct + 1)] = np.round(np.percentile(times_to_sz_sorted, pct), 0)
    data_all = np.array([percentiles, responses_sorted]).T

    return data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times


def run__convert_responses_sztimes_percentile_space(data):
    """run converting responses """
    data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times = convert_responses_sztimes_percentile_space(
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
    sns.lineplot(data=df, x='time to seizure (%tile space)', y='dFF (zscored)', ax=ax)
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
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1,
                                    run_trials=['PS04 t-018', 'PS11 t-011'])
    def run_misc(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        # expobj.TargetsSzInvasionTemporal.add_slmtargets_time_delay_sz_data(expobj=expobj)
        expobj.TargetsSzInvasionTemporal.collect_targets_sz_invasion_traces(expobj=expobj)
        expobj.save()


    # run_misc()

    # run__initTargetsSzInvasionTemporal()

    # run__collect_targets_sz_invasion_traces()
    # MAIN.plot__targets_sz_invasion_meantraces()

    # run_collect_time_delay_sz_stims()

    # RUNNING BELOW FOR ANALYSIS CURRENTLY
    # fig, ax = plt.subplots(figsize=[3, 3])
    # run_check_collect_time_delay_sz_stims(fig=fig, ax=ax)
    # fig.show()
    #
    #
    run__collect_szinvasiontime_vs_photostimresponses()
    # plot__szinvasiontime_vs_photostimresponses()
    # plot__szinvasiontime_vs_photostimresponses_indivexp()

    # run collecting and plotting zscored photostim responses vs. time delay to sz invasion
    run__collect_szinvasiontime_vs_photostimresponses_zscored()
    # plot__szinvasiontime_vs_photostimresponseszscored()

    run__collect_szinvasiontime_vs_photostimresponses_zscored_df()
    RESULTS.data = run__plot_responses_vs_time_to_seizure_SLMTargets_2ddensity()
    # RESULTS.save_results()
    RESULTS.data_all, RESULTS.percentiles, RESULTS.responses_sorted, RESULTS.times_to_sz_sorted, \
    RESULTS.scale_percentile_times = run__convert_responses_sztimes_percentile_space(data=RESULTS.data)

    RESULTS.save_results()

    MAIN.plot_density_responses_sztimes(RESULTS.data_all, RESULTS.times_to_sz_sorted, RESULTS.scale_percentile_times,
                                        photostim_responses_zscore_type=MAIN.photostim_responses_zscore_type)
    plot_lineplot_responses_pctsztimes(percentiles=RESULTS.percentiles, responses_sorted=RESULTS.responses_sorted,
                                       response_type='dFF (zscored)',
                                       scale_percentile_times=RESULTS.scale_percentile_times)

    # RESULTS.range_of_sz_invasion_time = [-1, -1, -1]
    # main.saveclass()
    pass
