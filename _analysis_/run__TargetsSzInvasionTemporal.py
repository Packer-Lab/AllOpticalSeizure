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
from funcsforprajay.wrappers import plot_piping_decorator
from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal as main

import _alloptical_utils as Utils
from _main_.Post4apMain import Post4ap
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames


# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
# SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"

expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
# expobj = Utils.import_expobj(load_backup_path='/home/pshah/mnt/qnap/Analysis/2020-12-18/RL108/2020-12-18_t-013/backups/2020-12-18_RL108_t-013.pkl')

# expobj.PhotostimResponsesSLMTargets.adata.var[['stim_start_frame', 'wvfront in sz', 'seizure_num']]


# %% code development zone
# expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
# expobj.TargetsSzInvasionTemporal.plot_szinvasiontime_vs_photostimresponses(fig=None, ax=None)


# %% code deployment zone

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
def run__initTargetsSzInvasionTemporal(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal = main(expobj=expobj)
    expobj.save()


# run__initTargetsSzInvasionTemporal()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False,
                                run_post4ap_trials=1)  # , run_trials=['RL109 t-016', 'PS07 t-011', 'PS11 t-011'])
def run__collect_targets_sz_invasion_traces(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_targets_sz_invasion_traces(expobj=expobj)
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0)
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


# run_check_collect_time_delay_sz_stims()


fig, ax = plt.subplots(figsize=[3, 3])


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def run_plot_time_delay_sz_stims(**kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.plot_num_pos_neg_szinvasion_stims(**kwargs)


# run_plot_time_delay_sz_stims(fig=fig, ax=ax)


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


## zscored photostim responses vs time #################################################################################
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def run__collect_szinvasiontime_vs_photostimresponses_zscored(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses_zscored(expobj=expobj)
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


## binning and plotting zscored photostim responses vs. time ###########################################################
# TODO - MAIN WORK RIGHT NOW:
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def run__plot_responses_vs_time_to_seizure_SLMTargets_2ddensity(**kwargs):
    """run plotting of responses vs. time as 2d density plot (which will bin the data automatically)"""
    expobj: Post4ap = kwargs['expobj']
    data = expobj.TargetsSzInvasionTemporal.plot_responses_vs_time_to_seizure_SLMTargets_2ddensity(
        positive_times_only=False,
        plot=True, expobj=expobj)
    expobj.save()
    return data

main._data_for_photostim_response_vs_time = run__plot_responses_vs_time_to_seizure_SLMTargets_2ddensity()
# main.saveclass()

# %%

if __name__ == '__main__':
    # run__initTargetsSzInvasionTemporal()
    # run_collect_time_delay_sz_stims()
    # run_check_collect_time_delay_sz_stims()
    # run_plot_time_delay_sz_stims(fig=fig, ax=ax)

    # - should be done: before running the following below, need to update the expobj slmtargets adata to the new PhotostimResponses Class adata !!!
    # run__collect_szinvasiontime_vs_photostimresponses()
    # plot__szinvasiontime_vs_photostimresponses()
    # plot__szinvasiontime_vs_photostimresponses_indivexp()

    # run__collect_targets_sz_invasion_traces()
    # main.plot_targets_sz_invasion_meantraces_full()

    # run collecting and plotting zscored photostim responses vs. time delay to sz invasion
    # run__collect_szinvasiontime_vs_photostimresponses_zscored()
    # plot__szinvasiontime_vs_photostimresponseszscored()
    # main.range_of_sz_invasion_time = [-1, -1, -1]
    main.saveclass()
