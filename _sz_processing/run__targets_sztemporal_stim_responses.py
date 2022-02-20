"""TODO/GOALS:
2) plotting average traces around time of seizure invasion for all targets across all exps
    - plot also the mean FOV Flu at the bottom
3) plot average stim response before and after time of seizure invasion for all targets across all exps

"""
import funcsforprajay.funcs as pj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from funcsforprajay.wrappers import plot_piping_decorator

import _alloptical_utils as Utils
from _main_.Post4apMain import Post4ap
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"

expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
# expobj = Utils.import_expobj(load_backup_path='/home/pshah/mnt/qnap/Analysis/2020-12-18/RL108/2020-12-18_t-013/backups/2020-12-18_RL108_t-013.pkl')

# expobj.slmtargets_data.var[['stim_start_frame', 'wvfront in sz', 'seizure_num']]


# %% code development zone
expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
expobj.TargetsSzInvasionTemporal.plot_szinvasiontime_vs_photostimresponses(fig=None, ax=None)



# %% code deployment zone

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def init__TargetsSzInvasionTemporal(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    from _sz_processing.ClassTargetsSzInvasionTemporal import _TargetsSzInvasionTemporal
    expobj.TargetsSzInvasionTemporal = _TargetsSzInvasionTemporal()
    expobj.save()
# init__TargetsSzInvasionTemporal()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def run_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_time_delay_sz_stims(expobj=expobj)
    expobj.save()
# run_collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def run_check_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_num_pos_neg_szinvasion_stims(expobj=expobj)
    expobj.save()
# run_check_collect_time_delay_sz_stims()


fig, ax = plt.subplots(figsize=[3,3])
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def run_plot_time_delay_sz_stims(**kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.plot_num_pos_neg_szinvasion_stims(**kwargs)
# run_plot_time_delay_sz_stims(fig=fig, ax=ax)


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def run__collect_szinvasiontime_vs_photostimresponses(**kwargs):
    """run collecting dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionTemporal.collect_szinvasiontime_vs_photostimresponses(expobj=expobj)
    expobj.save()
run__collect_szinvasiontime_vs_photostimresponses()


def plot__szinvasiontime_vs_photostimresponses():
    """collects and plots all exps datapoints for szinvasion time vs. photostim responses on one plot"""
    x = []
    y = []
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
    def _return_szinvasiontime_vs_photostimresponses(x=x, y=y, **kwargs):
        expobj: Post4ap = kwargs['expobj']
        _x, _y = expobj.TargetsSzInvasionTemporal.return_szinvasiontime_vs_photostimresponses()
        x += _x
        y += _y
        return x, y
    result = _return_szinvasiontime_vs_photostimresponses()[-1]
    x, y = result[0], result[1]


    fig, ax = plt.subplots(figsize=[6,3])
    ax.grid(zorder=0)
    ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.05, ec='white')
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)
    ax.set_title(f"All exps targets time to sz invasion vs. photostim response", wrap=True)
    ax.set_xlabel(f"Time to sz invasion for target (secs)")
    ax.set_ylabel(f"Photostim response for target (dFF)", wrap=True)
    ax.set_ylim([-7.5, 7.5])
    fig.tight_layout(pad=2.5)
    fig.show()

plot__szinvasiontime_vs_photostimresponses()

# TODO NEED TO CONFIRM THE QUALITY OF THE PHOTOSTIM RESPONSES OF delta(trace_dFF)

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
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
plot__szinvasiontime_vs_photostimresponses_indivexp()





### below should be almost all refactored to _TargetsSzInvasionTemporal class



# %% 3) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME

# create a df containing delay to sz invasion for each target for each stim frame (dim: n_targets x n_stims)
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def collect_time_delay_sz_stims(self, **kwargs):
    expobj: Post4ap = kwargs['expobj']
    df = pd.DataFrame(columns=expobj.slmtargets_data.var['stim_start_frame'], index=expobj.slmtargets_data.obs.index)
    for target in expobj.slmtargets_data.obs.index:
        # target = 0
        print(f'\- collecting from target #{target} ... ') if (int(target) % 10) == 0 else None
        cols_ = [idx for idx, col in enumerate([*expobj.slmtargets_data.obs]) if 'time_del' in col]
        sz_times = expobj.slmtargets_data.obs.iloc[int(target), cols_]
        fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if
                    not pd.isnull(time)]
        for szstart, szstop in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
            fr_time = [i for i in fr_times if szstart < i < szstop]
            if len(fr_time) >= 1:
                for stim in expobj.stim_start_frames:
                    if szstart < stim < szstop:
                        time_del = - (stim - fr_time[0]) / expobj.fps
                        df.loc[target, stim] = round(time_del, 2) # secs

    self.time_del_szinv_stims = df
    self._update_expobj(expobj)
    expobj.save()
# collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def check_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    # plot num stims outside of sz invasion and num stims after sz invasion
    pos_values_collect = []
    neg_values_collect = []
    for idx, row in expobj.time_del_szinv_stims.iterrows():
        values = row[row.notnull()]
        pos_values_collect += list(values[values > 0])
        neg_values_collect += list(values[values < 0])

    expobj.time_delay_sz_stims_pos_values_collect = pos_values_collect
    expobj.time_delay_sz_stims_neg_values_collect = neg_values_collect

    print(f"avg num stim before sz invasion: {len(pos_values_collect)/expobj.n_targets_total}")
    print(f"avg num stim after sz invasion: {len(neg_values_collect)/expobj.n_targets_total}")

# check_collect_time_delay_sz_stims()

@plot_piping_decorator(figsize=(8, 4), nrows=1, ncols=1, verbose=False)
def plot_time_delay_sz_stims(**kwargs):
    fig, ax, expobj = kwargs['fig'], kwargs['ax'], kwargs['expobj']

    ax.hist(expobj.time_delay_sz_stims_pos_values_collect, fc='blue')
    ax.hist(expobj.time_delay_sz_stims_neg_values_collect, fc='purple')


# %% 4) PLOT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS


# %% 1) collecting mean of seizure invasion Flu traces from all targets for each experiment

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def check_targets_sz_invasion_time(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    print(expobj.slmtargets_data.obs)

# check_targets_sz_invasion_time()


# create dictionary containing mean Raw Flu trace around sz invasion time of each target (as well as other info as keyed into dict)
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)#, run_trials=['RL109 t-016', 'PS07 t-011', 'PS11 t-011'])
def collect_targets_sz_invasion_traces(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    fov_traces_ = []
    traces_ = []
    pre = 5
    post = 10
    for target, coord in enumerate(expobj.slmtargets_data.obs['SLM target coord']):
        # target, coord = 0, expobj.slmtargets_data.obs['SLM target coord'][0]
        print(f'\- collecting from target #{target} ... ') if (target % 10) == 0 else None
        cols_ = [idx for idx, col in enumerate([*expobj.slmtargets_data.obs]) if 'time_del' in col]
        sz_times = expobj.slmtargets_data.obs.iloc[target, cols_]
        fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if not pd.isnull(time)]

        # collect each frame seizure invasion time Flu snippet for current target
        target_traces = []
        fov_traces = []
        for fr in fr_times:
            target_tr = expobj.raw_SLMTargets[target][fr-int(pre*expobj.fps): fr+int(post*expobj.fps)]
            fov_tr = expobj.meanRawFluTrace[fr-int(pre*expobj.fps): fr+int(post*expobj.fps)]
            target_traces.append(target_tr)
            fov_traces.append(fov_tr)
            # ax.plot(pj.moving_average(to_plot, n=4), alpha=0.2)
        traces_.append(np.mean(target_traces, axis=0)) if len(target_traces) > 1 else None
        fov_traces_.append(np.mean(fov_traces, axis=0)) if len(target_traces) > 1 else None

    traces_ = np.array(traces_)
    fov_traces_ = np.array(fov_traces_)
    MEAN_TRACE = np.mean(traces_, axis=0)
    FOV_MEAN_TRACE = np.mean(fov_traces_, axis=0)
    expobj.mean_targets_szinvasion_trace = {'fov_mean_trace': FOV_MEAN_TRACE,
                                            'mean_trace': MEAN_TRACE,
                                            'pre_sec': pre,
                                            'post_sec': post}
    expobj.save()


# collect_targets_sz_invasion_traces()

# %% 2) PLOT TARGETS FLU PERI- TIME TO INVASION

def plot_targets_sz_invasion_meantraces_full():
    fig, ax = plt.subplots(figsize=[3, 6])

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
    def plot_targets_sz_invasion_meantraces(**kwargs):
        expobj: Post4ap = kwargs['expobj']

        fov_mean_trace, to_plot, pre, post = expobj.mean_targets_szinvasion_trace['fov_mean_trace'], expobj.mean_targets_szinvasion_trace['mean_trace'], expobj.mean_targets_szinvasion_trace['pre_sec'], expobj.mean_targets_szinvasion_trace['post_sec']
        invasion_spot = int(pre*expobj.fps)
        to_plot = pj.moving_average(to_plot, n=6)
        fov_mean_trace = pj.moving_average(fov_mean_trace, n=6)
        to_plot_normalize = (to_plot - to_plot[0]) / to_plot[0]
        fov_mean_normalize = (fov_mean_trace - fov_mean_trace[0]) / fov_mean_trace[0]

        x_time = np.linspace(-pre, post, len(to_plot))

        ax.plot(x_time, to_plot_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3)
        # ax2.plot(x_time, fov_mean_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3, alpha=0.5, linestyle='--')

        ax.scatter(x=0, y=to_plot_normalize[invasion_spot], color='crimson', s=45, zorder=5)

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

plot_targets_sz_invasion_meantraces_full()







