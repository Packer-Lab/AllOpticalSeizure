from typing import Union

import numpy as np
from matplotlib import pyplot as plt
import funcsforprajay.funcs as pj
import _alloptical_utils as Utils
from _analysis_._ClassPhotostimAnalysisSlmTargets import plot_peristim_avg_fakestims, plot_peristim_avg_photostims

from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import \
    PhotostimResponsesQuantificationSLMtargets as main, PhotostimResponsesSLMtargetsResults

from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
from funcsforprajay import funcs as pfuncs

# expobj: Post4ap = import_expobj(prep='RL109', trial='t-018')
results: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

print(results)
# print(results.pre_stim_FOV_flu)

# %%)
# r.0) init and collect photostim responses, create anndata structure

@Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
def run__initPhotostimResponseQuant(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets = main(expobj)
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
def run__collect_photostim_responses_exp(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_exp(expobj=expobj)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1)#, run_trials=main.TEST_TRIALS)
def run__collect_fake_photostim_responses_exp(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.collect_fake_photostim_responses_exp(expobj=expobj)
    # print(expobj.PhotostimResponsesSLMTargets.fake_responses_SLMtargets_tracedFF.shape)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
def run__add_hit_trials_anndata(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.add_hit_trials_anndata()
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, run_trials=['PS11 t-011'])
def run__create_anndata_SLMtargets(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.create_anndata_SLMtargets(expobj=expobj)
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, run_trials=['PS11 t-011'])
def run__add_stim_group_anndata(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.add_stim_group_anndata(expobj=expobj)
    expobj.save()\


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1)#, run_trials=main.TEST_TRIALS)
def run__add_fakestim_adata_layer(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.add_fakestim_adata_layer()
    expobj.save()



# r.0.1) plotting peristim avg traces for photostims and fakestims
def run__plot_peristimavg():
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, set_cache=False, allow_rerun=1)
    def pre4apexps_collect_photostim_responses(**kwargs):
        expobj: alloptical = kwargs['expobj']
        if 'pre' in expobj.exptype:
            # all stims
            mean_photostim_responses = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                stims='all')
            return np.mean(mean_photostim_responses)


# r.1) plotting mean photostim response magnitude across experiments and experimental groups
def full_plot_mean_responses_magnitudes():
    """create plot of mean photostim responses magnitudes for all three exp groups"""

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False, allow_rerun=1)
    def pre4apexps_collect_photostim_responses(**kwargs):
        expobj: alloptical = kwargs['expobj']
        if 'pre' in expobj.exptype:
            # all stims
            mean_photostim_responses = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                stims='all')
            return np.mean(mean_photostim_responses)

    mean_photostim_responses_baseline = pre4apexps_collect_photostim_responses()

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True,  set_cache=False, allow_rerun=1)
    def post4apexps_collect_photostim_responses(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        if 'post' in expobj.exptype:
            # interictal stims
            mean_photostim_responses_interictal = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                stims=expobj.stim_idx_outsz)

            # ictal stims
            mean_photostim_responses_ictal = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                stims=expobj.stim_idx_insz)

            return np.mean(mean_photostim_responses_interictal), np.mean(mean_photostim_responses_ictal)

    func_collector = post4apexps_collect_photostim_responses()

    if len(func_collector) > 0:
        mean_photostim_responses_interictal, mean_photostim_responses_ictal = np.asarray(func_collector)[:, 0], np.asarray(
            func_collector)[:, 1]

        pplot.plot_bar_with_points(
            data=[mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal],
            x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False, colors=['blue', 'green', 'purple'],
            expand_size_x=0.4, title='Average Photostim resopnses', y_label='% dFF (delta(trace(dFF))')

        return mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal



# r.2) z scoring

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1)
def run__z_score_photostim_responses(**kwargs):
    expobj: alloptical = kwargs['expobj']
    # expobj.PhotostimResponsesSLMTargets.z_score_photostim_responses()
    # expobj.PhotostimResponsesSLMTargets.z_score_photostim_responses_interictal()
    expobj.PhotostimResponsesSLMTargets.z_score_photostim_responses_baseline()
    expobj.save()


# plotting mean photostim response magnitude Z SCORED across experiments and experimental groups
def full_plot_mean_responses_magnitudes_zscored():
    """create plot of mean photostim responses magnitudes (zscored) for all three exptype groups"""

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False)
    def pre4apexps_collect_photostim_responses_zscored(**kwargs):
        expobj: alloptical = kwargs['expobj']
        if 'pre' in expobj.exptype:
            # all stims
            mean_photostim_responses_zscored = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_zscored(zscore_type='dFF (zscored)', stims='all')
            return mean_photostim_responses_zscored

    mean_photostim_responses_baseline_zscored = pre4apexps_collect_photostim_responses_zscored()

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False)
    def post4apexps_collect_photostim_responses_zscored(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        if 'post' in expobj.exptype:
            # interictal stims
            mean_photostim_responses_interictal_zscored = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_zscored(zscore_type='dFF (zscored)', stims=expobj.stim_idx_outsz)

            # ictal stims
            mean_photostim_responses_ictal_zscored = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_zscored(zscore_type='dFF (zscored)', stims=expobj.stim_idx_insz)

            return mean_photostim_responses_interictal_zscored, mean_photostim_responses_ictal_zscored

    func_collector = post4apexps_collect_photostim_responses_zscored()

    if len(func_collector) > 0:
        mean_photostim_responses_interictal_zscored, mean_photostim_responses_ictal_zscored = np.asarray(func_collector)[:, 0], np.asarray(
            func_collector)[:, 1]

        # process data to make them flat arrays
        data = []
        for array in [mean_photostim_responses_baseline_zscored, mean_photostim_responses_interictal_zscored, mean_photostim_responses_ictal_zscored]:
            data.append(pfuncs.flattenOnce(array))


        # return mean_photostim_responses_baseline_zscored, mean_photostim_responses_interictal_zscored, mean_photostim_responses_ictal_zscored

        return data

# 5) plotting photostim responses in relation to pre-stim mean FOV Flu

# 5.1) plotting pre-stim mean FOV Flu for three stim type groups
def plot__prestim_FOV_Flu(results):
    """plot avg pre-stim Flu values across baseline, interictal, and ictal stims"""

    baseline__prestimFOV_flu = []
    for exp__prestim_flu in results.pre_stim_FOV_flu['baseline']:
        baseline__prestimFOV_flu.append(np.round(np.mean(exp__prestim_flu), 5))

    interictal__prestimFOV_flu = []
    for exp__prestim_flu in results.pre_stim_FOV_flu['interictal']:
        interictal__prestimFOV_flu.append(np.round(np.mean(exp__prestim_flu), 5))

    ictal__prestimFOV_flu = []
    for exp__prestim_flu in results.pre_stim_FOV_flu['ictal']:
        ictal__prestimFOV_flu.append(np.round(np.mean(exp__prestim_flu), 5))

    pplot.plot_bar_with_points(data=[baseline__prestimFOV_flu, interictal__prestimFOV_flu, ictal__prestimFOV_flu],
                               bar=False, x_tick_labels=['baseline', 'interictal', 'ictal'],
                               colors=['blue', 'green', 'purple'],
                               expand_size_x=0.4, title='Average Pre-stim FOV Flu', y_label='raw Flu')


def plot__photostim_responses_vs_prestim_FOV_flu(alpha=0.1, s=50, xlim=None, ylim=None, log=False):
    """plot avg target photostim responses in relation to pre-stim Flu value across baseline, interictal, and ictal stims.

    x-axis = pre-stim mean FOV flu, y-axis = photostim responses"""

    # import alloptical_utils_pj as aoutils
    # expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
    from _utils_.alloptical_plotting import dataplot_frame_options
    dataplot_frame_options()

    fig, ax = plt.subplots(figsize=(5, 5))

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0)
    def _plot_data_pre4ap(**kwargs):
        expobj: alloptical = kwargs['expobj']
        ax = kwargs['ax']
        assert 'pre' in expobj.exptype, f'wrong expobj exptype. {expobj.exptype}. expected pre'

        x_data = expobj.PhotostimResponsesSLMTargets.adata.var['pre_stim_FOV_Flu']
        y_data = expobj.PhotostimResponsesSLMTargets.adata.var['avg targets photostim response']

        ax.scatter(x_data, y_data, facecolor='blue', alpha=alpha, s=s)

        return x_data, y_data

    func_collector = _plot_data_pre4ap(ax=ax)

    assert len(func_collector) > 0
    x_data_baseline, y_data_baseline = pj.flattenOnce(np.asarray(func_collector)[:, 0]), pj.flattenOnce(np.asarray(func_collector)[:, 1])
                                       # pj.flattenOnce(np.asarray(func_collector)[:, 2])



    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
    def _plot_data_post4ap(**kwargs):
        expobj: alloptical = kwargs['expobj']
        ax = kwargs['ax']
        assert 'post' in expobj.exptype, f'wrong expobj exptype. {expobj.exptype}. expected post'

        # inter-ictal stims
        interictal_stims_idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_group == 'interictal')[0]
        x_data_interic = expobj.PhotostimResponsesSLMTargets.adata.var['pre_stim_FOV_Flu'][interictal_stims_idx]
        y_data_interic = expobj.PhotostimResponsesSLMTargets.adata.var['avg targets photostim response'][
            interictal_stims_idx]
        ax.scatter(x_data_interic, y_data_interic, facecolor='green', alpha=alpha, s=s)

        # ictal stims
        ictal_stims_idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_group == 'ictal')[0]
        x_data_ic = expobj.PhotostimResponsesSLMTargets.adata.var['pre_stim_FOV_Flu'][ictal_stims_idx]
        y_data_ic = expobj.PhotostimResponsesSLMTargets.adata.var['avg targets photostim response'][ictal_stims_idx]
        ax.scatter(x_data_ic, y_data_ic, facecolor='purple', alpha=alpha, s=s)

        return x_data_interic, y_data_interic, x_data_ic, y_data_ic

    func_collector = _plot_data_post4ap(ax=ax)
    assert len(func_collector) > 0
    x_data_interic, y_data_interic, x_data_ic, y_data_ic = pj.flattenOnce(np.asarray(func_collector)[:, 0]), \
                                                               pj.flattenOnce(np.asarray(func_collector)[:, 1]), \
                                                               pj.flattenOnce(np.asarray(func_collector)[:, 2]), \
                                                               pj.flattenOnce(np.asarray(func_collector)[:, 3])
                                                               # pj.flattenOnce(np.asarray(func_collector)[:, 4])

    # return this in the future in a separate function
    pre_stim_FOVflu_vs_targets_responses_results = {'baseline_FOVflu': x_data_baseline,
                                                             'baseline_targets_responses': y_data_baseline,
                                                             'interic_FOVflu': x_data_interic,
                                                             'interic_targets_responses': y_data_interic,
                                                             'ictal_FOVflu': x_data_ic,
                                                             'ictal_targets_responses': y_data_ic}

    # complete plot
    ax.set_title('pre_stim_FOV vs. avg photostim response of targets', wrap=True)
    # ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    ax.set_xlabel('pre-stim FOV avg Flu (raw)')
    ax.set_ylabel('avg dFF of targets')
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    ax.set_xscale('log')

    fig.tight_layout(pad=2)
    # Utils.save_figure(fig, save_path_suffix="plot__pre-stim-fov_vs_avg-photostim-response-of-targets.png")
    fig.show()

# plot__photostim_responses_vs_prestim_FOV_flu(alpha=0.05, s=25, xlim=[100, 2000], ylim=[-100, 100], log=True)


# %% RUN SCRIPT
if __name__ == '__main__':

    # "Initializing PhotostimResponsesQuantificationSLMtargets Analysis and Results Collection"
    # run__initPhotostimResponseQuant()
    #
    #
    "Collecting photostim responses for SLM Targets. Create anndata object to store photostim responses."
    # run__collect_photostim_responses_exp()
    # run__collect_fake_photostim_responses_exp()
    # run__create_anndata_SLMtargets()
    # run__add_fakestim_adata_layer()
    # run__add_stim_group_anndata()


    "Plotting photostim and fakestim peristim avg traces"
    # plot_peristim_avg_fakestims()
    # plot_peristim_avg_photostims()

    "Plotting mean photostim responses magnitudes across three brain states."
    # main.allexps_plot_photostim_responses_magnitude()   # <- plotting here after next to quantify the response magnitude of photostim vs. fake stims across experiments.
    # results.mean_photostim_responses_baseline, results.mean_photostim_responses_interictal, results.mean_photostim_responses_ictal = full_plot_mean_responses_magnitudes()
    #


    "Create and plot zscored photostim responses."
    run__z_score_photostim_responses()
    # main.allexps_plot_photostim_responses_magnitude_zscored()
    #
    #



    "Collecting and plotting zscored photostim responses across groups"
    # results.mean_photostim_responses_baseline_zscored, results.mean_photostim_responses_interictal_zscored, results.mean_photostim_responses_ictal_zscored = full_plot_mean_responses_magnitudes_zscored()
    # # make plot
    # # plot zscored responses
    # data = full_plot_mean_responses_magnitudes_zscored()
    # pplot.plot_bar_with_points(
    #     data=data,
    #     x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False, colors=['navy', 'green', 'purple'],
    #     expand_size_x=0.4, title='Average Photostim responses (zscored to baseline?)', y_label='dFF (zscored)')

    # pplot.plot_hist_density(data=data, mean_line=False, figsize=[4, 5], title='photostim responses (zscored)',
    #                         show_legend=True, num_bins=35, line_colors=['navy', 'green', 'purple'],
    #                         fill_color=['lightgray', 'lightgray', 'lightgray'], alpha=0.2, show_bins=True,
    #                         legend_labels=['baseline', 'interictal', 'ictal'])
    # results.save_results()
    #
    #

    "Measuring photostim responses in relation to pre-stim mean FOV Flu"
    # results.pre_stim_FOV_flu = main.collect__prestim_FOV_Flu()
    # results.save_results()
    # plot__prestim_FOV_Flu(results)
    # main.run__collect_photostim_responses_magnitude_avgtargets()
    # plot__photostim_responses_vs_prestim_FOV_flu()



    "Measuring photostim responses in relation to pre-stim mean targets_annulus Flu"
    # main.run__targets_annulus_prestim_Flu()

    # results.expavg_pre_stim_targets_annulus_F = PhotostimResponsesQuantificationSLMtargets.retrieve__targets_annlus_prestim_Flu()

    # results.expavg_pre_stim_targets_annulus_results_ictal = main.retrieve__targets_annlus_prestim_Flu_duringsz()
    # results.save_results()

    # main.plot__targets_annulus_prestim_Flu(results)
    # main.plot__targets_annulus_prestim_Flu_outszvsinsz(results)
    #
    # main.plot__targets_annulus_prestim_Flu_combined(results)
    #
    # results.pre_stim_targets_annulus_vs_targets_responses_results = main.retrieve__photostim_responses_vs_prestim_targets_annulus_flu()
    # results.save_results()
    # main.plot__photostim_responses_vs_prestim_targets_annulus_flu(results)
    # main.plot__targets_annulus_prestim_Flu_all_points(results)



    pass

