from typing import Union

import numpy as np

import _alloptical_utils as Utils

from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import \
    PhotostimResponsesQuantificationSLMtargets as main, PhotostimResponsesSLMtargetsResults

from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
from funcsforprajay import funcs as pfuncs

# expobj: Post4ap = import_expobj(prep='RL109', trial='t-018')
RESULTS: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

print(RESULTS)
# print(RESULTS.pre_stim_FOV_flu)

"##### -------------------- ALL OPTICAL PHOTOSTIM ANALYSIS #############################################################"

# %% c.0) feb 19- 2022: TRYING NEW CLASS DRIVEN APPROACH FOR EACH ANALYSIS

# @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=True)
# def delete_old(**kwargs):
#     expobj: Union[alloptical, Post4ap] = kwargs['expobj']
#     delattr(expobj, 'PhotostimResponsesQuantificationSLMtargets')
#     expobj.save()
# delete_old()


# expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
# expobj.PhotostimResponsesSLMTargets = main(expobj)
# expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_exp(expobj=expobj)
# expobj.PhotostimResponsesSLMTargets.create_anndata_SLMtargets(expobj=expobj)
# expobj.PhotostimResponsesSLMTargets.plot_photostim_responses_magnitude(expobj=expobj, stims='all')
# expobj.save()


# RESULTS.pre_stim_FOV_flu = main.collect__prestim_FOV_Flu()
# RESULTS.save_results()
# print(RESULTS)
# print(RESULTS.pre_stim_FOV_flu)


# main.run__collect_photostim_responses_magnitude_avgtargets()
main.plot__photostim_responses_vs_prestim_FOV_flu()

# %% r.0) init and collect photostim responses, create anndata structure

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



@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, run_trials=['PS11 t-011'])
def run__create_anndata_SLMtargets(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.create_anndata_SLMtargets(expobj=expobj)
    expobj.save()

run__create_anndata_SLMtargets()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, run_trials=['PS11 t-011'])
def run__add_stim_group_anndata(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.add_stim_group_anndata(expobj=expobj)
    expobj.save()

run__add_stim_group_anndata()



# %% r.1) plotting mean photostim response magnitude across experiments and experimental groups
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



# %% r.2) z scoring

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, run_trials=['PS11 t-011'])
def run__z_score_photostim_responses_and_interictalzscores(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.z_score_photostim_responses()
    expobj.PhotostimResponsesSLMTargets.z_score_photostim_responses_interictal()
    expobj.save()
run__z_score_photostim_responses_and_interictalzscores()


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




# %% RUN SCRIPT
if __name__ == '__main__':

    # "Initializing PhotostimResponsesQuantificationSLMtargets Analysis and Results Collection"
    # run__initPhotostimResponseQuant()
    #
    #
    # "Collecting photostim responses for SLM Targets. Create anndata object to store photostim responses."
    # run__collect_photostim_responses_exp()
    # run__create_anndata_SLMtargets()
    # run__add_stim_group_anndata()
    #
    # "Plotting mean photostim responses magnitudes across three brain states."
    # main.allexps_plot_photostim_responses_magnitude()
    # RESULTS.mean_photostim_responses_baseline, RESULTS.mean_photostim_responses_interictal, RESULTS.mean_photostim_responses_ictal = full_plot_mean_responses_magnitudes()
    #
    #
    # "Create and plot zscored photostim responses."
    # run__z_score_photostim_responses_and_interictalzscores()
    # main.allexps_plot_photostim_responses_magnitude_zscored()
    #
    #
    # "Collecting and plotting zscored photostim responses across groups"
    # RESULTS.mean_photostim_responses_baseline_zscored, RESULTS.mean_photostim_responses_interictal_zscored, RESULTS.mean_photostim_responses_ictal_zscored = full_plot_mean_responses_magnitudes_zscored()
    # # make plot
    # # plot zscored responses
    # data = full_plot_mean_responses_magnitudes_zscored()
    # pplot.plot_bar_with_points(
    #     data=data,
    #     x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False, colors=['navy', 'green', 'purple'],
    #     expand_size_x=0.4, title='Average Photostim responses (zscored to baseline?)', y_label='dFF (zscored)')
    #
    # pplot.plot_hist_density(data=data, mean_line=False, figsize=[4, 5], title='photostim responses (zscored)',
    #                         show_legend=True, num_bins=35, line_colors=['navy', 'green', 'purple'],
    #                         fill_color=['lightgray', 'lightgray', 'lightgray'], alpha=0.2, show_bins=True,
    #                         legend_labels=['baseline', 'interictal', 'ictal'])
    # RESULTS.save_results()
    #
    #
    # "Measuring photostim responses in relation to pre-stim mean FOV Flu"
    # RESULTS.pre_stim_FOV_flu = main.collect__prestim_FOV_Flu()
    # RESULTS.save_results()
    # main.plot__prestim_FOV_Flu(RESULTS)
    # main.run__collect_photostim_responses_magnitude_avgtargets()
    # main.plot__photostim_responses_vs_prestim_FOV_flu()
    #


    "Measuring photostim responses in relation to pre-stim mean targets_annulus Flu"
    main.run__add_targets_annulus_prestim_anndata()



    pass
