from typing import Union

import numpy as np

import _alloptical_utils as Utils
from _analysis_.ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesQuantificationSLMtargets as main
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

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



# %%

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



@Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
def run__create_anndata_SLMtargets(**kwargs):
    expobj: alloptical = kwargs['expobj']
    expobj.PhotostimResponsesSLMTargets.create_anndata_SLMtargets(expobj=expobj)
    expobj.save()





# %% c.1) plotting mean photostim response magnitude
def full_plot_mean_responses_magnitudes():
    """create plot of mean photostim responses magnitudes for all three exp groups"""

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, allow_rerun=True)
    def pre4apexps_collect_photostim_responses(**kwargs):
        expobj: alloptical = kwargs['expobj']
        if 'pre' in expobj.exptype:
            # all stims
            mean_photostim_responses = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude(
                stims='all')
            return np.mean(mean_photostim_responses)

    mean_photostim_responses_baseline = pre4apexps_collect_photostim_responses()

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def post4apexps_collect_photostim_responses(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        if 'post' in expobj.exptype:
            # interictal stims
            mean_photostim_responses_interictal = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude(
                stims=expobj.stim_idx_outsz)

            # ictal stims
            mean_photostim_responses_ictal = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude(
                stims=expobj.stim_idx_insz)

            return np.mean(mean_photostim_responses_interictal), np.mean(mean_photostim_responses_ictal)

    func_collector = post4apexps_collect_photostim_responses()
    mean_photostim_responses_interictal, mean_photostim_responses_ictal = np.asarray(func_collector)[:, 0], np.asarray(
        func_collector)[:, 1]

    # plot_photostim_response_of_groups
    pplot.plot_bar_with_points(data=[mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal],
                               x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False, colors=['blue', 'green', 'purple'],
                               expand_size_x=0.5)

    return mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal


main.mean_photostim_responses_baseline, main.mean_photostim_responses_interictal, main.mean_photostim_responses_ictal = full_plot_mean_responses_magnitudes()

pplot.plot_bar_with_points(
    data=[main.mean_photostim_responses_baseline, main.mean_photostim_responses_interictal, main.mean_photostim_responses_ictal],
    x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False, colors=['blue', 'green', 'purple'],
    expand_size_x=0.4, title='Average Photostim resopnses', y_label='% dFF (delta(trace(dFF))')

main.saveclass()


# %% RUN SCRIPT
if __name__ == '__main__':
    # run__initPhotostimResponseQuant()
    # run__collect_photostim_responses_exp()
    # run__create_anndata_SLMtargets()
    # main.allexps_plot_photostim_responses_magnitude()
    pass
