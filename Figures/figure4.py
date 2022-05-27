import numpy as np
from funcsforprajay.plotting.plotting import plot_bar_with_points

from _alloptical_utils import run_for_loop_across_exps
from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets, plot__avg_photostim_dff_allexps
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults, \
    PhotostimResponsesQuantificationSLMtargets
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap

main = PhotostimAnalysisSlmTargets

# %% RUNNING PLOTS:

# %% xx) plotting interictal photostim responses split by pre-ictal and post-ictal

RESULTS = PhotostimResponsesSLMtargetsResults.load()

fig, ax = plot_bar_with_points(data=[
    RESULTS.interictal_responses['preictal_responses'],
    RESULTS.interictal_responses['very_interictal_responses'],
    RESULTS.interictal_responses['postictal_responses']
                                     ],
    bar=False, title='photostim responses - targets', x_tick_labels=['pre-ictal', 'very interictal', 'post-ictal'],
    colors=['lightseagreen', 'gold', 'lightcoral'], figsize=(4, 4), y_label=RESULTS.interictal_responses['data_label'], show=False, ylims=[-0.5, 0.5], alpha=1)
fig.tight_layout(pad=0.2)
fig.show()

# %% A) schematic of variability of photostimulation responses

# main.plot__schematic_variability_measurement()

# main.plot__variability(figsize=[3, 5], rerun=False)

# %% B) plot mean response vs. variability for baseline + interictal

# main.plot__mean_response_vs_variability(rerun=1)


# %% C) seizure boundary classification throughout propagation


# %% D) plotting target annulus vs photostim responses

RESULTS = PhotostimResponsesSLMtargetsResults.load()

PhotostimResponsesQuantificationSLMtargets.plot__photostim_responses_vs_prestim_targets_annulus_flu(RESULTS)

# PhotostimResponsesQuantificationSLMtargets.plot__targets_annulus_prestim_Flu_all_points(RESULTS)





