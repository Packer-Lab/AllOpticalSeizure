from _analysis_._ClassPhotostimResponseQuantificationNonTargets import PhotostimResponsesQuantificationNonTargets
from _analysis_._ClassPhotostimResponsesAnalysisNonTargets import PhotostimResponsesAnalysisNonTargets

from _main_.AllOpticalMain import alloptical
from _utils_.io import import_expobj


# %% run processing/analysis/plotting:


# %% processing alloptical photostim and fakestim responses

# PhotostimResponsesAnalysisNonTargets.run__initPhotostimResponsesAnalysisNonTargets()

PhotostimResponsesQuantificationNonTargets.run__fakestims_processing()

# %% plotting alloptical and fakestim responses
# PhotostimResponsesAnalysisNonTargets.run__plot_sig_responders_traces(plot_baseline_responders=False)

# %% collecting all summed nontargets photostim and fakestim responses vs. total targets photostim and fakestim responses
PhotostimResponsesAnalysisNonTargets.run__summed_responses(rerun=0)


# %% plotting total nontargets photostim (and fakestim) responses vs. total targets photostim (and fakestim) responses

expobj: alloptical = import_expobj(exp_prep='RL108 t-009')

expobj.PhotostimResponsesNonTargets.plot__exps_summed_nontargets_vs_summed_targets(expobj=expobj)




