from _analysis_._ClassPhotostimAnalysisSlmTargets import plot_peristim_avg_fakestims
from _analysis_._ClassPhotostimResponseQuantificationNonTargets import PhotostimResponsesQuantificationNonTargets
from _analysis_.run__PhotostimResponseQuantificationSLMtargets import run__collect_fake_photostim_responses_exp, \
    run__add_fakestim_adata_layer

# %% RUN PROCESSING/ANALYSIS/PLOTTING OF COLLECTING FAKESTIMS TRACES/RESPONSES ACROSS SLM TARGETS AND NONTARGETS


PhotostimResponsesQuantificationNonTargets.run__fakestims_processing()

run__collect_fake_photostim_responses_exp()
run__add_fakestim_adata_layer()

plot_peristim_avg_fakestims()






