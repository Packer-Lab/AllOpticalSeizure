from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
    PhotostimResponsesNonTargetsResults
from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import PhotostimResponsesAnalysisNonTargets

# %% run processing/analysis/plotting:


# %% processing alloptical photostim and fakestim responses

# PhotostimResponsesAnalysisNonTargets.run__initPhotostimResponsesAnalysisNonTargets()

# PhotostimResponsesQuantificationNonTargets.run__fakestims_processing()

# %% plotting alloptical and fakestim responses
# PhotostimResponsesAnalysisNonTargets.run__plot_sig_responders_traces(plot_baseline_responders=False)

# %% collecting all summed nontargets photostim and fakestim responses vs. total targets photostim and fakestim responses
main = PhotostimResponsesAnalysisNonTargets
main.run__summed_responses(rerun=0)

# %% plotting exps total nontargets photostim (and fakestim) responses vs. total targets photostim (and fakestim) responses

# PhotostimResponsesAnalysisNonTargets.plot__exps_summed_nontargets_vs_summed_targets()

# %% collecting and plotting z scored summed total and nontargets responses

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()
main.collect__zscored_summed_activity_vs_targets_activity(results=results)
main.plot__summed_activity_vs_targets_activity(results=results)






