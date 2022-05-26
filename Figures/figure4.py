from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets, plot__avg_photostim_dff_allexps

main = PhotostimAnalysisSlmTargets

# %% RUNNING PLOTS:

# %% A) schematic of variability of photostimulation responses

# main.plot__schematic_variability_measurement()

main.plot__variability(figsize=[3, 5], rerun=False)

# %% B) plot mean response vs. variability for baseline + interictal

main.plot__mean_response_vs_variability(rerun=1)


# %% C) seizure boundary classification throughout propagation


# %% D)



