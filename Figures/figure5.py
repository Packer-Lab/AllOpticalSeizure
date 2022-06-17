# code for building figure 5
from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview, \
    TargetsSzInvasionSpatialResults_codereview

SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-responses-traces/"

main = TargetsSzInvasionSpatial_codereview
results = TargetsSzInvasionSpatialResults_codereview.load()

# todo change z score to baseline z score!!

# %% A)

main.plot__responses_v_distance_no_normalization(results=results, save_path_full=f'{SAVE_FIG}/responses_sz_distance_binned_line_plot.png')



# %% density- responses vs. sz distance


# main.plot_density_responses_szdistances(response_type=results.response_type, data_all=results.data_all,
#                                             distances_to_sz_sorted=results.distances_to_sz_sorted, scale_percentile_distances=results.scale_percentile_distances,
#                                             save_path_full=f'{SAVE_FIG}/responses_sz_distance_binned_density.png')
#

# %% responses vs. sz distance lineplot

# main.plot_lineplot_responses_pctszdistances(results.percentiles, results.responses_sorted,
#                                             response_type=results.response_type,
#                                             scale_percentile_distances=results.scale_percentile_distances,
#                                             save_path_full=f'{SAVE_FIG}/responses_sz_distance_binned_line_plot.png')




