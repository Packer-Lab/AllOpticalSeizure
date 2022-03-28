import sys;

import matplotlib.pyplot as plt

from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview as main, TargetsSzInvasionSpatialResults_codereview

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import _alloptical_utils as Utils

from _analysis_._ClassTargetsSzInvasionSpatial import TargetsSzInvasionSpatial, TargetsSzInvasionSpatialResults
results: TargetsSzInvasionSpatialResults_codereview = TargetsSzInvasionSpatialResults_codereview.load()

# Results__TargetsSzInvasionSpatial = TargetsSzInvasionSpatialResults.load()

from _main_.Post4apMain import Post4ap
SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


# running processing and analysis pipeline

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
def run__initTargetsSzInvasionSpatial(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionSpatial = TargetsSzInvasionSpatial(expobj=expobj)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1)
def run__collect_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsSzInvasionSpatial.collect__responses_vs_distance_to_seizure_SLMTargets(expobj=expobj, response_type=TargetsSzInvasionSpatial.response_type)
    expobj.save()


# %%
if __name__ == '__main__':
    """
    rerunning pipeline on mar 12/13 2022 - need to confirm that the results seem real - looking too good to be real lol.
    - first trying to rerun with no trials skipped explicitly. 
    
    """

    # run__initTargetsSzInvasionSpatial()
    # Results__TargetsSzInvasionSpatial.no_slmtargets_szboundary_stim = TargetsSzInvasionSpatial.run_calculating_min_distance_to_seizure()
    # Results__TargetsSzInvasionSpatial.save_results()
    #
    # TargetsSzInvasionSpatial.run__collect_responses_vs_distance_to_seizure_SLMTargets()

    # sys.exit()

    # TargetsSzInvasionSpatial.plot_responses_vs_distance_to_seizure_SLMTargets()
    #
    # TargetsSzInvasionSpatial.plot_collection_response_distance()
    #
    # Results__TargetsSzInvasionSpatial.data = TargetsSzInvasionSpatial.plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity(response_type=TargetsSzInvasionSpatial.response_type, positive_distances_only=False, plot=False)
    # #
    # Results__TargetsSzInvasionSpatial.save_results()
    #
    # # ********* NOTE: THIS TAKES FOREVER (.22/03/01)!!!!!! NOT SURE IF IT SHOULD BE TAKING THIS LONG OR NOT..... ******************* new estimate is that it should only take ~40 - 45 mins (.22/03/02)
    # Results__TargetsSzInvasionSpatial.data_all, Results__TargetsSzInvasionSpatial.percentiles, Results__TargetsSzInvasionSpatial.responses_sorted, \
    #     Results__TargetsSzInvasionSpatial.distances_to_sz_sorted, Results__TargetsSzInvasionSpatial.scale_percentile_distances = TargetsSzInvasionSpatial.convert_responses_szdistances_percentile_space(input_data=Results__TargetsSzInvasionSpatial.data)

    # Results__TargetsSzInvasionSpatial.save_results()

    # TargetsSzInvasionSpatial.plot_density_responses_szdistances(response_type=Results__TargetsSzInvasionSpatial.response_type,
    #                                                             data_all=Results__TargetsSzInvasionSpatial.data_all,
    #                                                             distances_to_sz_sorted=Results__TargetsSzInvasionSpatial.distances_to_sz_sorted,
    #                                                             scale_percentile_distances=Results__TargetsSzInvasionSpatial.scale_percentile_distances)
    # TargetsSzInvasionSpatial.plot_lineplot_responses_pctszdistances(Results__TargetsSzInvasionSpatial.percentiles,
    #                                                                 Results__TargetsSzInvasionSpatial.responses_sorted,
    #                                                                 response_type=Results__TargetsSzInvasionSpatial.response_type,
    #                                                                 scale_percentile_distances=Results__TargetsSzInvasionSpatial.scale_percentile_distances)

    # collect distance vs. respnses for distance bins
    bin_width, distances, num, avg_responses, conf_int =  main.collect__binned__distance_v_responses()
    results.binned__distance_vs_photostimresponses = {'bin_width_um': bin_width, 'distance_bins': distances, 'num_points_in_bin': num,
    'avg_photostim_response_in_bin': avg_responses, '95conf_int': conf_int}
    results.save_results()
    main.plot__responses_v_distance_no_normalization(results=results)











# todo refactor code into the results class here.
# todo run similar plot for temporal sz invasion analysis.



