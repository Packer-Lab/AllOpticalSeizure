import sys;

import matplotlib.pyplot as plt

import _analysis_._ClassTargetsSzInvasionSpatial_codereview

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import _alloptical_utils as Utils

from _analysis_._ClassTargetsSzInvasionSpatial import TargetsSzInvasionSpatial, TargetsSzInvasionSpatialResults

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


# %%
import numpy as np
import _alloptical_utils as Utils
from _main_.Post4apMain import Post4ap
from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview as main, TargetsSzInvasionSpatialResults_codereview
results: TargetsSzInvasionSpatialResults_codereview = TargetsSzInvasionSpatialResults_codereview.load()

# expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')

# use histogram to plot binned photostim responses over distance to sz wavefront

# szinvspatial = expobj.TargetsSzInvasionSpatial_codereview
# %%
# def add_photostimresponses(response_type=szinvspatial.response_type):
#     responses = expobj.PhotostimResponsesSLMTargets.adata.layers[response_type]
#     szinvspatial.adata.add_layer(layer_name=response_type, data=responses)
#     print(szinvspatial.adata)
# add_photostimresponses()

# szinvspatial.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz_um']

# %% collect distance vs. respnses for distance bins
bin_width = 20  # um
bins = np.arange(0,500,bin_width)  # 0 --> 500 um, split in XXum bins
num = [0 for _ in range(len(bins))]  # num of datapoints in binned distances
y = [0 for _ in range(len(bins))]  # avg responses at distance bin
responses = [[] for _ in range(len(bins))]  # collect all responses at distance bin

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=False)
def add_dist_responses(bins, num, y, responses, **kwargs):
    expobj = kwargs['expobj']
    szinvspatial = expobj.TargetsSzInvasionSpatial_codereview

    # print(num)
    # print(y)

    for _, row in szinvspatial.responses_vs_distance_to_seizure_SLMTargets.iterrows():
        dist = row['distance_to_sz_um']
        response = row[szinvspatial.response_type]
        for i, bin in enumerate(bins[:-1]):
            if bins[i] < dist < (bins[i + 1]):
                num[i] += 1
                y[i] += response
                responses[i].append(response)

    return num, y, responses

func_collector = add_dist_responses(bins=bins, num=num, y=y, responses=responses)

num, y, responses = func_collector[-1][0], func_collector[-1][1], func_collector[-1][2]


avg_responses = [y[i]/num_points for i, num_points in enumerate(num) if num_points != 0]
distances = bins + 10

avg_responses2 = [np.mean(responses_) for responses_ in responses]


# calculate 95% ci for avg responses
import scipy.stats as stats
conf_int = np.array([stats.t.interval(alpha=0.95, df = len(responses_)-1, loc=np.mean(responses_), scale=stats.sem(responses_)) for responses_ in responses])


results.binned__distance_vs_photostimresponses = {'bin_width_um': bin_width, 'distance_bins': distances, 'num_points_in_bin': num,
'avg_photostim_response_in_bin': avg_responses, '95conf_int': conf_int}


results.save_results()

# %%
import matplotlib.pyplot as plt
import funcsforprajay.funcs as pj

# distances_bins = results.binned__distance_vs_photostimresponses['distance_bins']
distances = results.binned__distance_vs_photostimresponses['distance_bins']
avg_responses = results.binned__distance_vs_photostimresponses['avg_photostim_response_in_bin']
conf_int = results.binned__distance_vs_photostimresponses['95conf_int']
num2 = results.binned__distance_vs_photostimresponses['num_points_in_bin']

conf_int_distances = pj.flattenOnce([[distances[i], distances[i + 1]] for i in range(len(distances) - 1)])
conf_int_values_neg = pj.flattenOnce([[val, val] for val in conf_int[1:, 0]])
conf_int_values_pos = pj.flattenOnce([[val, val] for val in conf_int[1:, 1]])


# %%

fig, axs = plt.subplots(figsize = (6, 6), nrows=2, ncols=1)
# ax.plot(distances[:-1], avg_responses, c='cornflowerblue', zorder=1)
ax = axs[0]
ax2 = axs[1]
ax.step(distances[:-1], avg_responses, c='cornflowerblue', zorder=2)
# ax.fill_between(x=(distances-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray', zorder=0)
# ax.scatter(distances[:-1], avg_responses, c='orange', zorder=4)
ax.set_ylim([-0.5, 0.8])
ax.set_title(f'photostim responses vs. distance to sz wavefront (binned every {results.binned__distance_vs_photostimresponses["bin_width_um"]}um)', wrap=True)
ax.set_xlabel('distance to sz wavefront (um)')
ax.set_ylabel(main.response_type)
ax.margins(0)

pixels = [np.array(num2)] * 10
ax2.imshow(pixels, cmap='Greys', vmin=-5, vmax=150, aspect=0.1)
# ax.show()

fig.tight_layout(pad=1)
fig.show()


# todo add heatmap for number of datapoints across the distance bins




# todo refactor code into the results class here.
# todo run similar plot for temporal sz invasion analysis.



