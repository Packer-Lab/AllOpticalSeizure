"""
TODO:
[x] add bar plots + stats comparing correlation of targets within exp to across experiment
    - for baseline, and for interictal
[x] add bar plot + stats comparing avg targets correlation within baseline to within interictal


Suppl figure:
- normal distribution of targets responses (normalized across x-axis to avg dFF for each target)


"""

# %%
figname = 'aim3_2p-stim-dynamic-01'
import sys; sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure'])

import numpy as np
from funcsforprajay.plotting.plotting import plot_bar_with_points
from scipy.stats import ttest_rel

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults
from _exp_metainfo_.exp_metainfo import ExpMetainfo
from _utils_.alloptical_plotting import save_figure

main = PhotostimAnalysisSlmTargets
RESULTS: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

import sys
import matplotlib.pyplot as plt

# sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import rep_fig_vis as rfv

## Set general plotting parameters
# plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/thesis_figures/'

rfv.set_fontsize(ExpMetainfo.figures.fontsize['extraplot'])

# %%
## Set parameters
n_cat = 2
n_misc_rows = 2
n_misc = 5
colour_list = ['#101820', '#1b362c', '#2f553d', '#4f7553', '#79936f', '#aeae92']
colours_misc_dict = {xx: colour_list[xx] for xx in range(len(colour_list))}

save_fig = True

np.random.seed(2)  # fix seed

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound should be = l, b, r, t

layout = {
    'main-top': {'panel_shape': (10, 2),
                 'bound': (0.05, 0.77, 0.78, 0.95),
                 'hspace': 0.2},
    'main-middle-left': {'panel_shape': (1, 1),
                         'bound': (0.07, 0.55, 0.15, 0.70)},
    'main-middle-middle': {'panel_shape': (1, 1),
                           'bound': (0.23, 0.55, 0.47, 0.70),
                           'wspace': 0.2},
    'main-middle-corr-left': {'panel_shape': (1, 2),
                         'bound': (0.53, 0.55, 0.63, 0.70),
                         'hspace': 0.1},
    'main-middle-corr-right': {'panel_shape': (1, 1),
                          'bound': (0.72, 0.55, 0.80, 0.70),
                          'wspace': 0.2},
}

test = 0
save_fig = True if not test > 0 else False
dpi = 150 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.


# %% C - mean response vs. variability
axs = axes['main-middle-middle']
main.plot__mean_response_vs_variability(fig, axs=axs, rerun=0, fontsize=ExpMetainfo.figures.fontsize['extraplot'])
rfv.add_label_axes(text='C', ax=axs[0], x_adjust=0.1)



# %% A, B - photostimulation variability across trials - quantified by CV

ax = axes['main-top'][0, 0]  #: CV representative examples
main.plot_variability_photostim_traces_by_targets(axs=axes['main-top'], fig=fig)
rfv.add_label_axes(text='A', ax=ax, y_adjust=0)
axes['main-top'][0, 0].text(s='Baseline', x = -4, y=0, rotation=90, fontsize=10)
axes['main-top'][0, 1].text(s='Interictal', x = -4, y=0, rotation=90, fontsize=10)


ax = axes['main-middle-left'][0]  #: CV quantification bar plot
main.plot__variability(fig=fig, ax=ax, fontsize=ExpMetainfo.figures.fontsize['extraplot'])
rfv.add_label_axes(text='B', ax=ax, x_adjust = 0.09)



# %% F - #: within exp, across targets correlation magnitudes // PCA eigen value decomposition
"""note: currently computing filtered on MID interictal stims. change one of the lines below to be able to compute on all interictal stims"""

# main.collect_all_targets_all_exps(run_pre4ap_trials=True, run_post4ap_trials=True)

results = RESULTS
# results.baseline_adata.obs['exp']
# results.interictal_adata.obs['exp']

ax = axes['main-middle-corr-right'][0]
rfv.add_label_axes(text='F', ax=ax, x_adjust=0.095)

# avg correlation magnitude
# main.correlation_magnitude_exps(fig=fig, axs=ax)

# STATS TEST
baseline = list(results.corr_targets['within - baseline'].values())[2:]  # skipping over PS04 because of improper photostim responses for correlation measurements
midinterictal = list(results.corr_targets['within - midinterictal'].values())[1:]
assert len(midinterictal) == len(
    baseline), f'mismatch in number of experiments in midinterictal {len(midinterictal)} and baseline {len(baseline)}'

# stats_score = ttest_ind(midinterictal, baseline)
stats_score = ttest_rel(midinterictal, baseline)
print(f"Figure 4F, baseline: {np.mean(baseline):.2f}, interictal (middle): {np.mean(midinterictal):.2f}, paired t-test: *p = {stats_score[1]: .2e}")
print(f"paired t-test score, baseline vs. mid-interictal: {stats_score}")

# MAKE PLOT
axs = ax
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(3, 4)) if fig is None and axs is None else (fig, axs)
plot_bar_with_points(data=[baseline, midinterictal], paired=True, fontsize=10, bar=False,
                     x_tick_labels=['Baseline', 'Interictal'], colors=['royalblue', 'forestgreen'],
                     y_label='Correlation (R)', show=False, alpha=1, fig=fig, ax=axs, s=15, ylims=[0, 1.0],
                     sig_compare_lines={'*': [0, 1]}, points_lw=0.5)
# axs.text(x=2.5, y=0.025, s=f't-test rel.: {stats_score[1]:.2e}', fontsize=3)



# %% E - correlation matrix of z scored responses across targets - selected one experiment
axs = (axes['main-middle-corr-left'],)  #: within exp, across targets correlation matrixes
main.correlation_matrix_all_targets(fig=fig, axs=axs)
rfv.add_label_axes(text='E', ax=axs[0][0])
# rfv.add_label_axes(text="F'", ax=axs[1][0], y_adjust=0)

# for ax in axes['main-middle-corr-left']:
#     rfv.naked(ax)
#     # ax.axis('off')







# %%
if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{figname}.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{figname}.pdf")

fig.show()

 
