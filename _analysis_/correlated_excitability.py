"""
original previous Figure created for thesis and previous draft of


"""

# %%

import numpy as np
import pandas as pd
from funcsforprajay.funcs import flattenOnce
from funcsforprajay.plotting.plotting import plot_bar_with_points
from scipy.stats import stats, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets, plot__avg_photostim_dff_allexps
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults, \
    PhotostimResponsesQuantificationSLMtargets
from _exp_metainfo_.exp_metainfo import ExpMetainfo
from _utils_.alloptical_plotting import plot_settings, save_figure

import statsmodels.api as sm
from statsmodels.formula.api import ols

main = PhotostimAnalysisSlmTargets
RESULTS: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

import matplotlib.pyplot as plt

# %% correlation matrix and magnitude of z scored responses across targets - across all experiments, BASELINE experiments only
main.correlation_matrix_all_targets()

# STATS TEST
baseline = list(RESULTS.corr_targets['within - baseline'].values())[2:]  # skipping over PS04 because of improper photostim responses for correlation measurements

# MAKE PLOT
fig, ax = plot_bar_with_points(data=[baseline], fontsize=10, bar=False,
                     x_tick_labels=['Experiments'], colors=['royalblue'],
                     y_label='Correlation (R)', show=False, alpha=1, ylims=[0, 0.5], figsize=(2,3), s=45, lw=1.5, capsize=5,
                     points_lw=1.5)
fig.show()
# fig.savefig('/home/pshah/Documents/figures/misc_plots/baseline_photostim_targets_correlation_bar.png', transparent=True)
save_figure(fig, save_path_full="/home/pshah/Documents/figures/misc_plots/baseline_photostim_targets_correlation_bar.png", transparent=True)
save_figure(fig, save_path_full="/home/pshah/Documents/figures/misc_plots/baseline_photostim_targets_correlation_bar.svg", transparent=True)
# axs.text(x=2.5, y=0.025, s=f't-test rel.: {stats_score[1]:.2e}', fontsize=3)


# %% within exp, across targets correlation magnitudes // PCA eigen value decomposition
"""note: currently computing filtered on MID interictal stims. change one of the lines below to be able to compute on all interictal stims"""

# main.collect_all_targets_all_exps(run_pre4ap_trials=True, run_post4ap_trials=True)

results = RESULTS
# results.baseline_adata.obs['exp']
# results.interictal_adata.obs['exp']

# avg correlation magnitude
# main.correlation_magnitude_exps(fig=fig, axs=ax)

# STATS TEST
baseline = list(results.corr_targets['within - baseline'].values())[2:]  # skipping over PS04 because of improper photostim responses for correlation measurements
midinterictal = list(results.corr_targets['within - midinterictal'].values())[1:]
assert len(midinterictal) == len(
    baseline), f'mismatch in number of experiments in midinterictal {len(midinterictal)} and baseline {len(baseline)}'

# stats_score = ttest_ind(midinterictal, baseline)
stats_score = ttest_rel(midinterictal, baseline)
print(f"baseline: {np.mean(baseline):.2f}, interictal (middle): {np.mean(midinterictal):.2f}, paired t-test: *p = {stats_score[1]: .2e}")
print(f"paired t-test score, baseline vs. mid-interictal: {stats_score}")

# MAKE PLOT
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(3, 4))
plot_bar_with_points(data=[baseline, midinterictal], paired=True, fontsize=10, bar=False,
                     x_tick_labels=['Baseline', 'Interictal'], colors=['royalblue', 'forestgreen'],
                     y_label='Correlation (R)', show=False, alpha=1, fig=fig, ax=axs, s=15, ylims=[0, 1.0],
                     sig_compare_lines={'*': [0, 1]}, points_lw=0.5)
# axs.text(x=2.5, y=0.025, s=f't-test rel.: {stats_score[1]:.2e}', fontsize=3)

# %% mean response vs. variability
main.plot__mean_response_vs_variability(fig, axs=axs, rerun=0, fontsize=ExpMetainfo.figures.fontsize['extraplot'])

# %% correlation matrix of z scored responses across targets - selected one experiment
main.correlation_matrix_all_targets(fig=fig, axs=axs)


# %% photostimulation variability across trials - quantified by CV

main.plot_variability_photostim_traces_by_targets()
main.plot__variability(fontsize=ExpMetainfo.figures.fontsize['extraplot'])

# %%

## 4F - PCA APPROACH FOR COMPARING BASELINE VS. INTERICTAL STIMS REPSONE CORRELATIONS
pca_stats = main.pca_responses(rerun=0, results=RESULTS, fontsize=10)

pca_ev_exps = pd.DataFrame({
    'group': [],
    'expid': [],
    'PC': [],
    'EV': []
})

counter = 0
pca_ev_exps = []
fig, ax = plt.subplots(figsize=(5,5))
for id, group, pca_stat in pca_stats:
    color = 'royalblue' if 'pre' in group else 'forestgreen'
    ax.plot(range(1, 1 + 10), pca_stat, color=color, lw=0.7, marker='o', markersize = 3, markerfacecolor=color,
                markeredgecolor='white', mew=0.3)
    for i, ev in enumerate(pca_stat):
        _group = 'baseline' if 'pre' in group else 'interictal'
        _df = pd.DataFrame({
            'group': _group,
            'expid': id,
            'PC': i,
            'EV': ev,
        }, index=[counter])
        pca_ev_exps.append(_df)
        counter += 1

pca_ev_exps = pd.concat(pca_ev_exps)

#### RUN TWO WAY ANOVA:
# Performing two-way ANOVA on PCA ANALYSIS
model = ols('EV ~ C(group) + C(PC) + C(group):C(PC)', data=pca_ev_exps).fit()
twoway = sm.stats.anova_lm(model, typ=2)


# Create the legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

legend_elements = [Line2D([0], [0], color='royalblue', label='Baseline',
                          markerfacecolor='royalblue', markersize=6, markeredgecolor='black'),
                   Line2D([0], [0], color='forestgreen', label='Interictal',
                          markerfacecolor='forestgreen', markersize=6, markeredgecolor='black')]

bbox = np.array(ax.get_position())
bbox = Bbox.from_extents(bbox[1, 0] + 0.02, bbox[1, 1] - 0.02, bbox[1, 0] + 0.06, bbox[1, 1])
axlegend = fig.add_subplot()
axlegend.set_position(pos=bbox)
axlegend.legend(handles=legend_elements, loc='center')
axlegend.axis('off')

ax.text(x=8, y=0.2, s=f'C(group): {twoway.loc["C(group)", "PR(>F)"]:.2e}, C(group):C(PC): {twoway.loc["C(group):C(PC)", "PR(>F)"]:.2e}', fontsize=5)
fig.show()
