"""
TODO:
[x] add bar plots + stats comparing correlation of targets within exp to across experiment
    - for baseline, and for interictal
[x] add bar plot + stats comparing avg targets correlation within baseline to within interictal


Suppl figure - todo make:
- normal distribution of targets responses (normalized across x-axis to avg dFF for each target)
- corr matrices for each exp to exp comparison of baseline and interictal

"""

# %%

import numpy as np
import pandas as pd
from funcsforprajay.funcs import flattenOnce
from funcsforprajay.plotting.plotting import plot_bar_with_points
from scipy.stats import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets, plot__avg_photostim_dff_allexps
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults, \
    PhotostimResponsesQuantificationSLMtargets
from _utils_.alloptical_plotting import plot_settings
from alloptical_utils_pj import save_figure

import statsmodels.api as sm
from statsmodels.formula.api import ols

main = PhotostimAnalysisSlmTargets
RESULTS: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

import sys
import matplotlib.pyplot as plt
from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import matplotlib.image as mpimg
import rep_fig_vis as rfv

## Set general plotting parameters
# plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

fontsize = 8
rfv.set_fontsize(fontsize)

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
                         'bound': (0.10, 0.55, 0.18, 0.69)},
    # 'main-middle-middle': {'panel_shape': (2, 1),
    #                        'bound': (0.29, 0.55, 0.69, 0.69),
    #                        'wspace': 0.2},
    'main-middle-middle': {'panel_shape': (1, 1),
                           'bound': (0.30, 0.55, 0.55, 0.69),
                           'wspace': 0.2},
    'main-middle-right': {'panel_shape': (1, 1),
                          'bound': (0.68, 0.55, 0.78, 0.69)},
    'main-bottom-left': {'panel_shape': (2, 1),
                         'bound': (0.05, 0.35, 0.3, 0.47),
                         'wspace': 0.1},
    # 'main-bottom-low': {'panel_shape': (2, 1),
    #                     'bound': (0.04, 0.10, 0.52, 0.255),
    #                     'wspace': 0.01},
    'main-bottom-right': {'panel_shape': (1, 1),
                          'bound': (0.48, 0.36, 0.55, 0.45),
                          'wspace': 0.2},
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=300)


# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.


# %% A, B - add plots to axes

ax = axes['main-top'][0, 0]  #: CV representative examples
main.plot_variability_photostim_traces_by_targets(axs=axes['main-top'], fig=fig)
rfv.add_label_axes(text='A', ax=ax, y_adjust=0)
axes['main-top'][0, 0].text(s='Baseline', x = -4, y=0, rotation=90, fontsize=10)
axes['main-top'][0, 1].text(s='Interictal', x = -4, y=0, rotation=90, fontsize=10)


ax = axes['main-middle-left'][0]  #: CV quantification bar plot
main.plot__variability(fig=fig, ax=ax, fontsize=fontsize)
rfv.add_label_axes(text='B', ax=ax, x_adjust = 0.09)


# %% C - mean response vs. variability
axs = axes['main-middle-middle']
main.plot__mean_response_vs_variability(fig, axs=axs, rerun=0, fontsize=fontsize)
rfv.add_label_axes(text='C', ax=axs[0], x_adjust=0.1)

# %% D - splitting responses during interictal phases
ax = axes['main-middle-right'][0]  #: interictal split - z scores

# 1-WAY ANOVA
stats.f_oneway(RESULTS.interictal_responses['preictal_responses'],
               RESULTS.interictal_responses['very_interictal_responses'],
               RESULTS.interictal_responses['postictal_responses'])

# create DataFrame to hold data
data_nums = []
num_pre = len(RESULTS.interictal_responses['preictal_responses'])
num_mid = len(RESULTS.interictal_responses['very_interictal_responses'])
num_post = len(RESULTS.interictal_responses['postictal_responses'])
data_nums.extend(['pre'] * num_pre)
data_nums.extend(['mid'] * num_mid)
data_nums.extend(['post'] * num_post)

df = pd.DataFrame({'score': flattenOnce([RESULTS.interictal_responses['preictal_responses'],
                                         RESULTS.interictal_responses['very_interictal_responses'],
                                         RESULTS.interictal_responses['postictal_responses']]),
                   'group': data_nums})

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'],
                          alpha=0.05)

print(tukey)


# fig, ax = plt.subplots(figsize=(3,4))
data = [RESULTS.interictal_responses['preictal_responses'],
        RESULTS.interictal_responses['very_interictal_responses'],
        RESULTS.interictal_responses['postictal_responses']]
plot_bar_with_points(data=data, bar=False, title='', fontsize=fontsize,
                     x_tick_labels=['Pre', 'Mid', 'Post'], colors=['lightseagreen', 'gold', 'lightcoral'],
                     y_label='Responses ($\it{z}$-scored)', show=False, ylims=[-0.5, 0.8],
                     alpha=1, fig=fig, ax=ax, s=25, sig_compare_lines={'*': [1, 2]})

rfv.add_label_axes(text='D', ax=ax, x_adjust=0.1)

rfv.test_axes_plot(ax=ax)




# %% E - correlation matrix of z scored responses across targets - selected one experiment
axs = (axes['main-bottom-left'],)  #: within exp, across targets correlation matrixes
main.correlation_matrix_all_targets(fig=fig, axs=axs)
rfv.add_label_axes(text='E', ax=axs[0][0], y_adjust=0)
# rfv.add_label_axes(text="F'", ax=axs[1][0], y_adjust=0)

for ax in axes['main-bottom-left']:
    rfv.naked(ax)
    ax.axis('off')




# %% F - #: within exp, across targets correlation magnitudes // PCA eigen value decomposition
# main.collect_all_targets_all_exps(run_pre4ap_trials=True, run_post4ap_trials=True)

results = RESULTS
# results.baseline_adata.obs['exp']
# results.interictal_adata.obs['exp']

axs = axes['main-bottom-right']
rfv.add_label_axes(text='F', ax=axs[0], y_adjust=0.015, x_adjust=0.075)

# fig2, ax2 = plt.subplots(figsize = (3, 2))
pca_stats = main.pca_responses(fig=fig, ax=axs[0], rerun=0, results=RESULTS, fontsize=fontsize)
axs[0].set_ylabel('Explained variance')
axs[0].set_xlabel('PC')
axs[0].set_ylim([0, 0.6])

pca_ev_exps = pd.DataFrame({
    'group': [],
    'expid': [],
    'PC': [],
    'EV': []
})

counter = 0
pca_ev_exps = []
for id, group, pca_stat in pca_stats:
    color = 'royalblue' if 'pre' in group else 'forestgreen'
    axs[0].plot(range(1, 1 + 10), pca_stat, color=color, lw=0.7, marker='o', markersize = 3, markerfacecolor=color,
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

# avg correlation magnitude
# main.correlation_magnitude_exps(fig=fig, axs=axs)

# Create the legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

legend_elements = [Line2D([0], [0], color='royalblue', label='Baseline',
                          markerfacecolor='royalblue', markersize=6, markeredgecolor='black'),
                   Line2D([0], [0], color='forestgreen', label='Interictal',
                          markerfacecolor='forestgreen', markersize=6, markeredgecolor='black')]

bbox = np.array(axs[0].get_position())
bbox = Bbox.from_extents(bbox[1, 0] + 0.02, bbox[1, 1] - 0.02, bbox[1, 0] + 0.06, bbox[1, 1])
axlegend = fig.add_subplot()
axlegend.set_position(pos=bbox)
axlegend.legend(handles=legend_elements, loc='center')
axlegend.axis('off')

axs[0].text(x=8, y=0.2,
            s=f'C(group): {twoway.loc["C(group)", "PR(>F)"]:.2e}, C(group):C(PC): {twoway.loc["C(group):C(PC)", "PR(>F)"]:.2e}',
            fontsize=5)


# %%
if save_fig:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure4-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure4-RF.svg")

fig.show()
