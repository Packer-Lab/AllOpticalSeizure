"""
TODO:
[ ] combining the old fig 6 and 7 into one figure

- make schematic for distance of nontargets to seizure analysis
    fig legend for this: Non-targeted neurons were split into proximal (pink, < 100um from seizure wavefront) and distal (gold, > 200um from) to seizure wavefront groups for all trials to analyse their photostimulation response during ictal states.

- double check distances, etc. is everything in place - responses correct? distances correct?


STATS:
- change nontargets response magnitude stats to one way anova



suppl figure: write up RF code
- seizure boundary classification
- responses vs. pre-stim Flu targets annulus, across all conditions
    + stats tests on the averages? or can tests be carried out directly on the 2-D density plots?


"""
# %%
figname = 'aim3_2p-sz-excitability-extra'
import sys;

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure'])

import _utils_.alloptical_plotting
from _exp_metainfo_.exp_metainfo import ExpMetainfo

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
import rep_fig_vis as rfv

import numpy as np

from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview, \
    TargetsSzInvasionSpatialResults_codereview

SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-responses-sz-distance/"

main_spatial = TargetsSzInvasionSpatial_codereview
results_spatial = TargetsSzInvasionSpatialResults_codereview.load()

SAVE_FOLDER = f'/home/pshah/Documents/figures/thesis_figures/'

distance_lims = [19, 400]  # limit of analysis

# %% SETUP
## Set general plotting parameters

fs = ExpMetainfo.figures.fontsize['extraplot']
rfv.set_fontsize(fs)

np.random.seed(2)  # fix seed

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

layout = {
    'left': {'panel_shape': (1, 1),
             'bound': (0.10, 0.77, 0.45, 0.92)},
    'right': {'panel_shape': (1, 1),
              'bound': (0.55, 0.77, 0.90, 0.92),
              'wspace': 0.2}
}

test = 0
save_fig = True if not test > 0 else False
dpi = 100 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes,
                            show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# ADD PLOTS TO AXES  ##################################################################################################################
# %% A - dF/prestimF stim responses of targets vs. distance to seizure wavefront

# quick figure of responses vs distance, for dF/prestimF
ax = axes['left'][0]
# fig, ax = plt.subplots(figsize=(6,3), dpi=200)
main_spatial.plot__responses_v_distance_no_normalization_rolling_bins(results=results_spatial, fig=fig, axes=[ax], response_type='SLM Targets photostim responses (dF/prestimF)')
ax.axhline(0, ls='--', lw=1, color='black', zorder=0)
# ax.set_ylabel('dF/prestimF\n(norm. to baseline)')
# ax.set_ylabel('dF/stdF\n(norm. to baseline)')
# ax.set_ylabel('dFF\n(norm. to baseline)')
ax.set_title(f'Neuronal excitability \n in seizure penumbra')
ax.set_ylabel(r'Responses (dF/F)')
ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)')

rfv.add_label_axes(text='A', ax=ax, x_adjust=0.06)

# %% B - regression fit on z-scored responses 
bin_width = results_spatial.rolling_binned__distance_vs_photostimresponses['bin_width_um']
avg_responses = results_spatial.rolling_binned__distance_vs_photostimresponses['avg_photostim_response_in_bin'][
                :-bin_width]
responses = results_spatial.rolling_binned__distance_vs_photostimresponses['all responses (per bin)'][:-bin_width]
distances = results_spatial.rolling_binned__distance_vs_photostimresponses['distance_bins'][:-bin_width]


def regression_fit(x, y, **kwargs):
    """
    Fit a logarithmic regression function to the data
    :param x: x data
    :param y: y data
    :return: a, b
    """
    # plot distances vs. avg responses as scatter plot using matplotlib
    for i, dist in enumerate(x):
        kwargs['ax'].scatter(dist, avg_responses[i], s=3, alpha=1, color='gray')
    # ax.set_ylim([-2, 2.25])

    ## CURVE FITTING USING NUMPY POLYFIT
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    x_range = np.linspace(x[0], x[-1], 10000)
    y_pred = p(x_range)
    # find y intercept
    y_intercept = x_range[np.where(y_pred < 0)[0][-1]]
    print(f'3rd-order poly y-intercept: {y_intercept}')
    ## TEST PLOT OF y_pred
    if 'ax' in kwargs:
        kwargs['ax'].plot(x_range, y_pred, lw=2, color='blue', ls='--', label='3rd-poly')
        # plot y-intercept as scatter point on plot
        # kwargs['ax'].scatter(y_intercept, 0, s=10, color='red', label='y-intercept')

    # Check the accuracy :
    from sklearn.metrics import r2_score
    y_pred = p(x)
    Accuracy = r2_score(y, y_pred)
    print(f'R**2 (polyfit func): {Accuracy}\n')

    ## CURVE_FITING USING SCIPY - for LINEAR and LOG FIT
    def log_func(x, a, b):
        return a + b * np.log(x)

    def lin_func(x, a, b):
        return a + b * x

    # Finding the optimal parameters: LOG FUNCTION
    # change first value to 0.1 to avoid log(0) error
    __filter = np.where(np.array(x) == 0)[0][-1] + 1
    x_forlog = x[__filter:]
    y_forlog = y[__filter:]

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(log_func, x_forlog, y_forlog)
    print("a  = ", popt[0])
    print("b  = ", popt[1])

    # Predicting values:
    y_pred = log_func(x_forlog, popt[0], popt[1])

    # ## TEST PLOT OF y_pred
    # if 'ax' in kwargs:
    #     kwargs['ax'].plot(x_forlog, y_pred, lw=2, color='r', ls='--', label='log')

    # # Check the accuracy :
    # Accuracy = r2_score(y_forlog, y_pred)
    # print(f'R**2 (log func): {Accuracy}\n')

    # Finding the optimal parameters: LINEAR FUNCTION
    # change first value to 0.1 to avoid log(0) error
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(lin_func, x, y)
    print("a  = ", popt[0])
    print("b  = ", popt[1])

    # Predicting values:
    y_pred = lin_func(np.array(x, dtype=np.float64), popt[0], popt[1])

    # ## TEST PLOT OF y_pred
    # if 'ax' in kwargs:
    #     kwargs['ax'].plot(x, y_pred, lw=2, color='g', ls='--', label='linear')

    # # Check the accuracy :
    # Accuracy = r2_score(y, y_pred)
    # print(f'R**2 (lin func): {Accuracy}\n')

    # add legend
    # kwargs['ax'].legend()
    kwargs['ax'].axhline(0, zorder=0, c='black')
    # kwargs['fig'].show()

    return popt[0], popt[1], y_pred, y_intercept


# RUN LOGARITHMIC CURVE FIT
# fig, ax = plt.subplots(figsize=[5,3])
_,_, _, y_intercept = regression_fit(distances, avg_responses, ax=axes['right'][0])

# fig.show()
ax = axes['right'][0]
ax.set_title(f'Neuronal excitability \n in seizure penumbra')
ax.set_ylabel('Responses (z-score)')
ax.set_xlim([0, 450])
ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)')

rfv.add_label_axes(text='B', ax=ax, x_adjust=0.06)

# %% add plots to axes
if save_fig and dpi >= 250:
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{figname}.png")
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{figname}.pdf")

fig.show()
