# %%
import sys

from funcsforprajay.plotting.plotting import plot_bar_with_points
from matplotlib.transforms import Bbox
from scipy import stats

from _alloptical_utils import run_for_loop_across_exps
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import plot_settings
from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure

import cairosvg
from PIL import Image
from io import BytesIO

from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rep_fig_vis as rfv

plot_settings()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

date = '2021-01-24'
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial


# %% MAKE FIGURE LAYOUT  # TODO need to update for Figure 2!
layout = {
    'main-left': {'panel_shape': (1, 1),
                'bound': (0.05, 0.64, 0.30, 0.98)},
    'toprighttop': {'panel_shape': (2, 1),
                    'bound': (0.31, 0.86, 0.90, 0.95)},
    'toprightbottom': {'panel_shape': (2, 1),
                       'bound': (0.31, 0.64, 0.90, 0.85)},
    'bottom': {'panel_shape': (3, 1),
               'bound': (0.05, 0.40, 0.63, 0.57),
               'wspace': 0.4}
}

# fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=300)

# rfv.naked(axes['main-left'][0])

# %% F) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period

# run data analysis
exp_sz_occurrence, total_sz_occurrence = OnePhotonStimAnalysisFuncs.collectSzOccurrenceRelativeStim()


# make plot
bin_width = int(1 * expobj.fps)
period = len(np.arange(0, (expobj.stim_interval_fr // bin_width)))
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr // bin_width)) / period

bbox = Bbox.from_extents(0.70, 0.41, 0.85, 0.56)
_axes = np.empty(shape=(1,1), dtype=object)
# ax = fig.add_subplot(projection = 'polar')
# ax.set_position(pos=bbox)
# rfv.add_label_axes(s='F', ax=ax, y_adjust=0.035)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=50)

# by experiment
# for exp, values in exp_sz_occurrence.items():
#     plot = values
#     ax.bar(theta, plot, width=(2 * np.pi) / period, bottom=0.0, alpha=0.5)

# across all seizures
total_sz = np.sum(np.sum(total_sz_occurrence, axis=0))
sz_prob = np.sum(total_sz_occurrence, axis=0) / total_sz

ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=0.5)

ax.set_rmax(1.1)
ax.set_rticks([0.25, 0.5, 0.75, 1])  # radial ticks
ax.set_rlabel_position(-60)  # Move radial labels away from plotted line
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels(['0', '', '50', ''])
ax.set_title("sz probability occurrence (binned every 1s)", va='bottom')
ax.spines['polar'].set_visible(False)
fig.show()




