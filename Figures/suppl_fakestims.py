"""
Supplementary Figure: Artificial stimulation responses of targets.

A: Timing signal of experimental photostimulation trial during a single experiment.

"""
import sys

import numpy as np
from funcsforprajay.plotting.plotting import plot_bar_with_points
from matplotlib.transforms import Bbox
from scipy import stats

from _alloptical_utils import run_for_loop_across_exps
from _utils_.paq_utils import paq_read
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import plot_settings
from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure

from _analysis_._ClassPhotostimAnalysisSlmTargets import plot_peristim_avg_fakestims

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import matplotlib.pyplot as plt
import rep_fig_vis as rfv

plot_settings()


SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

rfv.set_fontsize(10)

save_fig = True


# %% MAKE FIGURE LAYOUT
layout = {
    'left': {'panel_shape': (1, 1),
          'bound': (0.10, 0.80, 0.70, 0.95)},
    'bottom-left': {'panel_shape': (1, 1),
          'bound': (0.10, 0.45, 0.30, 0.65)},
    # 'bottom-right': {'panel_shape': (1, 1),
    #       'bound': (0.40, 0.45, 0.60, 0.65)},
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=300)

# rfv.show_test_figure_layout(fig, axes=axes, show=1)  # test what layout looks like quickly, but can also skip and moveon to plotting data.


# %% plotting fakestim timings

expobj: Post4ap = import_expobj(exp_prep='RL109 t-013')

paq, _ = paq_read(expobj.paq_path, plot=False)
print(paq['chan_names'])
chan_num = paq['chan_names'].index('markpoints2packio')

# %%
ax = axes['left'][0]
rfv.add_label_axes(text='A', ax=ax)

stim_signal = paq['data'][chan_num][expobj.frame_start_time_actual:]
x_range = np.linspace(0, len(stim_signal)/expobj.paq_rate, len(stim_signal))
ax.plot(x_range, stim_signal, color='black', label='photostimulation trial')
fake_stim_times = [expobj.frame_time(fr) for fr in expobj.fake_stim_start_frames]
ax.scatter(fake_stim_times, [0.5] * len(fake_stim_times), s=5, color='orange', label='artifical stim. trial')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([5, 112])
ax.set_xlabel('Time (secs)')
ax.set_ylabel('Stim timing signal')


# %% bottom left

ax = axes['bottom-left'][0]
rfv.add_label_axes(text='B', ax=ax)

plot_peristim_avg_fakestims(fig, axs=axes['bottom-left'])

# %% bottom right

# expobj.PhotostimResponsesNonTargets.fakestims_allopticalAnalysisNontargets(expobj=expobj)

# %%
# fig.show()

if save_fig:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure-suppl-fakestims-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure-suppl-fakestims-RF.svg")






