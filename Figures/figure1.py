## FIGURE 1 - LIVE IMAGING OF SEIZURES IN AWAKE ANIMALS
import sys

from _results_.sz4ap_results import plotHeatMapSzAllCells

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

from _utils_.io import import_expobj

import _alloptical_utils as Utils
from _utils_ import alloptical_plotting as aoplot

import numpy as np
import matplotlib.pyplot as plt

from _main_.Post4apMain import Post4ap


# %% 1) suite2p cells gcamp imaging for seizures, with simultaneous LFP recording

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')




# %%
# fig, axs = plt.subplots(2, 1, figsize=(6, 6))
# fig, axs[0] = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='time', fig=fig, ax=axs[0], show=False)
# fig, axs[1] = aoplot.plotLfpSignal(expobj=expobj, stim_span_color='', x_axis='time', fig=fig, ax=axs[1], show=False)
# axs[0].set_xlim([400 * expobj.fps, 470 * expobj.fps])
# axs[1].set_xlim([400 * expobj.paq_rate, 470 * expobj.paq_rate])
# fig.show()

# %% plot heatmap of raw neuropil corrected s2p signal from s2p cells

time = (400, 460)
frames = (time[0] * expobj.fps, time[1] * expobj.fps)
paq = (time[0] * expobj.paq_rate, time[1] * expobj.paq_rate)

plotHeatMapSzAllCells(expobj=expobj, sz_num=4)


# %% 2) red channel image of 4ap injection

# path of image at edge of injection site:
# /Users/prajayshah/data/oxford-data-to-process/2021-01-31/2021-01-31_s-007/2021-01-31_s-007_Cycle00001_Ch2_000001.ome.tif


# %% plotting recruitment of cells across space.
#   - measuring recruitment of each cell as 60% of maximum signal (after smoothing the signal)





# %% how many seizure propagate vs. remain stationary in the FOV




