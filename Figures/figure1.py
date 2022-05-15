## FIGURE 1 - LIVE IMAGING OF SEIZURES IN AWAKE ANIMALS
import sys

from funcsforprajay.funcs import smoothen_signal

from _results_.sz4ap_results import plotHeatMapSzAllCells

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

from _utils_.io import import_expobj

import _alloptical_utils as Utils
from _utils_ import alloptical_plotting as aoplot

import numpy as np
import matplotlib.pyplot as plt

from _main_.Post4apMain import Post4ap

# %% 3) plotting and or calculating time to sz invasion and speed of sz propagation across FOV (um^2 / sec)
#   - speed of propagation = rate of sz area recruitment per unit of time

# marking invasion of sz based on 1st or 2nd derivative of smoothed mean FOV signal


expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')

fig, axs = plt.subplots(figsize=(12, 4))
fig, axs = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='time', fig=fig, ax=axs, show=False)
for x in expobj.seizure_frames:
    axs.axvline(x=x, zorder=0, color='gray')
axs.set_xlim([int(150 * expobj.fps), int(260 * expobj.fps)])
fig.show()


# %%

fov_signal_frames = [frame for frame in range(expobj.n_frames) if frame not in expobj.photostim_frames]  # - exclude photostim frames

seizure_frames_adjusted = []
for frame in expobj.seizure_frames:
    if frame in fov_signal_frames:
        sub_ = len(np.where(frame > expobj.photostim_frames)[0])
        seizure_frames_adjusted.append((frame - sub_))


fov_signal = [expobj.meanRawFluTrace[fr] for fr in expobj.frames_photostim_ex]
smoothed_fov = smoothen_signal(fov_signal, w=int(expobj.fps * 1))
first_d = np.gradient(fov_signal)
plot = smoothen_signal(first_d, w=int(expobj.fps * 1))
second_d = np.gradient(plot)
# plot = second_d

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax2 = ax.twinx()
ax.plot(smoothed_fov, color='black', zorder=5)
ax2.plot(plot, color='orange', zorder=5)
ax2.axhline(y=0)
for x in seizure_frames_adjusted:
    ax.axvline(x=x, zorder=0, color='gray')
ax.set_xlim([3000, 6500])
fig.show()


# %% 3.1) calulating time to first 50% of max of first derivative


# %% 3.2) calculating the first in sz stim frame with wavefront




# %% 1) suite2p cells gcamp imaging for seizures, with simultaneous LFP recording

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')

# fig, axs = plt.subplots(2, 1, figsize=(6, 6))
# fig, axs[0] = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='time', fig=fig, ax=axs[0], show=False)
# fig, axs[1] = aoplot.plotLfpSignal(expobj=expobj, stim_span_color='', x_axis='time', fig=fig, ax=axs[1], show=False)
# axs[0].set_xlim([400 * expobj.fps, 470 * expobj.fps])
# axs[1].set_xlim([400 * expobj.paq_rate, 470 * expobj.paq_rate])
# fig.show()

# plot heatmap of raw neuropil corrected s2p signal from s2p cells
time = (400, 460)
frames = (time[0] * expobj.fps, time[1] * expobj.fps)
paq = (time[0] * expobj.paq_rate, time[1] * expobj.paq_rate)

plotHeatMapSzAllCells(expobj=expobj, sz_num=4)

# plotting recruitment of cells across space.
#   - measuring recruitment of each cell as 60% of maximum signal (after smoothing the signal)


# %% 2) red channel image of 4ap injection

# path of image at edge of injection site:
# /Users/prajayshah/data/oxford-data-to-process/2021-01-31/2021-01-31_s-007/2021-01-31_s-007_Cycle00001_Ch2_000001.ome.tif




# %% how many seizure propagate vs. remain stationary in the FOV




