# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS - FOCUS ON THE SEIZURES!
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj
import tifffile as tf
from skimage.transform import resize
from mpl_toolkits import mplot3d

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/Nontargets_responses_2021-11-11'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


expobj, experiment = aoutils.import_expobj(aoresults_map_id='post e.1')  # PLACEHOLDER IMPORT OF EXPOBJ TO MAKE THE CODE WORK

# %% 1) SEIZURE WAVEFRONT PLOTTING AND ANALYSIS
"""################################# SEIZURE EVENTS PLOTTING ##############################################################
########################################################################################################################

# PLOT HEATMAP of SEIZURE EVENTS
"""
sz = 2
sz_onset, sz_offset = expobj.stims_bf_sz[sz], expobj.stims_af_sz[sz+1]

# -- approach of dFF normalize to the mean of the Flu data 2 seconds before the seizure
pre_sz = 2*int(expobj.fps)
sz_flu = expobj.raw[[expobj.cell_id.index(cell) for cell in expobj.good_cells], sz_onset - pre_sz: sz_offset]
sz_flu_smooth = np.array([pj.smooth_signal(signal, w=5) for signal in sz_flu])  # grouped average of the raw signal
x_norm = np.array([pj.dff(flu[pre_sz:], np.mean(flu[:pre_sz])) * 100 for flu in sz_flu_smooth])


stims = [(stim - sz_onset) for stim in expobj.stim_start_frames if sz_onset <= stim < sz_offset]
stims_off = [(stim + expobj.stim_duration_frames - 1) for stim in stims]

x_bf = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_bf_sz[sz])[0][0]]
x_af = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_af_sz[sz+1])[0][0]]

lfp_signal = expobj.lfp_signal[x_bf:x_af]

# -- ordering cells based on their order of reaching top 5% signal threshold
x_95 = [np.percentile(trace, 95) for trace in x_norm]

x_peak = [np.min(np.where(x_norm[i] > x_95[i])) for i in range(len(x_norm))]
new_order = np.argsort(x_peak)
x_ordered = x_norm[new_order]

# plot heatmap of dFF processed Flu signals for all cells for selected sz and ordered as determined above
aoplot.plot_traces_heatmap(x_ordered, stim_on=stims, stim_off=stims_off, cmap='Spectral_r', figsize=(10, 6),
                           title=('%s - seizure %s - sz flu smooth - %s to %s' % (trial, sz, sz_onset, sz_offset)),
                           xlims=None, vmin=100, vmax=500, lfp_signal=lfp_signal)

# just the bottom half cells that seems to show more of an order
x_ordered = x_norm[new_order[250:]]
aoplot.plot_traces_heatmap(x_ordered, stim_on=stims, stim_off=stims_off, cmap='Spectral_r', figsize=(10, 6),
                           title=('%s - seizure %s - sz flu smooth - %s to %s' % (trial, sz, sz_onset, sz_offset)),
                           xlims=None, vmin=100, vmax=500, lfp_signal=lfp_signal)


# PLOT cell location with cmap based on their order of reaching top 5% signal during sz event

cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
aoplot.plot_cells_loc(expobj, cells=cell_ids_ordered, show_s2p_targets=False, color_float_list=list(range(len(cell_ids_ordered))),
                      title='cell locations ordered by recruitment in sz # %s' % sz, invert_y=True, cmap='Purples')

# just the bottom half cells that seems to show more of an order
cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
aoplot.plot_cells_loc(expobj, cells=cell_ids_ordered[250:], show_s2p_targets=False, color_float_list=list(np.array(x_peak)[new_order][250:]),
                      title='cell locations ordered by recruitment in sz # %s' % sz, invert_y=True, cmap='Purples')





# %% 2) 4D plotting of the seizure wavefront (2D space vs. time vs. Flu intensity)

# import expobj
expobj, experiment = aoutils.import_expobj(aoresults_map_id='post h.0', do_processing=False)
aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Frames', figsize=[20, 3])


downsampled_tiff_path = f"{expobj.tiff_path[:-4]}_downsampled.tif"

# get the 2D x time Flu array
print(f"loading tiff from: {downsampled_tiff_path}")
im_stack = tf.imread(downsampled_tiff_path)
print('|- Loaded experiment tiff of shape: ', im_stack.shape)

# binning down the imstack to a reasonable size
im_binned_ = resize(im_stack, (im_stack.shape[0], 100, 100))

# building up the flat lists to use for 3D plotting....

start = 10500//4
end = 16000//4
sub = end - start

im_binned = im_binned_[start:end, :, :]

x_size=im_binned.shape[1]
y_size=im_binned.shape[2]
t_size=sub

time = np.asarray([np.array([[i]*x_size]*y_size) for i in range(t_size)])

x = np.asarray(list(range(x_size))*y_size*t_size)
y = np.asarray([i_y for i_y in range(y_size) for i_x in range(x_size)] * t_size)
t = time.flatten()
flu = im_binned.flatten()

im_array = np.array([x, y, t], dtype=np.float)

assert len(x) == len(y) == len(t), print(f'length mismatch between x{len(x)}, y{len(y)}, t{len(t)}')

# plot 3D projection scatter plot
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
ax.scatter(im_array[2], im_array[1], im_array[0], c=flu, cmap='Reds', linewidth=0.5, alpha=0.005)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.grid(False)

ax.set_xlabel('time (frames)')
ax.set_ylabel('y axis')
ax.set_zlabel('x axis')
fig.tight_layout()
fig.show()
