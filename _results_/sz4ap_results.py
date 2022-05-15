# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS - FOCUS ON THE SEIZURES!
import os
import numpy as np
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
from _utils_ import alloptical_plotting as aoplot
from funcsforprajay import funcs as pj
import tifffile as tf
from skimage.transform import resize
from _main_.Post4apMain import Post4ap

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/Nontargets_responses_2021-11-11'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


expobj = aoutils.import_expobj(aoresults_map_id='post a.0')  # PLACEHOLDER IMPORT OF EXPOBJ TO MAKE THE CODE WORK

# %% 1) SEIZURE WAVEFRONT PLOTTING AND ANALYSIS
"""################################# SEIZURE EVENTS PLOTTING ##############################################################
########################################################################################################################

# PLOT HEATMAP of SEIZURE EVENTS
"""

def plotHeatMapSzAllCells(expobj: Post4ap, sz_num: int):

    sz_onset, sz_offset = expobj.seizure_lfp_onsets[sz_num], expobj.seizure_lfp_offsets[sz_num]

    # -- approach of dFF normalize to the mean of the Flu data 2 seconds before the seizure
    pre_sz = 3*int(expobj.fps)
    post_sz = 4*int(expobj.fps)

    frame_start = int(sz_onset - pre_sz)
    frame_end = int(sz_offset + post_sz)

    sz_flu = expobj.raw[[expobj.cell_id.index(cell) for cell in expobj.good_cells], frame_start: frame_end]
    sz_flu_smooth = np.array([pj.smoothen_signal(signal, w=5) for signal in sz_flu])  # grouped average smoothing of the raw signal
    # x_norm = np.array([pj.dff(flu[pre_sz:], np.mean(flu[:pre_sz])) * 100 for flu in sz_flu_smooth])
    x_norm = sz_flu_smooth

    stims = [(stim - sz_onset) for stim in expobj.stim_start_frames if sz_onset <= stim < sz_offset]
    stims_off = [(stim + expobj.stim_duration_frames - 1) for stim in stims]

    # x_bf = expobj.stim_start_times[expobj.stim_start_frames.index(expobj.stims_bf_sz[sz_num])]
    # x_af = expobj.stim_start_times[expobj.stim_start_frames.index(expobj.stims_af_sz[sz_num+1])]

    paq_start = expobj.frame_clock_actual[frame_start]
    paq_end = expobj.frame_clock_actual[frame_end]

    lfp_signal = expobj.lfp_signal[paq_start: paq_end]

    # # test cropping
    # fig, axs = plt.subplots(nrows=2, figsize = (6,6))
    # axs[0].plot(expobj.meanRawFluTrace[frame_start: frame_end], c='forestgreen', zorder=1)
    # axs[1].plot(expobj.lfp_signal[paq_start: paq_end])
    # fig.show()
    # # test cropping

    # -- ordering cells based on their order of reaching top 60% signal threshold
    x_95 = [np.percentile(trace, 75) for trace in x_norm]

    x_peak = [np.min(np.where(x_norm[i] > x_95[i])) for i in range(len(x_norm))]
    new_order = np.argsort(x_peak)
    x_ordered = x_norm[new_order]

    # # plot heatmap of dFF processed Flu signals for all cells for selected sz and ordered as determined above
    # fig, ax= plt.subplots(figsize=(5,3))
    # fig, ax = aoplot.plot_traces_heatmap(expobj=expobj, arr=x_ordered, stim_on=stims, stim_off=stims_off, cmap='jet',
    #                            title=('%s - seizure %s - sz flu smooth - %s to %s' % (expobj.t_series_name, sz_num, sz_onset, sz_offset)),
    #                            xlims=None, vmin=100, vmax=500, fig=fig, ax=ax, show=False)
    # ax2 = ax.twinx()
    # x_c = np.linspace(0, x_ordered.shape[1] - 1, len(lfp_signal))
    # # ax2.plot(x_c, kwargs['lfp_signal'] * 50 + arr.shape[0] - 100, c='black')
    # ax2.plot(x_c, lfp_signal, c='black')
    # ax2.set_ylabel('LFP (mV)')
    # fig.tight_layout(pad=0.2)
    # fig.show()


    # just the bottom half cells that seems to show more of an order
    fig, ax = plt.subplots(figsize=(5, 3))
    x_ordered = x_norm[new_order[:]]
    fig, ax = aoplot.plot_traces_heatmap(expobj=expobj, arr=x_ordered, cmap='afmhot', cbar=True,
                                         title=f'{expobj.t_series_name} - seizure {sz_num} - sz flu smooth',
                                         xlims=None, vmin=100, vmax=500, fig=fig, ax=ax, show=False, x_label='Time (secs)')
    ax2 = ax.twinx()
    x_c = np.linspace(0, x_ordered.shape[1] - 1, len(lfp_signal))
    # ax2.plot(x_c, kwargs['lfp_signal'] * 50 + arr.shape[0] - 100, c='black')
    ax2.plot(x_c, lfp_signal, c='white', lw=0.35)

    x_labels = [item for item in ax.get_xticks()]
    y_labels = [-5, -4]
    ax2.set_yticks(y_labels)
    ax2.set_yticklabels([])
    # ax2.set_ylabel('LFP (mV)')
    ax2.set_ylim([-5, 7])
    fig.tight_layout(pad=0.2)
    fig.show()

    print('done plotting.')
    # # PLOT cell location with cmap based on their order of reaching top 5% signal during sz event
    # cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
    # aoplot.plot_cells_loc(expobj, cells=cell_ids_ordered, show_s2p_targets=False, color_float_list=list(range(len(cell_ids_ordered))),
    #                       title='cell locations ordered by recruitment in sz # %s' % sz_num, invert_y=True, cmap='Purples')
    #
    # # just the bottom half cells that seems to show more of an order
    # cell_ids_ordered = list(np.array(expobj.cell_id)[new_order])
    # aoplot.plot_cells_loc(expobj, cells=cell_ids_ordered[250:], show_s2p_targets=False, color_float_list=list(np.array(x_peak)[new_order][250:]),
    #                       title='cell locations ordered by recruitment in sz # %s' % sz_num, invert_y=True, cmap='Purples')






# %% 2-dc) 4D plotting of the seizure wavefront (2D space vs. time vs. Flu intensity)

def fourDplot_SzWavefront(expobj: Post4ap):

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


# %% 3) plotting of



