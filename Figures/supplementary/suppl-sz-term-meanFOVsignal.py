"""
quick caption:

A) representative example of post-seizure termination mean fluorescence level goes below interictal periods
B) quantification of mean FOV fluorescence in 30 secs post-seizure termination


"""


# %%
import sys

from funcsforprajay.funcs import flattenOnce
from funcsforprajay.plotting.plotting import plot_bar_with_points
from scipy.stats import sem

from _alloptical_utils import run_for_loop_across_exps
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis, ExpSeizureResults
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import plot_settings
from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure

from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs, OnePhotonStimResults

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rep_fig_vis as rfv

plot_settings()

Results: ExpSeizureResults = ExpSeizureResults.load()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

save_fig = True


''
# %% MAKE FIGURE LAYOUT
rfv.set_fontsize(8)

layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.15, 0.75, 0.45, 0.90)},
    'B': {'panel_shape': (1, 1),
          'bound': (0.6, 0.75, 0.67, 0.90)}
}

dpi = 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)


rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0.01, x_adjust=0.1)
rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0.01, x_adjust=0.101)


# rfv.show_test_figure_layout(fig, axes=axes, show=True)  # test what layout looks like quickly, but can also skip and moveon to plotting data.



# %% B - quantification of post-sz norm mean Flu of each seizure


mean_post_seizure_termination_all = Results.meanFOV_post_sz_term

plot_bar_with_points([mean_post_seizure_termination_all], x_tick_labels=[''], bar=False, ax=axes['B'][0], show=False, y_label='Mean FOV Flu \n (norm. to interictal)',
                     fig=fig, ylims=[60, 110], s=60, colors=['cornflowerblue'], alpha=1, fontsize=10)




# # %% archive version - delete once results have been collected for this
#
# exclude_sz = {
#     'RL109 t-018': [2],
#     'PS06 t-013': [4]
# }  #: seizures to exclude from analysis (for any of the exclusion criteria)
#
#
# @run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=1, allow_rerun=True,
#                                 skip_trials=['PS04 t-018'])
# def __function(**kwargs):
#     expobj: Post4ap = kwargs[
#         'expobj']  # ; plt.plot(expobj.meanRawFluTrace, lw=0.5); plt.suptitle(expobj.t_series_name); plt.show()
#
#     mean_interictal = np.mean(
#         [expobj.meanRawFluTrace[x] for x in range(expobj.n_frames) if x not in expobj.seizure_frames])
#
#     mean_post_seizure_termination = []
#     num_frames = []
#     for onset, offset in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
#         if expobj.t_series_name in exclude_sz.keys() and expobj.seizure_lfp_onsets.index(onset) in exclude_sz[expobj.t_series_name]:
#             pass
#         else:
#             frames = np.arange(int(offset + expobj.fps * 5), int(offset + expobj.fps * 30))
#             frames = [frame for frame in frames if frame not in expobj.photostim_frames]
#             try:
#                 if onset > 0 and offset + int(expobj.fps * 30) < expobj.n_frames:
#                     pre_onset_frames = np.arange(int(onset - expobj.fps * 10), int(onset))
#                     pre_onset_frames = [frame for frame in pre_onset_frames if frame not in expobj.photostim_frames]
#                     # pre_onset = expobj.meanRawFluTrace[onset - int(expobj.fps * 30): onset]
#                     pre_onset = expobj.meanRawFluTrace[pre_onset_frames]
#                     pre_mean = np.mean(pre_onset)
#                     trace = expobj.meanRawFluTrace[frames]
#                     # trace = expobj.meanRawFluTrace[int(offset - expobj.fps * 10): int(offset + expobj.fps * 30)]
#                     trace_norm = (trace / mean_interictal) * 100
#                     # trace_norm = (trace / pre_mean) * 100
#                     mean_post_seizure_termination.append(np.round(np.mean(trace_norm), 3))
#                     num_frames.append(len(frames))
#             except:
#                 pass
#     return np.mean(mean_post_seizure_termination)
#
# mean_post_seizure_termination_all = __function()




# %% A - example of all seizures from RL108 t-013

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')

# f, ax = plt.subplots(figsize=(5, 3))
ax = axes['A'][0]
mean_interictal = np.mean([expobj.meanRawFluTrace[x] for x in range(expobj.n_frames) if x not in expobj.seizure_frames])

traces = []
num_frames = []
for onset, offset in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
    frames = np.arange(int(offset - expobj.fps * 10), int(offset + expobj.fps * 40))
    frames = [frame for frame in frames if frame not in expobj.photostim_frames]
    try:
        if onset > 0 and offset + int(expobj.fps * 30) < expobj.n_frames:
            pre_onset_frames = np.arange(int(onset - expobj.fps * 10), int(onset))
            pre_onset_frames = [frame for frame in pre_onset_frames if frame not in expobj.photostim_frames]
            # pre_onset = expobj.meanRawFluTrace[onset - int(expobj.fps * 30): onset]
            pre_onset = expobj.meanRawFluTrace[pre_onset_frames]
            pre_mean = np.mean(pre_onset)
            trace = expobj.meanRawFluTrace[frames]
            # trace = expobj.meanRawFluTrace[int(offset - expobj.fps * 10): int(offset + expobj.fps * 30)]
            trace_norm = (trace / mean_interictal) * 100
            # trace_norm = (trace / pre_mean) * 100
            # ax.plot(trace_norm, lw=0.3)
            traces.append(trace_norm)
            num_frames.append(len(frames))
    except:
        pass
x_range = np.linspace(0, 30, min(num_frames))
mean_ = np.mean([trace[:min(num_frames)] for trace in traces], axis=0)
sem_ = sem([trace[:min(num_frames)] for trace in traces])
ax.plot(x_range, mean_, c='black', lw=0.5)
ax.fill_between(x_range, mean_ - sem_, mean_ + sem_, alpha=0.3)
ax.axhline(y=100)
ax.set_ylim([60, 160])
ax.set_ylabel('Mean FOV Flu \n (norm. to interictal)')
ax.set_xlabel('Time (secs)')
# f.show()


# %%
if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/suppl-sz-term-meanFOVsignal-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/suppl-sz-term-meanFOVsignal-RF.svg")


fig.show()


