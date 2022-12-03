import sys

import numpy as np

from _utils_.alloptical_plotting import plot_SLMtargets_Locs, get_ax_for_multi_plot, multi_plot_subplots

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
from typing import Union, List, Dict

import _alloptical_utils as Utils
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesQuantificationSLMtargets, \
    PhotostimResponsesSLMtargetsResults
from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
import funcsforprajay.funcs as pj
import funcsforprajay.plotting as pjplot

import rep_fig_vis as rfv

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"
SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-responses-traces/"

results = PhotostimResponsesSLMtargetsResults.load()


# %%

class PhotostimImages(Quantification):
    """collecting photostim timed images"""

    save_path = SAVE_LOC + 'PhotostimImages.pkl'

    def __init__(self, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimImages MODULE to expobj: {expobj.t_series_name}')

    def makeSingleStimTrialTimedImg(self, expobj: Union[alloptical, Post4ap]):
        """make an average img (optionally save as tiff) of single photostim trials responses

        TODO: need to collect TIFF avg from suite2p registered tiff stack
        - subtract from baseline avg. this way both the photostim trials, and seizure fluorescence will be visible.

        """

        # print(expobj.PhotostimResponsesSLMTargets.adata)

        # get stim timing frames and set up frames to collect images from
        pre_stim = [(stim - round(0.5 * expobj.fps), stim) for stim in expobj.stim_start_frames]
        post_stim = [(stim + expobj.stim_duration_frames, stim + expobj.stim_duration_frames + round(0.5 * expobj.fps))
                     for stim in expobj.stim_start_frames]

        stack = pj.ImportTiff(expobj.reg_tif_crop_path)

        expobj.curr_trial_frames

        fig, axs, counter, ncols, nrows = multi_plot_subplots(num_total_plots=len(expobj.stim_start_frames), ncols=10)
        for pre, post in zip(pre_stim, post_stim):
            pre_stack_avg = np.mean(stack[pre[0]: pre[1]], axis=0)
            post_stack_avg = np.mean(stack[post[0]: post[1]], axis=0)
            # fig, ax = pjplot.plotImg(pre_stack_avg, suptitle=f'pre stim , stim fr: {pre[1]}', show=False)
            # plot_SLMtargets_Locs(expobj=expobj, background=pre_stack_avg, suptitle=f'pre stim , stim fr: {pre[1]}',
            #                      facecolors=None)

            ax, counter = get_ax_for_multi_plot(axs, counter, ncols)

            fig, ax = plot_SLMtargets_Locs(expobj=expobj, background=post_stack_avg, suptitle=f'post stim , stim fr: {pre[1]}',
                                           facecolors=None, fig=fig, ax=ax, show=False)

            print('inter')
            # pjplot.plotImg(post_stack_avg, title=f'post stim , stim fr: {pre[1]}')
        fig.show()
        print('inter 2')

        # make 8fr tiff average after stim end


# collect methods to run

@Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
def run__initPhotostimImages(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimImages = PhotostimImages(expobj=expobj)
    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0, set_cache=False)
def run__makeSingleStimTrialTimedImg(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimImages.makeSingleStimTrialTimedImg(expobj=expobj)
    expobj.save()


# %%
if __name__ == '__main__':
    # run__initPhotostimImages()
    run__makeSingleStimTrialTimedImg()
