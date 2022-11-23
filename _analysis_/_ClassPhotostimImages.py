import sys

import numpy as np

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
        "make an average img (optionally save as tiff) of single photostim trials responses"


        # get stim timing frames
        print(expobj.PhotostimResponsesSLMTargets.adata)
        pre_stim = [(stim - round(0.5*expobj.fps), stim) for stim in expobj.stim_start_frames]
        post_stim = [(stim + expobj.stim_duration_frames, stim + expobj.stim_duration_frames + round(0.5*expobj.fps)) for stim in expobj.stim_start_frames]

        stack = pj.ImportTiff(expobj.tiff_path)
        for pre, post in zip(pre_stim, post_stim):
            pre_stack_avg = np.mean(stack[pre[0]: pre[1]], axis=0)
            post_stack_avg = np.mean(stack[post[0]: post[1]], axis=0)
            pjplot.plotImg(pre_stack_avg, title=f'pre stim , stim fr: {pre[1]}')
            pjplot.plotImg(post_stack_avg, title=f'post stim , stim fr: {pre[1]}')



        # make 8fr tiff average after stim end


# collect methods to run

@Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
def run__initPhotostimImages(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimImages = PhotostimImages(expobj=expobj)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0, set_cache=False)
def run__makeSingleStimTrialTimedImg(**kwargs):
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']
    expobj.PhotostimImages.makeSingleStimTrialTimedImg(expobj=expobj)
    expobj.save()


# %%
if __name__ == '__main__':
    run__initPhotostimImages()
    run__makeSingleStimTrialTimedImg()















