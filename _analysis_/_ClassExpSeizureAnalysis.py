# NOTE: ALOT OF THIS CODE IS PRIMARILY COPIED AND REFACTORED OVER FROM alloptical_sz_processing IN AN EFFORT TO class-ify THE ANALYSIS OF THE SZ PROCESSING. COPIED OVER BITS ARE ARCHIVED UNDER THAT ORIGINAL SCRIPT.
import sys

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import numpy as np
import os
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
import _utils_.alloptical_plotting as aoplot
import funcsforprajay.funcs as pj
import tifffile as tf

from _analysis_._utils import Quantification
from _exp_metainfo_.exp_metainfo import AllOpticalExpsToAnalyze, ExpMetainfo
from _main_.Post4apMain import Post4ap
from _main_.TwoPhotonImagingMain import TwoPhotonImaging
from _utils_.alloptical_plotting import multi_plot_subplots, get_ax_for_multi_plot, plot_SLMtargets_Locs, plotMeanRawFluTrace, plot_lfp_stims

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

SAVE_PATH_PREFIX = '/home/pshah/mnt/qnap/Analysis/Procesing_figs/sz_processing_boundaries_non-targets_2022-04-19/'

import _utils_.io as io_



# %%
class ExpSeizureAnalysis(Quantification):
    """Processing/analysis of seizures overall for photostimulation experiments. Including analysis of seizure individual timed photostim trials."""
    
    save_path = SAVE_LOC + 'Quant__ExpSeizureAnalysis.pkl'

    def __init__(self, expobj: Post4ap, not_flip_stims=None):
        super().__init__(expobj)
        print(f'\t\- ADDING NEW ExpSeizureAnalysis MODULE to expobj: {expobj.t_series_name}')
        if not_flip_stims is None:
            self.not_flip_stims: list = expobj.not_flip_stims if hasattr(expobj, 'not_flip_stims') else []  #: this is the main attr that stores information about whether a stim needs its sz boundary classification flipped or not
        else: self.not_flip_stims: list = not_flip_stims
        self.slmtargets_szboundary_stim = expobj.slmtargets_szboundary_stim if hasattr(expobj, 'slmtargets_szboundary_stim') else {}  #: stims for slm targets
        self.nontargets_szboundary_stim = expobj.s2prois_szboundary_stim if hasattr(expobj, 'nontargets_szboundary_stim') else {}

    def __repr__(self):
        return f"ExpSeizureAnalysis <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # 0) misc functions
    # plotting of LFP trace
    @staticmethod
    def plot__photostim_timings_lfp(exp_prep, **kwargs):
        expobj = io_.import_expobj(exp_prep = exp_prep)
        aoplot.plot_lfp_stims(expobj, x_axis = 'Time (secs)', sz_markings = False, legend = False,
                              **kwargs)

    @staticmethod
    def plot__exp_sz_lfp_fov(expobj: TwoPhotonImaging = None, prep=None, trial=None):
        """plot FOV mean raw trace and lfp trace (with stim markings)"""
        if prep and trial and (expobj is None):
            expobj = io_.import_expobj(prep=prep, trial=trial)
        assert expobj, 'expobj not initialized properly.'
        fig, axs = plt.subplots(2, 1, figsize=(20, 6))
        fig, ax = plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', fig=fig, ax=axs[0],
                                      show=False)
        plot_lfp_stims(expobj=expobj, fig=fig, ax=axs[1], show=False)
        # fig, ax = plotLfpSignal(expobj=expobj, stim_span_color='', x_axis='time', fig=fig, ax=axs[1], show=False)
        fig.show()

    @staticmethod
    def run__avg_stim_images(expobj: Post4ap):
        expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=True, save_img=True)

    # 1.0) calculate time delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def FOVszInvasionTime(**kwargs):
        """
        The general approach to calculate seizure invasion time delay (for imaging FOV) is to calculate the first stim
        (which are usually every 10 secs) which has the seizure wavefront in the FOV relative to the LFP onset of the seizure
        (which is at the 4ap inj. site).

        :param kwargs: no args taken. used only to pipe in experiments from for loop.

        """

        expobj: Post4ap = kwargs['expobj']
        time_delay_sec = [-1] * len(expobj.stim_start_frames)
        sz_num = [-1] * len(expobj.stim_start_frames)
        for i in range(expobj.numSeizures):
            lfp_onset_fr = expobj.seizure_lfp_onsets[i]
            if lfp_onset_fr != 0:
                start = expobj.seizure_lfp_onsets[i]
                stop = expobj.seizure_lfp_offsets[i]
                _stim_insz = [stim_fr for stim_fr in expobj.stim_start_frames if start < stim_fr < stop]
                stims_wv = [stim_fr for stim_fr in _stim_insz if stim_fr in expobj.stimsWithSzWavefront]
                stims_nowv = [stim_fr for stim_fr in _stim_insz if stim_fr not in expobj.stimsWithSzWavefront]
                if len(stims_wv) > 0:
                    for stim in stims_wv:
                        if stim in expobj.stimsWithSzWavefront:
                            sz_start_sec = start / expobj.fps
                            _time_delay_sec = (stim / expobj.fps) - sz_start_sec
                            idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = round(_time_delay_sec, 3)
                            sz_num[idx] = i
                    for stim in stims_nowv:
                        if stim < stims_wv[0]:  # first in seizure stim frame with the seizure wavefront
                            idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = "bf invasion"  # before seizure invasion to the FOV
                            sz_num[idx] = i
                        elif stim > stims_wv[-1]:  # last in seizure stim frame with the seizure wavefront
                            idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = "af invasion"  # after seizure wavefront has passed the FOV
                            sz_num[idx] = i

        expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='delay_from_sz_onset_sec',
                                                               values=time_delay_sec)
        expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='seizure_num', values=sz_num)
        expobj.save()

    # 1.1) plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def plot_sz_invasion(**kwargs):
        expobj: Post4ap = kwargs['expobj']

        sz_nums = np.unique([i for i in list(expobj.slmtargets_data.var.seizure_num) if type(i) is int and i > 0])
        fig, axs, counter, ncols, nrows = multi_plot_subplots(num_total_plots=len(sz_nums))
        for sz in sz_nums:
            idx = np.where(expobj.slmtargets_data.var.seizure_num == sz)[0][0]  # first seizure invasion frame
            stim_frm = expobj.slmtargets_data.var.stim_start_frame[idx]
            time_del = expobj.slmtargets_data.var.delay_from_sz_onset_sec[idx]

            # plotting
            avg_stim_img_path = f'{expobj.analysis_save_path[:-1]}avg_stim_images/{expobj.metainfo["date"]}_{expobj.metainfo["trial"]}_stim-{stim_frm}.tif'
            bg_img = tf.imread(avg_stim_img_path)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, cells=in_sz, edgecolors='yellowgreen', background=bg_img)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, cells=out_sz, edgecolors='white', background=bg_img)
            ax = get_ax_for_multi_plot(axs, counter, ncols)
            fig, ax = plot_SLMtargets_Locs(expobj, fig=fig, ax=ax,
                                           title=f"sz #: {sz}, stim_fr: {stim_frm}, time inv.: {time_del}s",
                                           show=False, background=bg_img)

            try:
                inframe_coord1_x = expobj.slmtargets_data.var["seizure location"][idx][0][0]
                inframe_coord1_y = expobj.slmtargets_data.var["seizure location"][idx][0][1]
                inframe_coord2_x = expobj.slmtargets_data.var["seizure location"][idx][1][0]
                inframe_coord2_y = expobj.slmtargets_data.var["seizure location"][idx][1][1]
                ax.plot([inframe_coord1_x, inframe_coord2_x], [inframe_coord1_y, inframe_coord2_y], c='darkorange',
                        linestyle='dashed', alpha=1, lw=2)
            except TypeError:
                print('hitting nonetype error')

        fig.suptitle(f"{expobj.t_series_name} {expobj.date}")
        fig.show()

    # 2.1) procedure for classification of cells in/out of sz boundary

    # 2.1.2) classify and plot seizure boundary classification for all stims that occur during all seizures in an experiment
    def classify_sz_boundaries_all_stims(self, expobj: Post4ap, cells: str = 'slm targets'):
        """sets up and runs the function for classifying cells across the sz boundary for each photostim frame.
        also makes plots for each stim as well.

        CLASSIFIES BOTH SLM TARGETS, AND S2P ROIS NONTARGETS.
        """

        print(' \nworking on classifying cells for stims start frames...')
        if cells == 'slm targets':
            self.slmtargets_szboundary_stim = {}
        elif cells == 'nontargets':
            self.nontargets_szboundary_stim = {}
        else:
            raise ValueError('cells arg needs to be either `slm targets` or `nontargets`')

        sz_stim_frames = []
        for i in range(expobj.numSeizures):
            sz_stim_frames.append(list(expobj.stimsSzLocations[expobj.stimsSzLocations['sz_num'] == i].index))

        assert expobj.stims_in_sz[1:] == pj.flattenOnce(sz_stim_frames), 'not all ictal stims accounted for in list of stim frames to classify.'

        for sz_num, stims_of_interest in enumerate(sz_stim_frames):
            print('\n|-', stims_of_interest)

            nrows = len(stims_of_interest) // 4 + 1
            if nrows == 1:
                nrows += 1
            ncols = 4
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
            counter = 0

            for stim in stims_of_interest:
                ax = axs[counter // ncols, counter % ncols]

                # sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
                if not os.path.exists(expobj.sz_border_path(stim=stim)):
                    print(expobj.sz_border_path(stim=stim))
                if stim not in self.not_flip_stims:  # note that self.not_flip_stims is set manually after assessing the classification of each stim. contains a list of stims where the default classification was incorrect.
                    flip = False
                else:
                    flip = True

                try:
                    if cells == 'slm targets':
                        # CLASSIFY SLM TARGETS ACROSS SZ BOUNDARIES
                        in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(stim=stim, to_plot=True, title=str(stim),
                                                                                     text=str(stim), flip=flip, fig=fig, ax=ax)
                        self.slmtargets_szboundary_stim[stim] = in_sz  # for each stim, there will be a ls of cells that will be classified as in seizure or out of seizure
                    elif cells == 'nontargets':
                        ## CLASSIFY NONTARGETS ACROSS SZ BOUNDARIES
                        in_sz, out_sz, fig, ax = expobj.classify_cells_sz_bound(stim=stim, to_plot=True, title=str(stim),
                                                                                text=str(stim), flip=flip, fig=fig, ax=ax)
                        self.nontargets_szboundary_stim[stim] = in_sz
                    # axs[counter // ncols, counter % ncols] = ax
                except KeyError:
                    print(f"WARNING: stim frame # {stim} missing from boundary csv and/or stim sz locations")

                counter += 1

            fig.suptitle(f'{expobj.t_series_name} - Avg img around stims during sz - seizure # {sz_num + 1}', y=0.995)

            save_path_full = f"{SAVE_PATH_PREFIX}/{expobj.t_series_name} seizure # {sz_num + 1} - {len(stims_of_interest)} stims - {cells}.png"
            # print(f"saving fig to: {save_path_full}")
            Utils.save_figure(fig=fig, save_path_full=save_path_full)
            fig.show()

    def _procedure__classifying_sz_boundary(self, expobj: Post4ap, cells='nontargets'):
        """
        Full procedure for classifying targets (and eventually non-targets) as in or out of sz boundary for each stim.

        Procedure: Runs plotting of sz boundaries for all stims in sz, then asks for stims to correct classification as input,
        then runs plotting of sz boundaries again.

        Sz boundary is based on manual placement of two coordinates on the
        avg image of each stim frame. the purpose of this function is to review the classification for each stim during each seizure
        to see if the cells are classified correctly on either side of the boundary.

        :return:
        """

        # aoplot.plot_lfp_stims(expobj)
        self.plot__exp_sz_lfp_fov(expobj=expobj)
        # matlab_pairedmeasurements_path = '%s/paired_measurements/%s_%s_%s.mat' % (expobj.analysis_save_path[:-23], expobj.metainfo['date'], expobj.metainfo['animal prep.'], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
        # expobj.paqProcessing()
        # expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path)
        # expobj.save()

        # ######## CLASSIFY SLM PHOTOSTIM TARGETS AS IN OR OUT OF current SZ location in the FOV
        # -- FIRST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv

        # expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=True, save_img=True)
        expobj.sz_locations_stims() #if not hasattr(expobj, 'stimsSzLocations') else None

        ######## - all stims in sz are classified, with individual sz events labelled
        # expobj.ExpSeizure.classify_sz_boundaries_all_stims(expobj=expobj, cells='slm targets')

        # similar procedure for nontargets
        self.classify_sz_boundaries_all_stims(expobj=expobj, cells=cells)

        expobj.save()


    # flip of stim boundaries manually
    @staticmethod
    def enter_stims_to_flip(expobj: Post4ap):
        """when placing sz wavefront boundary, there are stim instances when the classification code places the wrong side of the
        image as inside the sz boundary. this code asks for which stims are being placed incorrectly and collects these into a list
        that is further accessed to assign the sz boundary.

        Note; remember that the sz boundary is placed manually as two points.
        """

        # 180 330 481 631 782 932 1083 1233 1384 1835 1986 2136 2287 2438 2588 3040 3190 3491  # temp stims to flip for PS04 t-018

        input_string = input("Enter a string of stims to flip (ensure to separate each stim frame # by exactly one space: ")
        not_flip_stims = input_string.split()

        for stim in not_flip_stims:
            assert int(stim) in expobj.stim_start_frames, f'stim {stim} not found as a stim start frame for {expobj}'

        expobj.ExpSeizure.not_flip_stims.extend([int(x) for x in not_flip_stims])

        print(f"\n stims in .not_flip_stims list: {expobj.ExpSeizure.not_flip_stims}")
        # expobj.not_flip_stims = expobj.stims_in_sz[
        #                         1:]  # specify here the stims where the flip=False leads to incorrect assignment
        expobj.save()

    # flip of stim boundaries manually
    @staticmethod
    def remove_stims_to_flip(expobj: Post4ap, stims=None):
        """
        To remove incorrectly given stims
        when placing sz wavefront boundary, there are stim instances when the classification code places the wrong side of the
        image as inside the sz boundary. this code asks for which stims are being placed incorrectly and collects these into a list
        that is further accessed to assign the sz boundary.

        Note; remember that the sz boundary is placed manually as two points.
        """

        if stims is None:
            input_string = input(f"Enter list of stims to remove from expobj.ExpSeizure.not_flip_stims for {expobj.t_series_name} (ensure to separate each stim frame # by exactly one space: ")
            remove_stims = input_string.split()
        else:
            remove_stims = stims

        expobj.ExpSeizure.not_flip_stims = [stim for stim in expobj.ExpSeizure.not_flip_stims if stim not in remove_stims]


        # for x in remove_stims:
        #     if x in expobj.ExpSeizure.not_flip_stims:
        #         expobj.ExpSeizure.not_flip_stims.remove(int(x))
        #     else:
        #         print(f"{x} is not in .not_flip_stims")
        # # [expobj.ExpSeizure.not_flip_stims.remove(int(x)) for x in remove_stims]

        print(f"\n stims in .not_flip_stims list: {expobj.ExpSeizure.not_flip_stims}")
        # expobj.not_flip_stims = expobj.stims_in_sz[
        #                         1:]  # specify here the stims where the flip=False leads to incorrect assignment
        expobj.save()

    # @staticmethod
    # @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, allow_rerun=True,
    #                                 run_trials=['PS04 t-018', 'RL109 t-017'])
    # def classifying_sz_boundary_archive(**kwargs):
    #     """
    #     classifies targets (and eventually non-targets) as in or out of sz boundary for each stim.
    #
    #     Sz boundary is based on manual placement of two coordinates on the
    #     avg image of each stim frame. the purpose of this function is to review the classification for each stim during each seizure
    #     to see if the cells are classified correctly on either side of the boundary.
    #
    #     :param kwargs:
    #     :return:
    #     """
    #
    #     expobj = kwargs['expobj']
    #
    #     # aoplot.plot_lfp_stims(expobj)
    #
    #     # matlab_pairedmeasurements_path = '%s/paired_measurements/%s_%s_%s.mat' % (expobj.analysis_save_path[:-23], expobj.metainfo['date'], expobj.metainfo['animal prep.'], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
    #     # expobj.paqProcessing()
    #     # expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path)
    #     # expobj.save()
    #
    #     # aoplot.plotSLMtargetsLocs(expobj, background=None)
    #
    #     # ######## CLASSIFY SLM PHOTOSTIM TARGETS AS IN OR OUT OF current SZ location in the FOV
    #     # -- FIRST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv
    #
    #     if not hasattr(expobj, 'sz_boundary_csv_done'):
    #         expobj.sz_boundary_csv_done = True
    #     else:
    #         AssertionError('confirm that sz boundary csv creation has been completed')
    #         # sys.exit()
    #
    #     expobj.sz_locations_stims() if not hasattr(expobj, 'stimsSzLocations') else None
    #
    #     # specify stims for classifying cells
    #     on_ = []
    #     if 0 in expobj.seizure_lfp_onsets:  # this is used to check if 2p imaging is starting mid-seizure (which should be signified by the first lfp onset being set at frame # 0)
    #         on_ = on_ + [expobj.stim_start_frames[0]]
    #     on_.extend(expobj.stims_bf_sz)
    #     if len(expobj.stims_af_sz) != len(on_):
    #         end = expobj.stims_af_sz + [expobj.stim_start_frames[-1]]
    #     else:
    #         end = expobj.stims_af_sz
    #     print('\-seizure start frames: ', on_)
    #     print('\-seizure end frames: ', end)
    #
    #     ##### import the CSV file in and classify cells by their location in or out of seizure
    #
    #     if not hasattr(expobj, 'not_flip_stims'):
    #         print(
    #             f"|-- {expobj.t_series_name} DOES NOT have previous not_flip_stims attr, so making a new empty list attr")
    #         expobj.not_flip_stims = []  # specify here the stims where the flip=False leads to incorrect assignment
    #     else:
    #         print(f"\-expobj.not_flip_stims: {expobj.not_flip_stims}")
    #
    #     # break
    #
    #     print(' \nworking on classifying cells for stims start frames...')
    #     # TODO need to implement rest of code for nontargets_szboundary_stim for nontargets
    #     expobj.slmtargets_szboundary_stim = {}
    #     expobj.nontargets_szboundary_stim = {}
    #
    #     ######## - all stims in sz are classified, with individual sz events labelled
    #
    #     stims_of_interest = expobj.stimsWithSzWavefront
    #     print(' \-all stims in seizures: \n \-', stims_of_interest)
    #     nrows = len(stims_of_interest) // 4 + 1
    #     if nrows == 1:
    #         nrows += 1
    #     ncols = 4
    #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    #     counter = 0
    #
    #     for stim in expobj.stimsWithSzWavefront:
    #         sz_num = expobj.stimsSzLocations.loc[stim, 'sz_num']
    #
    #         print(f"considering stim # {stim}")
    #
    #         ax = axs[counter // ncols, counter % ncols]
    #
    #         if os.path.exists(expobj.sz_border_path(stim=stim)):
    #             # first round of classifying (dont flip any cells over) - do this in the second round
    #             if stim not in expobj.not_flip_stims:
    #                 flip = False
    #             else:
    #                 flip = True
    #
    #             # # classification of suite2p ROIs relative to sz boundary
    #             # in_sz, out_sz, fig, ax = expobj.classify_cells_sz_bound(stim=stim, to_plot=True,
    #             #                                                         flip=flip, fig=fig, ax=ax, text='sz %s stim %s' % (sz_num, stim))
    #             # expobj.nontargets_szboundary_stim[stim] = in_sz
    #
    #             # # classification of SLM targets relative to sz boundary
    #             in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(stim=stim, to_plot=True,
    #                                                                          title=stim, flip=flip, fig=fig, ax=ax)
    #             expobj.slmtargets_szboundary_stim[
    #                 stim] = in_sz  # for each stim, there will be a ls of cells that will be classified as in seizure or out of seizure
    #
    #             axs[counter // ncols, counter % ncols] = ax
    #             counter += 1
    #         else:
    #             print(f"sz border path doesn't exist for stim {stim}: {expobj.sz_border_path(stim=stim)}")
    #
    #     fig.suptitle('%s %s - Avg img around stims during- all stims' % (
    #         expobj.metainfo['animal prep.'], expobj.metainfo['trial']), y=0.995)
    #     save_path_full = f"{SAVE_PATH_PREFIX}/{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} {len(expobj.stimsWithSzWavefront)} events.png"
    #     print(f"saving fig to: {save_path_full}")
    #     fig.savefig(save_path_full)
    #
    #     expobj.save()
    #     print('end end end.')



    # 4.1) counting seizure incidence across all imaging trials
    @staticmethod
    def count_sz_incidence_2p_trials():
        for key in list([*AllOpticalExpsToAnalyze.trial_maps['post']]):
            # import initial expobj
            expobj = io_.import_expobj(aoresults_map_id=f'pre {key}.0', verbose=False)
            prep = expobj.metainfo['animal prep.']
            # look at all run_post4ap_trials trials in expobj and for loop over all of those run_post4ap_trials trials
            for trial in expobj.metainfo['post4ap_trials']:
                # import expobj
                expobj = io_.import_expobj(prep=prep, trial=trial, verbose=False)
                total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

                # count seizure incidence (avg. over mins) for each experiment (animal)
                if hasattr(expobj, 'seizure_lfp_onsets'):
                    n_seizures = len(expobj.seizure_lfp_onsets)
                else:
                    n_seizures = 0

                print(f'Seizure incidence for {prep}, {trial}, {expobj.metainfo["exptype"]}: ',
                      np.round(n_seizures / total_time_recording, 2))

    # 4.1.1) measure seizure incidence across onePstim trials
    @staticmethod
    def count_sz_incidence_1p_trials():
        for exp_prep in ExpMetainfo.onephotonstim.post_4ap_trials:
            expobj = io_.import_expobj(exp_prep=exp_prep, verbose=False)
            total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

            # count seizure incidence (avg. over mins) for each experiment (animal)
            if hasattr(expobj, 'seizure_lfp_onsets'):
                n_seizures = len(expobj.seizure_lfp_onsets)
            else:
                n_seizures = 0

            print('Seizure incidence for %s, %s, %s: ' % (
                expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
                  np.round(n_seizures / total_time_recording, 2))

    # 4.1.2) plot seizure incidence across onePstim and twoPstim trials
    twop_trials_sz_incidence = [0.35, 0.251666667, 0.91, 0.33, 0.553333333, 0.0875, 0.47, 0.33, 0.52]  # sz/min
    onep_trials_sz_incidence = [0.38, 0.26, 0.19, 0.436666667, 0.685]  # sz/min

    @classmethod
    def plot__sz_incidence(cls):
        pj.plot_bar_with_points(data=[cls.twop_trials_sz_incidence, cls.onep_trials_sz_incidence],
                                x_tick_labels=['2p stim', '1p stim'],
                                colors=['purple', 'green'], y_label='sz incidence (events/min)',
                                title='rate of seizures during exp', expand_size_x=0.4, expand_size_y=1, ylims=[0, 1],
                                shrink_text=0.8)

        pj.plot_bar_with_points(data=[cls.twop_trials_sz_incidence + cls.onep_trials_sz_incidence],
                                x_tick_labels=['Experiments'],
                                colors=['#2E3074'], y_label='Seizure incidence (events/min)', alpha=0.7, bar=False,
                                title='rate of seizures during exp', expand_size_x=0.7, expand_size_y=1, ylims=[0, 1],
                                shrink_text=0.8)

    # 4.2) measure seizure LENGTHS across all imaging trials (including any spont imaging you might have)

    @staticmethod
    def count_sz_lengths_2p_trials():
        for key in list([*AllOpticalExpsToAnalyze.trial_maps['post']]):
            # import initial expobj
            expobj = io_.import_expobj(aoresults_map_id=f'pre {key}.0', verbose=False)
            prep = expobj.metainfo['animal prep.']
            # look at all run_post4ap_trials trials in expobj
            # if 'post-4ap trials' in expobj.metainfo.keys():
            #     a = 'post-4ap trials'
            # elif 'post4ap_trials' in expobj.metainfo.keys():
            #     a = 'post4ap_trials'
            # for loop over all of those run_post4ap_trials trials
            for trial in expobj.metainfo['post4ap_trials']:
                # import expobj
                expobj = io_.import_expobj(prep=prep, trial=trial, verbose=False)
                # count the average length of each seizure
                if hasattr(expobj, 'seizure_lfp_onsets'):
                    n_seizures = len(expobj.seizure_lfp_onsets)
                    counter = 0
                    sz_lengths_total = 0
                    if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                        for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                            if sz_onset != 0:
                                sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] -
                                                     expobj.frame_clock_actual[sz_onset]) / expobj.paq_rate
                                counter += 1
                        avg_len = sz_lengths_total / counter
                        expobj.avg_sz_len = avg_len

                        print('Avg. seizure length (secs) for %s, %s, %s: ' % (prep, trial, expobj.metainfo['exptype']),
                              np.round(expobj.avg_sz_len, 2))

                else:
                    n_seizures = 0
                    print('no sz events for %s, %s, %s ' % (prep, trial, expobj.metainfo['exptype']))

    # 4.2.1) measure seizure LENGTHS across onePstim trials
    @staticmethod
    def count_sz_lengths_1p_trials():
        for exp_prep in ExpMetainfo.onephotonstim.post_4ap_trials:
            expobj = io_.import_expobj(exp_prep=exp_prep, verbose=False)
            # count the average length of each seizure
            if hasattr(expobj, 'seizure_lfp_onsets'):
                n_seizures = len(expobj.seizure_lfp_onsets)
                counter = 0
                sz_lengths_total = 0
                if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                    for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                        if sz_onset != 0:
                            sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] -
                                                 expobj.frame_clock_actual[sz_onset]) / expobj.paq_rate
                            counter += 1
                    avg_len = sz_lengths_total / counter
                    expobj.avg_sz_len = avg_len
                    print('Avg. seizure length (secs) for %s, %s, %s: ' % (
                        expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
                          np.round(expobj.avg_sz_len, 2))

            else:
                n_seizures = 0
                print('Avg. seizure length (secs) for %s, %s, %s ' % (
                    expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']))

    # 4.2.2) plot seizure length across onePstim and twoPstim trials
    twop_trials_sz_lengths = [24.0, 93.73, 38.86, 84.77, 17.16, 83.78, 15.78, 36.88]
    onep_trials_sz_lengths = [30.02, 34.25, 114.53, 35.57]

    @classmethod
    def plot__sz_lengths(cls):
        pj.plot_bar_with_points(data=[cls.twop_trials_sz_lengths, cls.onep_trials_sz_lengths],
                                x_tick_labels=['2p stim', '1p stim'],
                                colors=['purple', 'green'], y_label='seizure length (secs)',
                                title='Avg. length of sz', expand_size_x=0.4, expand_size_y=1, ylims=[0, 120],
                                title_pad=15,
                                shrink_text=0.8)

        pj.plot_bar_with_points(data=[cls.twop_trials_sz_lengths + cls.onep_trials_sz_lengths],
                                x_tick_labels=['Experiments'],
                                colors=['green'], y_label='Seizure length (secs)', alpha=0.7, bar=False,
                                title='Avg sz length', expand_size_x=0.7, expand_size_y=1, ylims=[0, 120],
                                shrink_text=0.8)

    # 6.0-dc) ANALYSIS: calculate time delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment
    # -- this section has been moved to _ClassExpSeizureAnalysis .22/02/20 -- this copy here is now archived

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def calc__szInvasionTime(**kwargs):
        """
        The general approach to calculate seizure invasion time delay is to calculate the first stim (which are usually every 10 secs)
        which has the seizure wavefront in the FOV relative to the LFP onset of the seizure (which is at the 4ap inj site).


        :param kwargs: no args taken. used only to pipe in experiments from for loop.
        """

        expobj = kwargs['expobj']
        time_delay_sec = [-1] * len(expobj.stim_start_frames)
        sz_num = [-1] * len(expobj.stim_start_frames)
        for i in range(expobj.numSeizures):
            # i = 1
            lfp_onset_fr = expobj.seizure_lfp_onsets[i]
            if lfp_onset_fr != 0:
                start = expobj.seizure_lfp_onsets[i]
                stop = expobj.seizure_lfp_offsets[i]
                _stim_insz = [stim_fr for stim_fr in expobj.stim_start_frames if start < stim_fr < stop]
                stims_wv = [stim_fr for stim_fr in _stim_insz if stim_fr in expobj.stimsWithSzWavefront]
                stims_nowv = [stim_fr for stim_fr in _stim_insz if stim_fr not in expobj.stimsWithSzWavefront]
                if len(stims_wv) > 0:
                    for stim in stims_wv:
                        if stim in expobj.stimsWithSzWavefront:
                            sz_start_sec = start / expobj.fps
                            _time_delay_sec = (stim / expobj.fps) - sz_start_sec
                            idx = np.where(expobj.slmtargets_data.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = round(_time_delay_sec, 3)
                            sz_num[idx] = i
                    for stim in stims_nowv:
                        if stim < stims_wv[0]:  # first in seizure stim frame with the seizure wavefront
                            idx = np.where(expobj.slmtargets_data.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = "bf invasion"  # before seizure invasion to the FOV
                            sz_num[idx] = i
                        elif stim > stims_wv[-1]:  # last in seizure stim frame with the seizure wavefront
                            idx = np.where(expobj.slmtargets_data.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = "af invasion"  # after seizure wavefront has passed the FOV
                            sz_num[idx] = i

        expobj.slmtargets_data.add_variable(var_name='delay_from_sz_onset_sec', values=time_delay_sec)
        expobj.slmtargets_data.add_variable(var_name='seizure_num', values=sz_num)
        expobj.save()

    # 6.1) plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def plot__sz_invasion(**kwargs):
        expobj: Post4ap = kwargs['expobj']

        sz_nums = np.unique([i for i in list(expobj.slmtargets_data.var.seizure_num) if type(i) is int and i > 0])
        fig, axs, counter, ncols, nrows = multi_plot_subplots(num_total_plots=len(sz_nums))
        for sz in sz_nums:
            idx = np.where(expobj.slmtargets_data.var.seizure_num == sz)[0][0]  # first seizure invasion frame
            stim_frm = expobj.slmtargets_data.var.stim_start_frame[idx]
            time_del = expobj.slmtargets_data.var.delay_from_sz_onset_sec[idx]

            # plotting
            avg_stim_img_path = f'{expobj.analysis_save_path[:-1]}avg_stim_images/{expobj.metainfo["date"]}_{expobj.metainfo["trial"]}_stim-{stim_frm}.tif'
            bg_img = tf.imread(avg_stim_img_path)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, cells=in_sz, edgecolors='yellowgreen', background=bg_img)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, cells=out_sz, edgecolors='white', background=bg_img)
            ax = get_ax_for_multi_plot(axs, counter, ncols)
            fig, ax = plot_SLMtargets_Locs(expobj, fig=fig, ax=ax,
                                           title=f"sz #: {sz}, stim_fr: {stim_frm}, time inv.: {time_del}s",
                                           show=False, background=bg_img)

            try:
                inframe_coord1_x = expobj.slmtargets_data.var["seizure location"][idx][0][0]
                inframe_coord1_y = expobj.slmtargets_data.var["seizure location"][idx][0][1]
                inframe_coord2_x = expobj.slmtargets_data.var["seizure location"][idx][1][0]
                inframe_coord2_y = expobj.slmtargets_data.var["seizure location"][idx][1][1]
                ax.plot([inframe_coord1_x, inframe_coord2_x], [inframe_coord1_y, inframe_coord2_y], c='darkorange',
                        linestyle='dashed', alpha=1, lw=2)
            except TypeError:
                print('hitting nonetype error')

        fig.suptitle(f"{expobj.t_series_name} {expobj.date}")
        fig.show()

    # 7.0-dc) ANALYSIS: cross-correlation between mean FOV 2p calcium trace and LFP seizure trace - incomplete not working yet
    @staticmethod
    def cross_corr_2pFOV_LFP():
        """attempting to calculate a cross-correlogram between the mean FOV 2p calcium imaging trace and the LFP trace.
        thought was that the large signals in the overall mean FOV of the 2p imaging would be closely correlated to the LFP trace.

        progress: isn't working so well so far. might need to window-smooth the LFP trace down to the same sampling rate as the FOV 2p (or even further down since Ca2+ signal is not that fast).
        """
        import scipy.signal as signal

        expobj = io_.import_expobj(prep='RL109', trial='t-017')

        sznum = 1
        slice = np.s_[
                expobj.convert_frames_to_paqclock(expobj.seizure_lfp_onsets[sznum]): expobj.convert_frames_to_paqclock(
                    expobj.seizure_lfp_offsets[sznum])]

        # detrend
        detrended_lfp = signal.detrend(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual])[
                            slice] * -1

        # downsample LFP signal to the same # of datapoints as # of frames in 2p calcium trace
        CaUpsampled1 = signal.resample(expobj.meanRawFluTrace, len(detrended_lfp))[slice]

        pj.make_general_plot([CaUpsampled1], figsize=[20, 3])
        pj.make_general_plot([detrended_lfp], figsize=[20, 3])

        # use numpy or scipy.signal .correlate to correlate the two timeseries
        correlated = signal.correlate(CaUpsampled1, detrended_lfp)
        lags = signal.correlation_lags(len(CaUpsampled1), len(detrended_lfp))
        correlated /= np.max(correlated)

        f, axs = plt.subplots(nrows=3, ncols=1, figsize=[20, 9])
        axs[0].plot(CaUpsampled1)
        axs[1].plot(detrended_lfp)
        # axs[2].plot(correlated)
        axs[2].plot(lags, correlated)
        f.show()

# %%
if __name__ == '__main__':
    # DATA INSPECTION

    # ExpSeizureAnalysis.plot__exp_sz_lfp_fov(prep='RL109', trial='t-017')

    expobj: Post4ap = io_.import_expobj(exp_prep='RL109 t-018')

    # print(expobj.ExpSeizure.not_flip_stims)
    ExpSeizureAnalysis.remove_stims_to_flip(expobj=expobj, stims=[9230, 9379, 9528, 9677, 9826, 9974, 10123, 10272, 10471, 10570])
    # RL109 t-018 not flip stims entry: 2386 2534 2683 7891 8040 8189 8338 8486 8635 8784 8933 9082 9230 9379 9528 9677 9826 9974 10123 10272 10421 10570


    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, allow_rerun=True,
                                    run_trials=['RL109 t-018', 'PS07 t-011', 'PS06 t-013', 'PS11 t-011'])
    def __fix__not_flip_stims_expseizure_class(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        expobj.ExpSeizure.not_flip_stims = expobj.not_flip_stims
        print(expobj.ExpSeizure.not_flip_stims)
        expobj.save()


    # __fix__not_flip_stims_expseizure_class()