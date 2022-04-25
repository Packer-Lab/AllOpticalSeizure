import glob

import os
import sys
from typing import Union

from _main_.AllOpticalMain import alloptical
from _utils_.io import import_expobj

sys.path.append('/home/pshah/Documents/code/')
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import tifffile as tf
import bisect
from funcsforprajay import funcs as pj
from funcsforprajay import pnt2line
from _utils_.paq_utils import paq_read, frames_discard
# from _utils_ import alloptical_plotting as aoplot
import _alloptical_utils as Utils


class Post4ap(alloptical):

    def __init__(self, paths, metainfo, stimtype, discard_all):

        from _analysis_._ClassNonTargetsSzInvasionSpatial import NonTargetsSzInvasionSpatial
        self.NonTargetsSzInvasionSpatial: NonTargetsSzInvasionSpatial = None
        from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview
        self.TargetsSzInvasionSpatial_codereview: TargetsSzInvasionSpatial_codereview = None  # SLM targets spatial distance to sz wavefront vs. photostim responses analysis object
        from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal
        self.TargetsSzInvasionTemporal = TargetsSzInvasionTemporal(expobj=self)
        from _analysis_._ClassExpSeizureAnalysis import ExpSeizureAnalysis
        self.ExpSeizure = ExpSeizureAnalysis(expobj=self)
        self.not_flip_stims = []  # specify here the stims where the flip=False leads to incorrect assignment, just here as a placeholder though until i fully transfer this attr down to the ExpSeizure submodule.

        alloptical.__init__(self, paths, metainfo, stimtype)
        self.time_del_szinv_stims: pd.DataFrame = pd.DataFrame()  # df containing delay to sz invasion for each target for each stim frame (dim: n_targets x n_stims)
        # self.mean_targets_szinvasion_trace: dict = {}  # dictionary containing mean Raw Flu trace around sz invasion time of each target (as well as other info as keyed into dict)  # refactored this to a class instance attr under TargetsSzInvasionTemporal
        print('\ninitialized Post4ap expobj of exptype and trial: %s, %s, %s' % (self.metainfo['exptype'],
                                                                                 self.metainfo['trial'],
                                                                                 self.metainfo['date']))

        #### initializing data processing, data analysis and/or results associated attr's

        ## SEIZURES RELATED ATTRIBUTES
        self.seizure_frames = None  #: frame #s inside seizure
        self.seizure_lfp_onsets = None  #: frame #s corresponding to ONSET of seizure as manually inspected from the LFP signal
        self.seizure_lfp_offsets = None  #: frame #s corresponding to OFFSET of seizure as manually inspected from the LFP signal

        ##
        self.slmtargets_szboundary_stim = None  # dictionary of cells classification either inside or outside of boundary - SLM targets
        self.nontargets_szboundary_stim = None  # dictionary of cells classification either inside or outside of boundary - suite2p nontargets

        ## PHOTOSTIM SLM TARGETS

        self.SLMTargets_stims_dff_outsz = None
        self.SLMTargets_stims_dffAvg_outsz = None
        self.SLMTargets_stims_dfstdF_outsz = None
        self.SLMTargets_stims_dfstdF_avg_outsz = None
        self.SLMTargets_stims_raw_outsz = None
        self.SLMTargets_stims_rawAvg_outsz = None

        self.SLMTargets_tracedFF_stims_dff_outsz = None
        self.SLMTargets_tracedFF_stims_dffAvg_outsz = None
        self.SLMTargets_tracedFF_stims_dfstdF_outsz = None
        self.SLMTargets_tracedFF_stims_dfstdF_avg_outsz = None
        self.SLMTargets_tracedFF_stims_raw_outsz = None
        self.SLMTargets_tracedFF_stims_rawAvg_outsz = None
        
        self.SLMTargets_stims_dff_insz = None
        self.SLMTargets_stims_dffAvg_insz = None
        self.SLMTargets_stims_dfstdF_insz = None
        self.SLMTargets_stims_dfstdF_avg_insz = None
        self.SLMTargets_stims_raw_insz = None
        self.SLMTargets_stims_rawAvg_insz = None

        self.SLMTargets_tracedFF_stims_dff_insz = None
        self.SLMTargets_tracedFF_stims_dffAvg_insz = None
        self.SLMTargets_tracedFF_stims_dfstdF_insz = None
        self.SLMTargets_tracedFF_stims_dfstdF_avg_insz = None
        self.SLMTargets_tracedFF_stims_raw_insz = None
        self.SLMTargets_tracedFF_stims_rawAvg_insz = None


        # photostim responses
        self.responses_SLMtargets_dfstdf_outsz = None  # dFstdF responses for all SLM targets for photostim trials outside sz
        self.responses_SLMtargets_dfstdf_insz = None  # dFstdF responses for all SLM targets for photostim trials outside sz - excluding targets inside the sz boundary
        self.responses_SLMtargets_dfprestimf_outsz = None  # dF/prestimF responses for all SLM targets for photostim trials outside sz
        self.responses_SLMtargets_dfprestimf_insz = None  # dF/prestimF responses for all SLM targets for photostim trials inside sz - excluding targets inside the sz boundary
        self.responses_SLMtargets_tracedFF_outsz = None  # delta(trace_dFF) responses for all SLM targets for photostim trials outside sz
        self.responses_SLMtargets_tracedFF_insz = None  # delta(trace_dFF) responses for all SLM targets for photostim trials inside sz - excluding targets inside the sz boundary
        self.responses_SLMtargets_tracedFF_avg_df = None  # delta(trace_dFF) responses in dataframe for all stims averaged over all targets (+ out sz or in sz variable assignment)

        self.StimSuccessRate_SLMtargets_outsz = None  # photostim sucess rate (not sure exactly if across all stims or not?)
        self.StimSuccessRate_SLMtargets_insz = None  # photostim sucess rate (not sure exactly if across all stims or not?)


        ## breaking down success and failure stims
        self.outsz_traces_SLMtargets_tracedFF_successes_avg = None  # trace snippets for only successful stims - delta(trace_dff) - outsz stims
        self.outsz_traces_SLMtargets_tracedFF_failures_avg = None  # trace snippets for only failure stims - delta(trace_dff) - outsz stims
        self.outsz_traces_SLMtargets_successes_avg_dfstdf = None  # trace snippets for only successful stims - normalized by dfstdf - outsz stims
        self.outsz_traces_SLMtargets_failures_avg_dfstdf = None  # trace snippets for only failure stims - normalized by dfstdf - outsz stims
        self.insz_traces_SLMtargets_tracedFF_successes_avg = None  # trace snippets for only successful stims - delta(trace_dff) - insz stims only (not sure if sz boundary considered for excluding targets)
        self.insz_traces_SLMtargets_tracedFF_failures_avg = None  # trace snippets for only failure stims - delta(trace_dff) - ^^^
        self.insz_traces_SLMtargets_successes_avg_dfstdf = None  # trace snippets for only successful stims - normalized by dfstdf - ^^^
        self.insz_traces_SLMtargets_failures_avg_dfstdf = None  # trace snippets for only failures stims - normalized by dfstdf - ^^^

        ## distances and responses relative to distances to sz wavefront
        self.distance_to_sz = {'SLM Targets': {'uninitialized'},
                               's2p nontargets': {'uninitialized'}}  # calculating the distance between the sz wavefront and cells (data in pixels!)
        self.responses_vs_distance_to_seizure_SLMTargets = None  # dataframe that contains min distance to seizure for each target and responses (zscored)
        self.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets = None

        ## collect information about seizures
        self.collect_seizures_info(seizures_lfp_timing_matarray=paths['matlab_pairedmeasurement_path'], discard_all=discard_all)

        self.save()

    def __repr__(self):
        lastmod = time.ctime(os.path.getmtime(self.pkl_path))
        if not hasattr(self, 'metainfo'):
            information = f"uninitialized"
        else:
            prep = self.metainfo['animal prep.']
            trial = self.metainfo['trial']
            information = f"{prep} {trial}"
        return repr(f"({information}) TwoPhotonImaging.alloptical.Post4ap experimental data object, last saved: {lastmod}")

    @property
    def numSeizures(self):
        return len(self.seizure_lfp_onsets) - (len(self.seizure_lfp_onsets) - len(self.seizure_lfp_offsets))

    @property
    def stim_idx_outsz(self):
        return [idx for idx, stim in enumerate(self.stim_start_frames) if stim in self.stims_out_sz]

    @property
    def stim_idx_insz(self):
        return [idx for idx, stim in enumerate(self.stim_start_frames) if stim in self.stims_in_sz]

    @property
    def im_idx_insz(self):
        """frame # / frame idx for frames in seizure"""
        return self.seizure_frames

    @property
    def im_idx_outsz(self):
        """frame # / frame idx for frames out of seizure"""
        return [fr for fr in range(self.n_frames) if fr not in self.seizure_frames]

    def sz_border_path(expobj, stim):
        return "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.date, expobj.trial, stim)

    def _close_to_edge(expobj, yline: tuple):
        """returns whether the 'yline' (meant to represent the two y-values of the two coords representing the seizure wavefront)
         is close to the edge of the frame"""
        pixels = int(50 / expobj.pix_sz_x)
        if (yline[0] < pixels and yline[1] < pixels) or (
                yline[0] > expobj.frame_y - pixels and yline[0] > expobj.frame_y - pixels):
            return False
        else:
            return True

    def sz_locations_stims(expobj):
        """Creates a dataframe that, for each stim frame inside of ictal period, defines various information about the seizure boundary."""

        expobj.stimsSzLocations = pd.DataFrame(data=None, index=expobj.stims_in_sz, columns=['sz_num', 'coord1', 'coord2', 'wavefront_in_frame'])

        # specify stims for classifying cells
        on_ = []
        for sz_start in expobj.seizure_lfp_onsets:
            if sz_start == 0: # this is used to check if 2p imaging is starting mid-seizure (which should be signified by the first lfp onset being set at frame # 0)
                on_ = on_ + [expobj.stim_start_frames[0]]
            else:

                on_.extend([pj.findClosest(arr=expobj.stims_in_sz, input=sz_start)[0]]) if expobj.stim_start_frames[0] < sz_start < expobj.stim_start_frames[-1] else None
        # on_.extend(expobj.stims_bf_sz)
        # if len(expobj.seizure_lfp_offsets) != len(on_):
        #     end = expobj.stims_af_sz + [expobj.stim_start_frames[-1]]
        # else:
        #     end = expobj.stims_af_sz

        end = []
        for sz_stop in expobj.seizure_lfp_offsets:
            end.extend([pj.findClosest(arr=expobj.stims_in_sz, input=sz_stop)[0]]) if expobj.stim_start_frames[0] < sz_stop < expobj.stim_start_frames[-1] else None
        if len(end) != len(on_):
            end.extend([expobj.stim_start_frames[-1]])

        print(f'\n\t\- seizure start closest stim frames: {on_} [{len(on_)}]')
        print(f'\t\- seizure end closest stim frames: {end} [{len(end)}]\n')

        sz_num = 0
        for on, off in zip(on_, end):
            stims_of_interest = [stim for stim in expobj.stim_start_frames if on <= stim <= off if stim != expobj.stims_in_sz[0]]
            # stims_of_interest_ = [stim for stim in stims_of_interest if expobj._sz_wavefront_stim(stim=stim)]
            # expobj.stims_sz_wavefront.append(stims_of_interest_)

            for _, stim in enumerate(stims_of_interest):
                if os.path.exists(expobj.sz_border_path(stim=stim)):
                    xline, yline = pj.xycsv(csvpath=expobj.sz_border_path(stim=stim))
                    expobj.stimsSzLocations.loc[stim, :] = [sz_num, [xline[0], yline[0]], [xline[1], yline[1]], None]

                    j = expobj._close_to_edge(tuple(yline))
                    expobj.stimsSzLocations.loc[stim, 'wavefront_in_frame'] = j

            sz_num += 1
        expobj.save()

    @property
    def stimsWithSzWavefront(expobj):
        return list(expobj.stimsSzLocations[expobj.stimsSzLocations['wavefront_in_frame'] == True].index)

    def _InOutSz(self, cell_med: list, stim_frame: int): ## TODO update function description
        """
        Returns True if the given cell's location is inside the seizure boundary which is defined as the coordinates
        given in the .csv sheet.

        :param cell_med: from stat['med'] of the cell (stat['med'] refers to a suite2p results obj); the y and x (respectively) coordinates
        :param sz_border_path: path to the csv file generated by ImageJ macro for the seizure boundary
        :param to_plot: make plot showing the boundary start, end and the location of the cell in question
        :return: bool

        # examples
        >>> self = import_expobj(prep='RL108', trial='t-013')  # import expobj trial
        >>> cell_med = self.stat[0]['med']  # pick cell from suite2p list
        >>> stim_frame = self.stim_start_frames[1]
        >>> self._InOutSz(cell_med, stim_frame)
        """

        y = cell_med[0]
        x = cell_med[1]

        # for path in os.listdir(sz_border_path):
        #     if all(s in path for s in ['.csv', self.sheet_name]):
        #         csv_path = os.path.join(sz_border_path, path)

        # xline = []
        # yline = []
        # with open(sz_border_path) as csv_file:
        #     csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
        #     for row in csv_file:
        #         xline.append(int(float(row['xcoords'])))
        #         yline.append(int(float(row['ycoords'])))
        #
        # xline, yline = pj.xycsv(csvpath=sz_border_path)
        #
        # # assumption = line is monotonic
        # line_argsort = np.argsort(yline)
        # xline = np.array(xline)[line_argsort]
        # yline = np.array(yline)[line_argsort]

        coord1, coord2 = self.stimsSzLocations.loc[stim_frame, ['coord1', 'coord2']]
        xline = [coord1[0], coord2[0]]
        yline = [coord1[1], coord2[1]]

        i = bisect.bisect(yline, y)
        if i >= len(yline):
            i = len(yline) - 1
        elif i == 0:
            i = 1

        frame_x = int(self.frame_x / 2)
        half_frame_y = int(self.frame_y / 2)

        d = (x - xline[i]) * (yline[i - 1] - yline[i]) - (y - yline[i]) * (xline[i - 1] - xline[i])
        ds1 = (0 - xline[i]) * (yline[i - 1] - yline[i]) - (half_frame_y - yline[i]) * (xline[i - 1] - xline[i])
        ds2 = (frame_x - xline[i]) * (yline[i - 1] - yline[i]) - (half_frame_y - yline[i]) * (xline[i - 1] - xline[i])

        # if to_plot:  # plot the sz boundary points
        #     # pj.plot_cell_loc(self, cells=[cell], show=False)
        #     plt.scatter(x=xline[0], y=yline[0])
        #     plt.scatter(x=xline[1], y=yline[1])
        #     # plt.show()

        if np.sign(d) == np.sign(ds1):
            return True
        elif np.sign(d) == np.sign(ds2):
            return False
        else:
            return False

    def classify_cells_sz_bound(self, stim, to_plot=True, title=None, flip=False, fig=None, ax=None, text=None):
        """
        using Rob's suggestions to define boundary of the seizure in ImageJ and then read in the ImageJ output,
        and use this to classify cells as in seizure or out of seizure in a particular image (which will relate to stim time).

        :param sz_border_path: str; path to the .csv containing the points specifying the seizure border for a particular stim image
        :param to_plot: make plot showing the boundary start, end and the location of the cell in question
        :param title:
        :param flip: use True if the seizure orientation is from bottom right to top left.
        :return in_sz = ls; containing the cell_ids of cells that are classified inside the seizure area
        """
        from _utils_ import alloptical_plotting as aoplot

        in_sz = []
        out_sz = []
        for idx, s in enumerate(self.stat):
            if self.stat[idx]['original_index'] in self.s2p_nontargets:
                in_seizure = self._InOutSz(cell_med=s['med'], stim_frame=stim)

                if in_seizure is True:
                    in_sz.append(s['original_index'])  # this is the s2p cell id
                elif in_seizure is False:
                    out_sz.append(s['original_index'])  # this is the s2p cell id

        if flip:
            in_sz_final = out_sz
            out_sz_final = in_sz
        else:
            in_sz_final = in_sz
            out_sz_final = out_sz

        if to_plot:  # plot the sz boundary points
            # xline = []
            # yline = []
            # with open(sz_border_path) as csv_file:
            #     csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
            #     for row in csv_file:
            #         xline.append(int(float(row['xcoords'])))
            #         yline.append(int(float(row['ycoords'])))
            # # assumption = line is monotonic
            # line_argsort = np.argsort(yline)
            # xline = np.array(xline)[line_argsort]
            # yline = np.array(yline)[line_argsort]
            coord1, coord2 = self.stimsSzLocations.loc[stim, ['coord1', 'coord2']]
            xline = [coord1[0], coord2[0]]
            yline = [coord1[1], coord2[1]]

            # pj.plot_cell_loc(self, cells=[cell], show=False)
            # plot sz boundary points
            if fig is None:
                fig, ax = plt.subplots(figsize=[5, 5])

            ax.scatter(x=xline[0], y=yline[0], facecolors='#1A8B9D')
            ax.scatter(x=xline[1], y=yline[1], facecolors='#B2D430')
            # fig.show()

            # plot SLM targets in sz boundary
            # coords_to_plot = [s['med'] for cell, s in enumerate(self.stat) if cell in in_sz_final]
            # read in avg stim image to use as the background
            avg_stim_img_path = '%s/%s_%s_stim-%s.tif' % (
            self.analysis_save_path[:-1] + 'avg_stim_images', self.metainfo['date'], self.metainfo['trial'], stim)
            bg_img = tf.imread(avg_stim_img_path)
            fig, ax = aoplot.plot_cells_loc(self, cells=in_sz_final, fig=fig, ax=ax, title=title, show=False,
                                            background=bg_img, cmap='gray', text=text,
                                            edgecolors='orange', show_s2p_targets=False)
            fig, ax = aoplot.plot_cells_loc(self, cells=out_sz_final, fig=fig, ax=ax, title=title, show=False,
                                            background=bg_img, cmap='gray', text=text,
                                            edgecolors='yellowgreen', show_s2p_targets = False)

            # plt.gca().invert_yaxis()
            # plt.show()  # the indiviual cells were plotted in ._InOutSz

            # flip = input("do you need to flip the cell classification?? (ans: yes or no)")
        # else:
        #     flip = False
        #
        # # flip = True

        # # plot again, to make sure that the flip worked
        # fig, ax = plt.subplots(figsize=[5, 5])
        # ax.scatter(x=xline[0], y=yline[0], facecolors='#1A8B9D')
        # ax.scatter(x=xline[1], y=yline[1], facecolors='#B2D430')
        # # fig.show()
        #
        # # plot SLM targets in sz boundary
        # coords_to_plot = [self.target_coords_all[cell] for cell in in_sz]
        # fig, ax = aoplot.plotSLMtargetsLocs(self, targets_coords=coords_to_plot, fig=fig, ax=ax, cells=in_sz, title=title + ' corrected',
        #                           show=False)
        # plt.gca().invert_yaxis()
        # plt.show()  # the indiviual cells were plotted in ._InOutSz

        if to_plot:
            return in_sz_final, out_sz_final, fig, ax
        else:
            return in_sz_final, out_sz_final


    def classify_slmtargets_sz_bound(self, stim, to_plot=True, title=None, flip=False, fig=None, ax=None, text=None):
        """
        going to use Rob's suggestions to define boundary of the seizure in ImageJ and then read in the ImageJ output,
        and use this to classify cells as in seizure or out of seizure in a particular image (which will relate to stim time).

        :param sz_border_path: str; path to the .csv containing the points specifying the seizure border for a particular stim image
        :param to_plot: make plot showing the boundary start, end and the location of the cell in question
        :param title:
        :param flip: use True if the seizure orientation is from bottom right to top left.
        :return in_sz = ls; containing the cell_ids of cells that are classified inside the seizure area
        """
        from _utils_ import alloptical_plotting as aoplot

        in_sz = []
        out_sz = []
        for cell, _ in enumerate(self.target_coords_all):
            if cell % 10 == 0:
                msg = f"\t|- cell #: {cell}"
                print(msg)
            x = self._InOutSz(cell_med=[self.target_coords_all[cell][1], self.target_coords_all[cell][0]],
                              stim_frame=stim)

            if x is True:
                in_sz.append(cell)
            elif x is False:
                out_sz.append(cell)

        if flip:
            in_sz_2 = in_sz
            in_sz = out_sz
            out_sz = in_sz_2

        if to_plot:  # plot the sz boundary points
            # xline = []
            # yline = []
            # with open(sz_border_path) as csv_file:
            #     csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
            #     for row in csv_file:
            #         xline.append(int(float(row['xcoords'])))
            #         yline.append(int(float(row['ycoords'])))
            # # assumption = line is monotonic
            # line_argsort = np.argsort(yline)
            # xline = np.array(xline)[line_argsort]
            # yline = np.array(yline)[line_argsort]
            coord1, coord2 = self.stimsSzLocations.loc[stim, ['coord1', 'coord2']]
            xline = [coord1[0], coord2[0]]
            yline = [coord1[1], coord2[1]]

            # pj.plot_cell_loc(self, cells=[cell], show=False)
            # plot sz boundary points
            if fig is None:
                fig, ax = plt.subplots(figsize=[5, 5])

            ax.scatter(x=xline[0], y=yline[0], facecolors='#1A8B9D')
            ax.scatter(x=xline[1], y=yline[1], facecolors='#B2D430')
            ax.plot([xline[0], xline[1]], [yline[0], yline[1]], c='white',
                    linestyle='dashed', alpha=0.3)

            # fig.show()

            # plot SLM targets in sz boundary
            coords_to_plot_insz = [self.target_coords_all[cell] for cell in in_sz]
            coords_to_plot_outsz = [self.target_coords_all[cell] for cell in out_sz]
            # read in avg stim image to use as the background
            avg_stim_img_path = '%s/%s_%s_stim-%s.tif' % (
            self.analysis_save_path[:-1] + 'avg_stim_images', self.metainfo['date'], self.metainfo['trial'], stim)
            avg_stim_img_path = f"{self.avg_stim_images_path}/{self.t_series_name}_stim-{stim}.tif"
            bg_img = tf.imread(avg_stim_img_path)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, cells=in_sz, edgecolors='yellowgreen', background=bg_img)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, cells=out_sz, edgecolors='white', background=bg_img)
            fig, ax = aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, fig=fig, ax=ax, cells=in_sz,
                                                  title=title, show=False, background=bg_img, text=text,
                                                  edgecolors='red')
            # fig, ax = aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, fig=fig, ax=ax,
            #                                       cells=out_sz, title=title, show=False, background=bg_img,
            #                                       edgecolors='yellowgreen')

            # plt.gca().invert_yaxis()
            # plt.show()  # the indiviual cells were plotted in ._InOutSz

            # flip = input("do you need to flip the cell classification?? (ans: yes or no)")
        # else:
        #     flip = False
        #
        # # flip = True

        # # plot again, to make sure that the flip worked
        # fig, ax = plt.subplots(figsize=[5, 5])
        # ax.scatter(x=xline[0], y=yline[0], facecolors='#1A8B9D')
        # ax.scatter(x=xline[1], y=yline[1], facecolors='#B2D430')
        # # fig.show()
        #
        # # plot SLM targets in sz boundary
        # coords_to_plot = [self.target_coords_all[cell] for cell in in_sz]
        # fig, ax = aoplot.plotSLMtargetsLocs(self, targets_coords=coords_to_plot, fig=fig, ax=ax, cells=in_sz, title=title + ' corrected',
        #                           show=False)
        # plt.gca().invert_yaxis()
        # plt.show()  # the indiviual cells were plotted in ._InOutSz

        else:
            pass

        if to_plot:
            return in_sz, out_sz, fig, ax
        else:
            return in_sz, out_sz

    def is_cell_insz(self, cell, stim):
        """for a given cell and stim, return True if cell is inside the sz boundary."""
        if hasattr(self, 'slmtargets_szboundary_stim'):
            if stim in self.slmtargets_szboundary_stim.keys():
                if cell in self.slmtargets_szboundary_stim[stim]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            # return False  # not all expobj will have the sz boundary classes attr so for those just assume no seizure
            raise Exception(
                'cannot check for cell inside sz boundary because cell sz classification hasnot been performed yet')

    def subselect_tiffs_sz(self, onsets, offsets, on_off_type: str):
        """subselect raw tiff movie over all seizures as marked by onset and offsets. save under analysis path for object.
        Note that the onsets and offsets definitions may vary, so check exactly what was used in those args."""

        print('-----Making raw sz movies by cropping original raw tiff')
        if hasattr(self, 'analysis_save_path'):
            pass
        else:
            raise ValueError(
                'need to add the analysis_save_path attr before using this function -- this is where it will save to')

        print('reading in seizure trial from: ', self.tiff_path, '\n')
        stack = tf.imread(self.tiff_path)

        # subselect raw tiff movie over all seizures as marked by LFP onset and offsets
        for on, off in zip(onsets, offsets):
            select_frames = (on, off)
            print('cropping sz frames', select_frames)
            save_as = self.analysis_save_path + '/%s_%s_subselected_%s_%s_%s.tif' % (self.metainfo['date'],
                                                                                     self.metainfo['trial'],
                                                                                     select_frames[0], select_frames[1],
                                                                                     on_off_type)
            pj.subselect_tiff(tiff_stack=stack, select_frames=select_frames, save_as=save_as)
        print('\ndone. saved to:', self.analysis_save_path)

    def collect_seizures_info(self, seizures_lfp_timing_matarray=None, discard_all=True):
        from _utils_ import alloptical_plotting as aoplot

        print('\ncollecting information about seizures...')
        if seizures_lfp_timing_matarray is not None:
            self.seizures_lfp_timing_matarray = seizures_lfp_timing_matarray  # path to the matlab array containing paired measurements of seizures onset and offsets

        assert self.seizures_lfp_timing_matarray is not None

        # retrieve seizure onset and offset times from the seizures info array input
        paq = paq_read(file_path=self.paq_path, plot=False)

        # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
        # NOTE: the output of all of the following function is in dimensions of the FRAME CLOCK (not official paq clock time)
        if self.seizures_lfp_timing_matarray is not None:
            print('-- using matlab array to collect seizures %s: ' % seizures_lfp_timing_matarray)
            bad_frames, self.seizure_frames, self.seizure_lfp_onsets, self.seizure_lfp_offsets = frames_discard(
                paq=paq[0], input_array=self.seizures_lfp_timing_matarray, total_frames=self.n_frames,
                discard_all=discard_all)
            print(
                f"|- sz frame # onsets: {self.seizure_lfp_onsets}, \n|- sz frame # offsets {self.seizure_lfp_offsets}")
            print('\n|-now creating raw movies for each sz as well (saved to the /Analysis folder) ... ')
            self.subselect_tiffs_sz(onsets=self.seizure_lfp_onsets, offsets=self.seizure_lfp_offsets,
                                    on_off_type='lfp_onsets_offsets')

            print('\n|-now classifying photostims at phases of seizures ... ')
            self.stims_in_sz = [stim for stim in self.stim_start_frames if stim in self.seizure_frames]
            self.stims_out_sz = [stim for stim in self.stim_start_frames if stim not in self.seizure_frames]

            # self.stims_bf_sz = [self.stim_start_frames[self.stim_start_frames.index(sz_start) - 1] for sz_start in self.seizure_lfp_onsets]

            self.stims_bf_sz = [stim for stim in self.stim_start_frames
                                for sz_start in self.seizure_lfp_onsets
                                if 0 < (sz_start - stim) < 10 * self.fps]  # select stims that occur within 10 seconds before of the sz onset
            self.stims_af_sz = [stim for stim in self.stim_start_frames
                                for sz_start in self.seizure_lfp_offsets
                                if 0 < -1 * (
                                        sz_start - stim) < 10 * self.fps]  # select stims that occur within 10 seconds afterof the sz offset
            print(' \n|- stims_in_sz:', self.stims_in_sz, ' \n|- stims_out_sz:', self.stims_out_sz,
                  ' \n|- stims_bf_sz:', self.stims_bf_sz, ' \n|- stims_af_sz:', self.stims_af_sz)
            aoplot.plot_lfp_stims(expobj=self)
        else:
            print('-- no matlab array given to use for finding seizures.')
            bad_frames = frames_discard(paq=paq[0], input_array=seizures_lfp_timing_matarray,
                                        total_frames=self.n_frames,
                                        discard_all=discard_all)

        print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
        print('|- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])
        self.append_bad_frames(bad_frames=bad_frames)  # here only need to append the bad frames to the expobj.bad_frames property

        self.save_pkl()

    def find_closest_sz_frames(self):
        """finds time from the closest seizure onset on LFP (-ve values for forthcoming, +ve for past)
        FOR each photostim timepoint"""

        self.closest_sz = {'stim': [], 'closest sz on (frames)': [], 'closest sz off (frames)': [],
                           'closest sz (instance)': []}
        for stim in self.stim_start_frames:
            differences_on = stim - self.seizure_lfp_onsets
            differences_off = stim - self.seizure_lfp_offsets

            # some math to figure out the closest seizure on and off frames from the ls of sz LFP stamps and current stim time
            y = abs(differences_on)
            x = min(y)
            closest_sz_on = differences_on[np.where(y == x)[0][0]]
            y_off = abs(differences_off)
            x_off = min(y_off)
            closest_sz_off = differences_off[np.where(y_off == x_off)[0][0]]

            sz_number = np.where(differences_on == closest_sz_on)[0][
                0]  # the seizure instance out of the total # of seizures
            self.closest_sz['stim'].append(stim)
            self.closest_sz['closest sz on (frames)'].append(closest_sz_on)
            self.closest_sz['closest sz off (frames)'].append(closest_sz_off)
            self.closest_sz['closest sz (instance)'].append(sz_number)

    def MeanSeizureImages(self, baseline_tiff: str = None, frames_last: int = 0, force_redo: bool = False):
        """
        used to make mean images of all seizures contained within an individual expobj trial. the averaged images
        are also subtracted from baseline_tiff image to give a difference image that should highlight the seizure well.

        :param force_redo:
        :param baseline_tiff: path to the baseline tiff file to use
        :param frames_last: use to specify the tail of the seizure frames for images.
        :return:
        """

        if force_redo:
            continu = True
        elif hasattr(self, 'meanszimages_r'):
            if self.meanszimages_r is True:
                continu = False
            else:
                continu = True
        else:
            continu = True

        if continu:

            # if baseline_tiff is None:
            #     print('WARNING: not subtracting by baseline_tiff, none provided.')
            #     im_stack_base = np.zeros(shape=[self.frame_x, self.frame_y])
            # else:
            #     print('First loading up and plotting baseline (comparison) tiff from: ', baseline_tiff)
            #     im_stack_base = tf.imread(baseline_tiff, key=range(5000))  # reading in just the first 5000 frames of the spont
            #     avg_baseline = np.mean(im_stack_base, axis=0)
            #     plt.imshow(avg_baseline, cmap='gray')
            #     plt.suptitle('avg 5000 frames baseline from %s' % baseline_tiff[-35:], wrap=True)
            #     plt.show()

            tiffs_loc = '%s/*Ch3.tif' % self.tiff_path_dir
            tiff_path = glob.glob(tiffs_loc)[0]
            print('loading up run_post4ap_trials tiff from: ', tiff_path)
            im_stack = tf.imread(tiff_path, key=range(self.n_frames))
            print('Processing seizures from experiment tiff (wait for all seizure comparisons to be processed), \n '
                  'total tiff shape: ', im_stack.shape)
            avg_sub_list = []
            im_sub_list = []
            im_diff_list = []
            counter = 0
            for sz_on, sz_off in zip(self.seizure_lfp_onsets, self.seizure_lfp_offsets):
                # subselect for frames within sz on and sz off, and plot average and difference compared to the baseline
                if frames_last != 0:
                    im_sub = im_stack[sz_off - frames_last:sz_off]  # trying out last 1000 frames from seizure_offset
                else:
                    im_sub = im_stack[
                             sz_on:sz_off]  # take the whole seizure period (as defined by the LFP onset and offsets)
                avg_sub = np.mean(im_sub, axis=0)
                plt.imshow(avg_sub, cmap='gray')
                plt.suptitle('avg of seizure from %s to %s frames' % (sz_on, sz_off))
                plt.show()  # just plot for now to make sure that you are doing things correctly so far

                # im_diff = avg_sub - avg_baseline
                # plt.imshow(im_diff, cmap='gray')
                # plt.suptitle('diff of seizure from %s to %s frames' % (sz_on, sz_off))
                # plt.show()  # just plot for now to make sure that you are doing things correctly so far

                avg_sub_list.append(avg_sub)
                im_sub_list.append(im_sub)
                # im_diff_list.append(im_diff)

                counter += 1

                ## create downsampled TIFFs for each sz
                pj.SaveDownsampledTiff(stack=im_sub, save_as=self.analysis_save_path + '%s_%s_sz%s_downsampled.tiff' % (self.metainfo['date'], self.metainfo['trial'], counter))


            self.avg_sub_list = avg_sub_list
            self.meanszimages_r = True

        else:
            print('skipping remaking of mean sz images')


    # def _trialProcessing_nontargets(expobj, normalize_to='pre-stim', save=True, stims: Union[list, str] = 'all'):
    #     """
    #     Uses dfstdf traces for individual cells and photostim trials, calculate the mean amplitudes of response and
    #     statistical significance across all trials for all cells
    #
    #     Inputs:
    #         plane             - imaging plane n
    #     """
    #
    #     print('\n----------------------------------------------------------------')
    #     print('running trial Processing for nontargets ')
    #     print('----------------------------------------------------------------')
    #
    #     # define non targets from suite2p ROIs (exclude cells in the SLM targets exclusion - .s2p_cells_exclude)
    #     expobj.s2p_nontargets = [cell for cell in expobj.good_cells if cell not in expobj.s2p_cells_exclude]  ## exclusion of cells that are classified as s2p_cell_targets
    #
    #     ## collecting nontargets stim traces from in sz imaging frames
    #     # - - collect stim traces as usual for all stims, then use the sz boundary dictionary to nan cells/stims insize sz boundary
    #     # make trial arrays from dff data shape: [cells x stims x frames]
    #     # stim_timings_outsz = [stim for stim in expobj.stim_start_frames if stim not in expobj.seizure_frames]; stim_timings=expobj.stims_out_sz
    #     expobj._makeNontargetsStimTracesArray(stim_frames=expobj.stim_start_frames, normalize_to=normalize_to)
    #
    #     # create parameters, slices, and subsets for making pre-stim and post-stim arrays to use in stats comparison
    #     # test_period = expobj.pre_stim_response_window_msec / 1000  # sec
    #     # expobj.test_frames = int(expobj.fps * test_period)  # test period for stats
    #     expobj.pre_stim_frames_test = np.s_[expobj.pre_stim - expobj.pre_stim_response_frames_window: expobj.pre_stim]
    #     stim_end = expobj.pre_stim + expobj.stim_duration_frames
    #     expobj.post_stim_frames_slice = np.s_[stim_end: stim_end + expobj.post_stim_response_frames_window]
    #
    #     ## process out sz stims
    #     # mean pre and post stimulus (within post-stim response window) flu trace values for all cells, all trials
    #     stims_outsz = [i for i, stim in enumerate(expobj.stim_start_frames) if stim not in expobj.stims_in_sz]
    #     expobj.analysis_array_outsz = expobj.dfstdF_traces_nontargets[:, stims_outsz, :]  # NOTE: USING dF/stdF TRACES
    #     expobj.raw_traces_outsz = expobj.raw_traces_nontargets[:, stims_outsz, :]
    #     expobj.dff_traces_outsz = expobj.dff_traces_nontargets[:, stims_outsz, :]
    #
    #     ## checking that there are no additional nan's being added from the code below (unless its specifically for the cell exclusion part)
    #     # print(f"analysis array outsz nan's: {sum(np.isnan(expobj.analysis_array_outsz))}")
    #     # print(f"dfstdF_traces_nontargets nan's: {sum(np.isnan(expobj.dfstdF_traces_nontargets))}")
    #     # assert sum(np.isnan(expobj.dfstdF_traces_nontargets[0][0])) == sum(np.isnan(expobj.analysis_array_outsz[0][0])), print('there is a discrepancy in the number of nans in expobj.analysis_array_outsz')
    #
    #     expobj.pre_array_outsz = np.nanmean(expobj.analysis_array_outsz[:, :, expobj.pre_stim_frames_test],
    #                                         axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
    #     expobj.post_array_outsz = np.nanmean(expobj.analysis_array_outsz[:, :, expobj.post_stim_frames_slice],
    #                                          axis=1)  # [cells x poststim frames] (avg'd taken over all stims)
    #
    #     ## process in sz stims - use all cells
    #     # mean pre and post stimulus (within post-stim response window) flu trace values for all cells, all trials
    #     stims_sz = [i for i, stim in enumerate(expobj.stim_start_frames) if
    #                 stim in list(expobj.slmtargets_szboundary_stim.keys())]
    #     expobj.analysis_array_insz = expobj.dfstdF_traces_nontargets[:, stims_sz, :]  # NOTE: USING dF/stdF TRACES
    #     expobj.raw_traces_insz = expobj.raw_traces_nontargets[:, stims_sz, :]
    #     expobj.dff_traces_insz = expobj.dff_traces_nontargets[:, stims_sz, :]
    #     expobj.pre_array_insz = np.nanmean(expobj.analysis_array_insz[:, :, expobj.pre_stim_frames_test],
    #                                        axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
    #     expobj.post_array_insz = np.nanmean(expobj.analysis_array_insz[:, :, expobj.post_stim_frames_slice],
    #                                         axis=1)  # [cells x poststim frames] (avg'd taken over all stims)
    #
    #     ## process in sz stims - exclude cells inside sz boundary
    #     analysis_array_insz_ = expobj.analysis_array_insz
    #     raw_traces_ = expobj.raw_traces_insz
    #     dff_traces_ = expobj.dff_traces_insz
    #     ## add nan's where necessary
    #     for x, stim_idx in enumerate(stims_sz):
    #         stim = expobj.stim_start_frames[stim_idx]
    #         exclude_cells_list = [idx for idx, cell in enumerate(expobj.s2p_nontargets) if
    #                               cell in expobj.slmtargets_szboundary_stim[stim]]
    #         analysis_array_insz_[exclude_cells_list, x, :] = [np.nan] * expobj.analysis_array_insz.shape[2]
    #         raw_traces_[exclude_cells_list, x, :] = [np.nan] * expobj.raw_traces_insz.shape[2]
    #         dff_traces_[exclude_cells_list, x, :] = [np.nan] * expobj.dff_traces_insz.shape[2]
    #
    #     # mean pre and post stimulus (within post-stim response window) flu trace values for all trials, with excluded cells
    #     expobj.analysis_array_insz_exclude = analysis_array_insz_
    #     expobj.raw_traces_insz = raw_traces_
    #     expobj.dff_traces_insz = dff_traces_
    #
    #     expobj.pre_array_insz_exclude = np.nanmean(expobj.analysis_array_insz_exclude[:, :, expobj.pre_stim_frames_test],
    #                                                axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
    #     expobj.post_array_insz_exclude = np.nanmean(expobj.analysis_array_insz_exclude[:, :, expobj.post_stim_frames_slice],
    #                                                 axis=1)  # [cells x poststim frames] (avg'd taken over all stims)
    #
    #     # measure avg response value for each trial, all cells --> return array with 3 axes [cells x response_magnitude_per_stim (avg'd taken over response window)]
    #     expobj.post_array_responses = np.nanmean(expobj.analysis_array_outsz[:, :, expobj.post_stim_frames_slice],
    #                                              axis=2)
    #     expobj.post_array_responses_insz = np.nanmean(expobj.analysis_array_insz[:, :, expobj.post_stim_frames_slice],
    #                                                   axis=2)
    #     expobj.post_array_responses_insz_exclude = np.nanmean(
    #         expobj.analysis_array_insz_exclude[:, :, expobj.post_stim_frames_slice], axis=2)
    #
    #     expobj.wilcoxons = expobj._runWilcoxonsTest(array1=expobj.pre_array_outsz, array2=expobj.post_array_outsz)
    #     expobj.wilcoxons_insz = expobj._runWilcoxonsTest(array1=expobj.pre_array_insz, array2=expobj.post_array_insz)
    #     expobj.wilcoxons_insz_exclude = expobj._runWilcoxonsTest(array1=expobj.pre_array_insz_exclude,
    #                                                              array2=expobj.post_array_insz_exclude)
    #
    #     expobj.save() if save else None

    def calcMinDistanceToSz_newer(self, analyse_cells, show_debug_plot=False):
        """
        Make a dataframe of stim frames x cells, with values being the minimum distance to the sz boundary at the stim.

        :param analyse_cells: either 'SLM Targets' or 's2p nontargets'
        :param show_debug_plot:
        :return:

        """

        # assert hasattr(self.ExpSeizure, 'slmtargets_szboundary_stim'), 'no slmtargets_szboundary_stim found for experiment trial.'
        assert hasattr(self.ExpSeizure, 'nontargets_szboundary_stim'), 'no nontargets_szboundary_stim found for experiment trial.'

        print(f'\t\- Calculating min distances to sz boundaries for {analyse_cells} ... ')

        if analyse_cells == 'SLM Targets':
            coordinates = self.target_coords_all
            indexes = range(len(self.target_coords_all))
            cellsInSzStims = self.slmtargets_szboundary_stim
        elif analyse_cells == 's2p nontargets':
            print('WARNING (TESTING ANLAYSES): testing collecting min. distances for s2p nontargets')
            indexes = [cell for cell in self.s2p_nontargets if cell not in self.s2p_nontargets_exclude]
            cellsInSzStims = self.ExpSeizure.nontargets_szboundary_stim
            coordinates = []
            for stat_ in self.stat:
                if stat_['original_index'] in indexes:
                    y, x = stat_['med']
                    coordinates.append([x, y])
                ### todo need to test collecting coordinates for nontargets - seems wrong ones are being chosen.
        else:
            raise ValueError('incorrect analyse cells argument.')

        df = pd.DataFrame(data=None, index=indexes, columns=self.stim_start_frames)
        assert df.shape[0] == self.dff_traces_nontargets.shape[0], 'incorrect number of nontargets traces collected and nontargets being calculated for sz boundary distance.'
        plot_counter = 0  # counts the number of plots created across stims
        for _, stim_frame in enumerate(self.stim_start_frames):

            if stim_frame not in self.stimsWithSzWavefront:
                # exclude sz stims (set to nan) with unknown absolute locations of sz boundary
                df.loc[:, stim_frame] = np.nan
            elif stim_frame in self.stimsWithSzWavefront:
                if 0 <= plot_counter < 15 and show_debug_plot:
                    from _utils_.alloptical_plotting import multi_plot_subplots
                    from _utils_.alloptical_plotting import get_ax_for_multi_plot
                    num_plots = 12
                    fig, axs, counter, ncols, nrows = multi_plot_subplots(num_total_plots=num_plots,
                                                                          ncols=4)
                else:
                    axs = False
                    num_plots = 0
                    counter = 0


                print(f"\ncalculating min distances of targets to sz wavefront for stim frame: {stim_frame} ")

                # calculate line of best fit across the sz boundary coordinates
                xline, yline = pj.xycsv(csvpath=self.sz_border_path(stim=stim_frame))
                try:
                    cellsInSz = cellsInSzStims[stim_frame]
                except KeyError:
                    print('debug here!')
                coord1, coord2 = self.stimsSzLocations.loc[stim_frame, ['coord1', 'coord2']]
                m, c = pj.eq_line_2points(p1=[coord1[0], coord1[1]], p2=[coord2[0], coord2[1]])
                x_range = np.arange(0, self.frame_x * 1.6)
                y_range = list(map(lambda x: m * x + c, x_range))

                for idx, cell_coord in enumerate(coordinates):
                    cell_idx = indexes[idx]
                    cell_coord_ = [cell_coord[0], cell_coord[1], 0]

                    dist, nearest = pnt2line.pnt2line(pnt=cell_coord_, start=[x_range[0], y_range[0], 0], end=[x_range[-1], y_range[-1], 0])
                    dist = round(dist, 2)  # in pixels
                    if cell_idx in cellsInSz:  # flip distance to negative if target is inside sz boundary
                        dist = -dist
                        plot = True
                        # if stim_frame == 624:  # debugging
                        #     print('debug here! - problem with out of sz cells from this stim (and other stims too) being turned to negative distance')

                    else: plot = False
                        # print(dist, "target coord inside sz")
                    df.loc[cell_idx, stim_frame] = dist  # ADD DISTANCE (pixels) TO DATA FRAME !!!!!!!!!!!

                    #  create plots to make sure the distances are being calculated properly - checked on .22/03/15 - all looks good.
                    if axs is not False and counter < num_plots and plot:
                        title = f"distance: {dist}, {self.t_series_name}, stim: {stim_frame}"
                        # fig, ax = plt.subplots()  ## figure for debuggging

                        try:
                            ax, counter = get_ax_for_multi_plot(axs=axs, counter=counter, ncols=ncols)
                            # print(counter, num_plots)
                        except TypeError:
                            print(counter, num_plots)
                            print('debug here')

                        pj.plot_coordinates(coords=[(cell_coord_[0], cell_coord_[1])], frame_x=self.frame_x, frame_y=self.frame_y,
                                            edgecolors='red', show=False, fig=fig, ax=ax, title=title)

                        pj.plot_coordinates(coords=[(xline[0], yline[0]), (xline[1], yline[1])], frame_x=self.frame_x, frame_y=self.frame_y,
                                            edgecolors='green', show=False, fig=fig, ax=ax, title=title)

                        pj.plot_coordinates(coords=[(nearest[0], nearest[1])], frame_x=self.frame_x, frame_y=self.frame_y,
                                            edgecolors='blue', show=False, fig=fig, ax=ax, title=title)

                        ax.plot(np.arange(350, 450), [450]*100, color='white', lw=5, solid_capstyle='butt')  # 100 px measurement bar

                        ax.plot(x_range, y_range, lw=1)




                if 0 <= plot_counter < 15 and show_debug_plot:
                    fig.show()
                    plot_counter += 1

            # aoplot.plot_sz_boundary_location(self)
        self.distance_to_sz[analyse_cells] = df  ## set the dataframe for each of SLM Targets and s2p nontargets
        self.save()
        return self.distance_to_sz[analyse_cells]


    def calcMinDistanceToSz_old(self, analyse_cells, show_debug_plot=False):
        """
        Make a dataframe of stim frames x cells, with values being the minimum distance to the sz boundary at the stim.

        """

        if hasattr(self, 'slmtargets_szboundary_stim'):
            for analyse_cells in ['SLM Targets', 's2p nontargets']:

                print(f'\t\- Calculating min distances to sz boundaries for {analyse_cells} ... ')

                if analyse_cells == 'SLM Targets':
                    coordinates = self.target_coords_all
                    indexes = range(len(self.target_coords_all))
                elif analyse_cells == 's2p nontargets':  # TODO add collecting min. distances for s2p nontargets
                    print('WARNING: still need to add collecting min. distances for s2p nontargets')
                    indexes = self.s2p_nontargets
                    coordinates = []
                    for stat_ in self.stat:
                        coordinates.append(stat_['med']) if stat_['original_index'] in indexes else None
                else:
                    raise ValueError('incorrect analyse cells argument.')

                df = pd.DataFrame(data=None, index=indexes, columns=self.stim_start_frames)
                # fig2, ax2 = plt.subplots()  ## figure for debuggging
                plot_counter = 0
                for _, stim_frame in enumerate(self.stim_start_frames):

                    if analyse_cells == 'SLM Targets':
                        if stim_frame not in self.stimsWithSzWavefront:
                            # exclude sz stims (set to nan) with unknown absolute locations of sz boundary
                            df.loc[:, stim_frame] = np.nan
                        else:
                            if 0 <= plot_counter < 5 and show_debug_plot:
                                from _utils_.alloptical_plotting import multi_plot_subplots
                                from _utils_.alloptical_plotting import get_ax_for_multi_plot

                                # fig, axs, counter, ncols, nrows = multi_plot_subplots(num_total_plots=self.n_targets,
                                #                                                       ncols=4)
                            else:
                                axs = False


                            print(f"\ncalculating min distances of targets to sz wavefront for stim frame: {stim_frame} ")

                            xline, yline = pj.xycsv(csvpath=self.sz_border_path(stim=stim_frame))
                            targetsInSz = self.slmtargets_szboundary_stim[stim_frame]
                            coord1, coord2 = self.stimsSzLocations.loc[stim_frame, ['coord1', 'coord2']]
                            m, c = pj.eq_line_2points(p1=[coord1[0], coord1[1]], p2=[coord2[0], coord2[1]])
                            x_range = np.arange(0, self.frame_x * 1.6)
                            y_range = list(map(lambda x: m * x + c, x_range))

                            for target_idx, target_coord in enumerate(coordinates):
                                target_coord_ = [target_coord[0], target_coord[1], 0]
                                # dist, nearest = pnt2line.pnt2line(pnt=target_coord_, start=[coord1[0], coord1[1], 0], end=[coord2[0], coord2[1], 0])


                                # fig, ax = plt.sub¬plots(figsize=(3,3))
                                # ax.scatter(x=coord1[0], y=coord1[1])
                                # ax.plot(x_range, y_range)
                                # fig.show()

                                dist, nearest = pnt2line.pnt2line(pnt=target_coord_, start=[x_range[0], y_range[0], 0], end=[x_range[-1], y_range[-1], 0])
                                dist = round(dist, 2)
                                if target_idx in targetsInSz:  # flip distance to negative if target is inside sz boundary
                                    dist = -dist
                                    # print(dist, "target coord inside sz")
                                df.loc[target_idx, stim_frame] = dist

                                # #  create plots to make sure the distances are being calculated properly - checked on .22/03/15 - all looks good.
                                # if axs is not False:
                                #     title = f"distance: {dist}, {self.t_series_name}, stim: {stim_frame}"
                                #     # fig, ax = plt.subplots()  ## figure for debuggging
                                #     ax, counter = get_ax_for_multi_plot(axs=axs, counter=counter, ncols=ncols)
                                #     pj.plot_coordinates(coords=[(target_coord_[0], target_coord_[1])], frame_x=self.frame_x, frame_y=self.frame_y,
                                #                         edgecolors='red', show=False, fig=fig, ax=ax, title=title)
                                #
                                #     pj.plot_coordinates(coords=[(xline[0], yline[0]), (xline[1], yline[1])], frame_x=self.frame_x, frame_y=self.frame_y,
                                #                         edgecolors='green', show=False, fig=fig, ax=ax, title=title)
                                #
                                #     pj.plot_coordinates(coords=[(nearest[0], nearest[1])], frame_x=self.frame_x, frame_y=self.frame_y,
                                #                         edgecolors='blue', show=False, fig=fig, ax=ax, title=title)
                                #
                                #     ax.plot(np.arange(350, 450), [450]*100, color='white', lw=5)  # 100 px measurement bar
                                #
                                #     ax.plot(x_range, y_range, lw=1)




                            # if 0 <= plot_counter < 15 and show_debug_plot:
                            #     fig.show()
                            #     plot_counter += 1

                    # aoplot.plot_sz_boundary_location(self)
                self.distance_to_sz[analyse_cells] = df  ## set the dataframe for each of SLM Targets and s2p nontargets
                # fig2.show()
                self.save()
            return self.distance_to_sz
        else:
            print(f'WARNING: {self.t_series_name} doesnot have slmtargets_szboundary_stim completed')
            # return f"{self.t_series_name}"

    def avgResponseSzStims_SLMtargets(self, save=False):
        df = pd.DataFrame(columns=['stim_group', 'avg targets response'], index=self.stims_idx)
        for stim_idx in self.responses_SLMtargets_tracedFF_outsz.columns:
            df.loc[stim_idx, 'stim_group'] = 'interictal'
            df.loc[stim_idx, 'avg targets response'] = self.responses_SLMtargets_tracedFF_outsz.loc[:, stim_idx].mean()

        for stim_idx in self.responses_SLMtargets_tracedFF_insz.columns:
            df.loc[stim_idx, 'stim_group'] = 'ictal'
            df.loc[stim_idx, 'avg targets response'] = self.responses_SLMtargets_tracedFF_insz.loc[:, stim_idx].mean()

        self.responses_SLMtargets_tracedFF_avg_df = df
        self.save() if save else None

    # TODO measuring delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment
    def szInvasionTime(self):
        for i in range(self.numSeizures):
            pass

    # nontargets stim arrays
    def _makeNontargetsStimTracesArray(expobj, stim_frames, normalize_to='pre-stim', save=True, plot=False):
        """
        primary function to retrieve photostimulation trial timed Fluorescence traces for non-targets (ROIs taken from suite2p).
        :param expobj: alloptical experiment object
        :param normalize_to: str; either "baseline" or "pre-stim" or "whole-trace"
        :return: plot of avg_dFF of 100 randomly selected nontargets
        """
        print('\n\- Collecting peri-stim traces [Post4apMain version] ...')

        # collect photostim timed average dff traces of photostim targets
        dff_traces = []
        dff_traces_avg = []

        dfstdF_traces = []
        dfstdF_traces_avg = []

        raw_traces = []
        raw_traces_avg = []


        stim_cells_analyse = {}
        assert 'post' in expobj.exptype
        assert 'in/out sz' in expobj.NonTargetsSzInvasionSpatial.adata.layers
        if stim_frames == expobj.stims_out_sz:
            for cell in expobj.s2p_nontargets:
                stim_cells_analyse[cell] = expobj.stim_idx_outsz
        elif stim_frames == expobj.stims_in_sz:
            for cell in expobj.s2p_nontargets:
                if str(cell) in expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index']:  # note that some nontarget cells are excluded in Nontargets Sz Invasion Spatial . adata - not totally sure yet why - but should be able to find out by looking at that code
                    cell_idx = list(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index']).index(cell)
                    stimidx_to_analyse_for_cell = np.where(expobj.NonTargetsSzInvasionSpatial.adata.layers['in/out sz'][cell_idx, :] == 1)[0]
                    stim_cells_analyse[cell] = stimidx_to_analyse_for_cell
        elif stim_frames == expobj.stim_start_frames:
            for cell in expobj.s2p_nontargets:
                stim_cells_analyse[cell] = expobj.stims_idx

        expobj.s2p_nontargets_exclude = []
        for cell in [*stim_cells_analyse]:
            # print(f'considering cell # {cell},  {len(stim_cells_analyse[cell])} stims')
            cell_idx = expobj.cell_id.index(cell)
            flu_trials = [expobj.raw[cell_idx][expobj.stim_start_frames[stim_idx] - expobj.pre_stim: expobj.stim_start_frames[stim_idx] + expobj.stim_duration_frames + expobj.post_stim] for stim_idx in stim_cells_analyse[cell]]

            dff_trace = Utils.normalize_dff(expobj.raw[cell_idx],
                                            threshold_pct=50)  # normalize trace (dFF) to mean of whole trace

            # if normalize_to == 'baseline':  # probably gonna ax this anyways
            #     flu_dff = []
            #     mean_spont_baseline = np.mean(expobj.baseline_raw[cell_idx])
            #     for i in range(len(flu_trials)):
            #         trace_dff = ((flu_trials[i] - mean_spont_baseline) / mean_spont_baseline) * 100
            #
            #         # add nan if cell is inside sz boundary for this stim
            #         if hasattr(expobj, 'slmtargets_szboundary_stim'):
            #             if expobj.is_cell_insz(cell=cell, stim=stim_timings[i]):
            #                 trace_dff = [np.nan] * len(flu_trials[i])
            #
            #         flu_dff.append(trace_dff)
            #
            # elif normalize_to == 'whole-trace':
            mean_ = np.mean(expobj.raw[cell_idx])
            # dfstdf_trace = (expobj.raw[cell_idx] - np.mean(expobj.raw[cell_idx])) / np.std(expobj.raw[cell_idx], ddof=1)  # normalize trace (dFstdF) to std of whole trace
            if mean_ < 10:  # filter cells with low s2p neu corrected mean trace value
                expobj.s2p_nontargets_exclude.append(cell)
            else:
                # print(f's2p neu. corrected trace statistics: mean: {mean_} (min: {np.min(expobj.raw[cell_idx])}, max: {np.max(expobj.raw[cell_idx])}, std: {np.std(expobj.raw[cell_idx], ddof=1)})')

                flu_dfstdF = []
                flu_dff = []
                # flu_dff_ = [dff_trace[stim - expobj.pre_stim: stim + expobj.stim_duration_frames + expobj.post_stim] for stim in stim_frames]
                flu_dff_ = [dff_trace[stim - expobj.PhotostimAnalysisSlmTargets.pre_stim_fr: stim + expobj.stim_duration_frames + expobj.PhotostimAnalysisSlmTargets.post_stim_fr] for stim in stim_frames]

                for i in range(len(flu_dff_)):
                    trace = flu_dff_[i]
                    mean_pre = np.mean(trace[0:expobj.PhotostimAnalysisSlmTargets.pre_stim_fr])
                    trace_dff = trace - mean_pre  # correct dFF of this trial to mean of pre-stim dFF
                    std_pre = np.std(trace[0:expobj.PhotostimAnalysisSlmTargets.pre_stim_fr], ddof=1)
                    dFstdF = trace_dff / std_pre  # normalize dFF of this trial by std of pre-stim dFF

                    flu_dff.append(trace_dff)
                    flu_dfstdF.append(dFstdF)

                # elif normalize_to == 'pre-stim':
                #     flu_dff = []
                #     flu_dfstdF = []
                #     # print('|- splitting trace by photostim. trials and correcting by pre-stim period')
                #     for i in range(len(flu_trials)):
                #         trace = flu_trials[i]
                #         mean_pre = np.mean(trace[0:expobj.pre_stim])
                #
                #         std_pre = np.std(trace[0:expobj.pre_stim], ddof=1)
                #         # dFstdF = (((trace - mean_pre) / mean_pre) * 100) / std_pre  # make dF divided by std of pre-stim F trace
                #         dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                #
                #         if mean_pre < 1:
                #             # print('risky cell here at cell # %s, trial # %s, mean pre: %s [1.1]' % (cell, i+1, mean_pre))
                #             trace_dff = [np.nan] * len(trace)
                #             dFstdF = [np.nan] * len(
                #                 trace)  # - commented out to test if we need to exclude cells for this correction with low mean_pre since you're not dividing by a bad mean_pre value
                #         else:
                #             # trace_dff = ((trace - mean_pre) / mean_pre) * 100
                #             trace_dff = Utils.normalize_dff(trace, threshold_val=mean_pre)
                #             # std_pre = np.std(trace[0:expobj.pre_stim], ddof=1)
                #             # # dFstdF = (((trace - mean_pre) / mean_pre) * 100) / std_pre  # make dF divided by std of pre-stim F trace
                #             # dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                #
                #         # # add nan if cell is inside sz boundary for this stim -- temporarily commented out for a while
                #         # if 'post' in expobj.metainfo['exptype']:
                #         #     if hasattr(expobj, 'slmtargets_szboundary_stim'):
                #         #         if expobj.is_cell_insz(cell=cell, stim=stim_timings[i]):
                #         #             trace_dff = [np.nan] * len(trace)
                #         #             dFstdF = [np.nan] * len(trace)
                #         #     else:
                #         #         AttributeError(
                #         #             'no slmtargets_szboundary_stim attr, so classify cells in sz boundary hasnot been saved for this expobj')
                #
                #         flu_dff.append(trace_dff)
                #         flu_dfstdF.append(dFstdF)

                # else:
                #     TypeError('need to specify what to normalize to in get_targets_dFF (choose "baseline" or "pre-stim")')

                dff_traces.append(flu_dff)  # contains all individual dFF traces for all stim times
                dff_traces_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

                dfstdF_traces.append(flu_dfstdF)
                dfstdF_traces_avg.append(np.nanmean(flu_dfstdF, axis=0))

                raw_traces.append(flu_trials)
                raw_traces_avg.append(np.nanmean(flu_trials, axis=0))

        expobj.s2p_nontargets_analysis = [cell for cell in [*stim_cells_analyse] if cell not in expobj.s2p_nontargets_exclude]

        if normalize_to == 'baseline':
            print(
                '\nCompleted collecting pre to post stim traces -- normalized to spont imaging as baseline -- for %s cells' % len(
                    dff_traces_avg))
            expobj.dff_traces_nontargets = dff_traces
            expobj.dff_traces_nontargets_avg = dff_traces_avg
            # return dff_traces_nontargets, dff_traces_nontargets_avg
        elif normalize_to == 'pre-stim' or normalize_to == 'whole-trace':
            expobj.dff_traces_nontargets = np.asarray(dff_traces)
            expobj.dff_traces_nontargets_avg = np.asarray([i for i in dff_traces_avg])
            expobj.dfstdF_traces_nontargets = np.asarray(dfstdF_traces)
            expobj.dfstdF_traces_nontargets_avg = np.asarray([i for i in dfstdF_traces_avg])
            expobj.raw_traces_nontargets = np.asarray(raw_traces)
            expobj.raw_traces_nontargets_avg = np.asarray([i for i in raw_traces_avg])
            print(f'\nCompleted collecting pre to post stim traces -- normalized to pre-stim period or maybe whole-trace -- for '
                  f'{expobj.dff_traces_nontargets.shape[0]} cells, {expobj.dff_traces_nontargets.shape[1]} stims, {expobj.dff_traces_nontargets.shape[2]} frames')

        print('\nFinished collecting peri-stim traces ')

        expobj.save() if save else None

        plot = False
        if plot:
            cells_plot = np.random.randint(0, expobj.dff_traces_nontargets.shape[0], 100)
            fig, ax = plt.subplots(figsize = (4,4))
            for cell in cells_plot:
                ax.plot(expobj.dfstdF_traces_nontargets_avg[cell])
            ax.plot(np.mean(expobj.dfstdF_traces_nontargets_avg[cells_plot], axis=0), lw='2', color='black')
            fig.show()

            for i in cells_plot:
                print(f"cell: {i}, max: {np.max(expobj.dfstdF_traces_nontargets_avg[i])}")

            cell = 267
            for trace in expobj.dfstdF_traces_nontargets[cell]:
                plt.plot(trace)
            plt.plot(np.mean(expobj.dfstdF_traces_nontargets[cell], axis=0), color='black')
            plt.show()

