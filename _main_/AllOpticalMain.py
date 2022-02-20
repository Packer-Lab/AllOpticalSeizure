import re
import glob

import os
import sys


sys.path.append('/home/pshah/Documents/code/')
from Vape.utils import STAMovieMaker_noGUI as STAMM
import scipy.stats as stats
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import tifffile as tf
from funcsforprajay import funcs as pj
from funcsforprajay.wrappers import plot_piping_decorator
from _utils_.paq_utils import paq_read

import _alloptical_utils as Utils
from _main_.TwoPhotonImagingMain import TwoPhotonImaging



class alloptical(TwoPhotonImaging):

    def __init__(self, paths, metainfo, stimtype, quick=False):
        # self.metainfo = metainfo
        self.stim_type = stimtype
        self.naparm_path = paths['naparms_loc']
        assert os.path.exists(self.naparm_path)

        self.seizure_frames = []

        # set initial attr's
        self.stimProcessing(stim_channel='markpoints2packio')
        self.paqProcessing(lfp=True)
        self._findTargetsAreas()
        self.find_photostim_frames()

        self.pre_stim = int(1.0 * self.fps)  # length of pre stim trace collected (in frames)
        self.post_stim = int(3.0 * self.fps)  # length of post stim trace collected (in frames)
        self.pre_stim_response_window_msec = 500  # msec
        self.pre_stim_response_frames_window = int(
            self.fps * self.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)
        self.post_stim_response_window_msec = 500  # msec
        self.post_stim_response_frames_window = int(
            self.fps * self.post_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)

        #### initializing data processing, data analysis and/or results associated attr's

        ## PHOTOSTIM SLM TARGETS
        self.dFF_SLMTargets = None  # dFF normalized whole traces from SLM targets
        self.raw_SLMTargets = None  # raw whole traces from SLM targets from registered tiffs

        self.responses_SLMtargets_dfprestimf = None  # dF/prestimF responses for all SLM targets for each photostim trial
        self.responses_SLMtargets_tracedFF = None  # delta(poststim dFF and prestim dFF) responses for all SLM targets for each photostim trial - from trace dFF processed trials
        self.StimSuccessRate_SLMtargets_dfprestimf = None
        self.StimSuccessRate_SLMtargets_dfstdf = None
        self.StimSuccessRate_SLMtargets_tracedFF = None
        self.hits_SLMtargets_tracedFF = None
        self.hits_SLMtargets_dfprestimf = None
        self.hits_SLMtargets_dfstdf = None

        from _utils_._anndata import AnnotatedData2
        self.slmtargets_data: AnnotatedData2 = None   # anndata object of # targets vs. # stims - photostim responses - layers are used to store different processing of data


        # .get_alltargets_stim_traces_norm(pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, stims=expobj.stim_start_frames)
        # - various attrs. for collecting photostim timed trace snippets from raw Flu values
        self.SLMTargets_stims_dff = None
        self.SLMTargets_stims_dffAvg = None
        self.SLMTargets_stims_dfstdF = None
        self.SLMTargets_stims_dfstdF_avg = None
        self.SLMTargets_stims_raw = None
        self.SLMTargets_stims_rawAvg = None

        self.SLMTargets_tracedFF_stims_dff = None
        self.SLMTargets_tracedFF_stims_dffAvg = None
        self.SLMTargets_tracedFF_stims_dfstdF = None
        self.SLMTargets_tracedFF_stims_dfstdF_avg = None
        self.SLMTargets_tracedFF_stims_raw = None
        self.SLMTargets_tracedFF_stims_rawAvg = None

        ## breaking down success and failure stims
        self.traces_SLMtargets_successes_avg_dfstdf = None  # trace snippets for only successful stims - normalized by dfstdf
        self.traces_SLMtargets_successes_avg_dfprestimf = None  # trace snippets for only successful stims - normalized by dfprestimf
        self.traces_SLMtargets_tracedFF_successes_avg = None  # trace snippets for only successful stims - delta(trace_dff)
        self.traces_SLMtargets_failures_avg_dfstdf = None  # trace snippets for only failure stims - normalized by dfstdf
        self.traces_SLMtargets_failures_avg_dfprestimf = None  # trace snippets for only failure stims - normalized by dfstdf
        self.traces_SLMtargets_tracedFF_failures_avg = None  # trace snippets for only successful stims - delta(trace_dff)


        ## NON PHOTOSTIM SLM TARGETS
        # TODO add attr's related to non targets cells
        self.dff_traces: np.ndarray
        self.dff_traces_avg: np.ndarray
        self.dfstdF_traces: np.ndarray
        self.dfstdF_traces_avg: np.ndarray
        self.raw_traces: np.ndarray
        self.raw_traces_avg: np.ndarray

        # put object through 2p imaging processing workflow
        TwoPhotonImaging.__init__(self, tiff_path=paths['tiffs_loc'], paq_path=paths['paqs_loc'], metainfo=metainfo, analysis_save_path=paths['analysis_save_path'],
                                  suite2p_path=None, suite2p_run=False, quick=quick)

        # self.tiff_path_dir = paths[0]
        # self.tiff_path = paths[1]

        # self._parsePVMetadata()

        ## CREATE THE APPROPRIATE ANALYSIS SUBFOLDER TO USE FOR SAVING ANALYSIS RESULTS TO

        print('\ninitialized alloptical expobj of exptype and trial: \n', self.metainfo)



        self.save()

    def __repr__(self):
        lastmod = time.ctime(os.path.getmtime(self.pkl_path))
        if not hasattr(self, 'metainfo'):
            information = f"uninitialized"
        else:
            prep = self.metainfo['animal prep.']
            trial = self.metainfo['trial']
            information = f"{prep} {trial}"

        return repr(f"({information}) TwoPhotonImaging.alloptical experimental data object, last saved: {lastmod}")

    @property
    def slmtargets_ids(self):
        return list(range(len(self.target_coords_all)))

    @property
    def stims_idx(self):
        return list(range(len(self.stim_start_frames)))

    def collect_traces_from_targets(self, force_redo: bool = False, save: bool = True):

        if force_redo:
            continu = True
        elif hasattr(self, 'raw_SLMTargets'):
            print('skipped re-collecting of raw traces from SLM targets')
            continu = False
        else:
            continu = True

        if continu:
            print('\n\ncollecting raw Flu traces from SLM target coord. areas from registered TIFFs')
            # read in registered tiff
            reg_tif_folder = self.s2p_path + '/reg_tif/'
            reg_tif_list = os.listdir(reg_tif_folder)
            reg_tif_list.sort()
            start = self.curr_trial_frames[0] // 2000  # 2000 because that is the batch size for suite2p run
            end = self.curr_trial_frames[1] // 2000 + 1

            mean_img_stack = np.zeros([end - start, self.frame_x, self.frame_y])
            # collect mean traces from target areas of each target coordinate by reading in individual registered tiffs that contain frames for current trial
            targets_trace_full = np.zeros([len(self.slmtargets_ids), (end - start) * 2000], dtype='float32')
            counter = 0
            for i in range(start, end):
                tif_path_save2 = self.s2p_path + '/reg_tif/' + reg_tif_list[i]
                with tf.TiffFile(tif_path_save2, multifile=False) as input_tif:
                    print('|- reading tiff: %s' % tif_path_save2)
                    data = input_tif.asarray()

                targets_trace = np.zeros([len(self.target_coords_all), data.shape[0]], dtype='float32')
                for coord in range(len(self.target_coords_all)):
                    target_areas = np.array(
                        self.target_areas)  # TODO update this so that it doesn't include the extra exclusion zone
                    x = data[:, target_areas[coord, :, 1], target_areas[coord, :, 0]]  # = 1
                    targets_trace[coord] = np.mean(x, axis=1)

                targets_trace_full[:, (i - start) * 2000: ((i - start) * 2000) + data.shape[
                    0]] = targets_trace  # iteratively write to each successive segment of the targets_trace array based on the length of the reg_tiff that is read in.

                mean_img_stack[counter] = np.mean(data, axis=0)
                counter += 1

            # final part, crop to the *exact* frames for current trial
            self.raw_SLMTargets = targets_trace_full[:,
                                  self.curr_trial_frames[0] - start * 2000: self.curr_trial_frames[1] - (start * 2000)]

            self.dFF_SLMTargets = Utils.normalize_dff(self.raw_SLMTargets, threshold_pct=10)

            # targets_trace_dFF_full = normalize_dff(targets_trace_full, threshold_pct=10)
            # self.dFF_SLMTargets = targets_trace_dFF_full[:, self.curr_trial_frames[0] - start * 2000: self.curr_trial_frames[1] - (start * 2000)]

            # # plots to compare dFF normalization for each trace - temp checking for one target
            # target = 0
            # pj.make_general_plot(data_arr=[self.raw_SLMTargets[target], self.dFF_SLMTargets[target][::4]],
            #                      x_range=[range(len(self.raw_SLMTargets[target])),
            #                               range(len(self.dFF_SLMTargets[target]))[::4]],
            #                      figsize=(20, 5), nrows=1, ncols=1,
            #                      suptitle=f"raw trace (blue), dFF trace (norm. to bottom 10 pct) (green)",
            #                      colors=['blue', 'green'], y_labels=['raw values', 'dFF values'])

            self.meanFluImg_registered = np.mean(mean_img_stack, axis=0)

            self.save() if save else None

    def get_alltargets_stim_traces_norm(self, process: str, targets_idx: int = None, subselect_cells: list = None,
                                        pre_stim=15, post_stim=200, filter_sz: bool = False, stims: list = None):  # TODO remove the sz filter - this is all more practicably done in other locations of the process
        """
        primary function to measure the dFF and dF/setdF trace SNIPPETS for photostimulated targets.
        :param stims:
        :param targets_idx: integer for the index of target cell to process
        :param subselect_cells: ls of cells to subset from the overall set of traces (use in place of targets_idx if desired)
        :param pre_stim: number of frames to use as pre-stim
        :param post_stim: number of frames to use as post-stim
        :param filter_sz: whether to filter out stims that are occuring seizures
        :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
        """

        print(
            f"\n---------- Collecting {process} stim trace snippets of SLM targets for {self.metainfo['exptype']} {self.metainfo['animal prep.']} {self.metainfo['trial']} [1.] ---------- ")

        if filter_sz:
            print('|-filter_sz active')
            if hasattr(self, 'slmtargets_szboundary_stim') and self.slmtargets_szboundary_stim is not None:
                pass
            else:
                print('|- WARNING: classifying of sz boundaries not completed for this expobj, not collecting any stim trace snippets', self.metainfo['animal prep.'], self.metainfo['trial'])

        if stims is None:
            stim_timings = self.stim_start_frames
        else:
            stim_timings = stims

        if process == 'trace raw':  ## specify which data to process (i.e. do you want to process whole trace dFF traces?)
            data_to_process = self.raw_SLMTargets
        elif process == 'trace dFF':
            if not hasattr(self, 'dFF_SLMTargets'):
                self.collect_traces_from_targets(force_redo=True)
            data_to_process = self.dFF_SLMTargets
        else:
            raise ValueError('need to provide `process` as either `trace raw` or `trace dFF`')

        if subselect_cells:
            num_targets = len(data_to_process[subselect_cells])
            targets_trace = data_to_process[subselect_cells]
        else:
            num_targets = len(self.slmtargets_ids)
            targets_trace = data_to_process

        # collect photostim timed average dff traces of photostim targets
        targets_dff = np.zeros(
            [num_targets, len(self.stim_start_frames), pre_stim + self.stim_duration_frames + post_stim])
        # SLMTargets_stims_dffAvg = np.zeros([num_targets, pre_stim_sec + post_stim_sec])

        targets_dfstdF = np.zeros(
            [num_targets, len(self.stim_start_frames), pre_stim + self.stim_duration_frames + post_stim])
        # targets_dfstdF_avg = np.zeros([num_targets, pre_stim_sec + post_stim_sec])

        targets_raw = np.zeros(
            [num_targets, len(self.stim_start_frames), pre_stim + self.stim_duration_frames + post_stim])
        # targets_raw_avg = np.zeros([num_targets, pre_stim_sec + post_stim_sec])

        if targets_idx is not None:
            print('collecting stim traces for cell ', targets_idx + 1)
            if filter_sz:
                flu = [targets_trace[targets_idx][stim - pre_stim: stim + self.stim_duration_frames + post_stim] for
                       stim in
                       stim_timings if
                       stim not in self.seizure_frames]
            else:
                flu = [targets_trace[targets_idx][stim - pre_stim: stim + self.stim_duration_frames + post_stim] for
                       stim in
                       stim_timings]
            # flu_dfstdF = []
            # flu_dff = []
            for i in range(len(flu)):
                trace = flu[i]
                mean_pre = np.mean(trace[0:pre_stim])
                if process == 'trace raw':
                    trace_dff = ((trace - mean_pre) / mean_pre) * 100
                elif process == 'trace dFF':
                    trace_dff = (trace - mean_pre)
                else:
                    ValueError('not sure how to calculate peri-stim traces...')
                std_pre = np.std(trace[0:pre_stim])
                dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                targets_raw[targets_idx, i] = trace
                targets_dff[targets_idx, i] = trace_dff
                targets_dfstdF[targets_idx, i] = dFstdF
            print(f"shape of targets_dff[targets_idx]: {targets_dff[targets_idx].shape}")
            return targets_raw[targets_idx], targets_dff[targets_idx], targets_dfstdF[targets_idx]

        else:
            for cell_idx in range(num_targets):
                print('\- collecting stim traces for cell %s' % subselect_cells[cell_idx]) if subselect_cells else None

                if filter_sz:
                    if hasattr(self, 'slmtargets_szboundary_stim') and self.slmtargets_szboundary_stim is not None:
                        flu = []
                        for stim in stim_timings:
                            if stim in self.slmtargets_szboundary_stim.keys():  # some stims dont have sz boundaries because of issues with their TIFFs not being made properly (not readable in Fiji), usually it is the first TIFF in a seizure
                                if cell_idx not in self.slmtargets_szboundary_stim[stim]:
                                    flu.append(targets_trace[cell_idx][
                                               stim - pre_stim: stim + self.stim_duration_frames + post_stim])
                    else:
                        flu = []
                        # print('classifying of sz boundaries not completed for this expobj, not collecting any stim trace snippets', self.metainfo['animal prep.'], self.metainfo['trial'])
                    # flu = [targets_trace[cell_idx][stim - pre_stim_sec: stim + self.stim_duration_frames + post_stim_sec] for
                    #        stim
                    #        in stim_timings if
                    #        stim not in self.seizure_frames]
                else:
                    flu = [targets_trace[cell_idx][stim - pre_stim: stim + self.stim_duration_frames + post_stim] for
                           stim in stim_timings]

                # flu_dfstdF = []
                # flu_dff = []
                # flu = []
                if len(flu) > 0:
                    for i in range(len(flu)):
                        trace = flu[i]
                        mean_pre = np.mean(trace[0:pre_stim])
                        if process == 'trace raw':
                            trace_dff = ((trace - mean_pre) / mean_pre) * 100  # values of trace_dff are %dF/prestimF compared to raw
                        elif process == 'trace dFF':
                            trace_dff = (trace - mean_pre)  # don't need to do mean normalization if process traces that are already dFF normalized
                        else:
                            ValueError('need to provide `process` as either `trace raw` or `trace dFF`')
                        std_pre = np.std(trace[0:pre_stim])
                        dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                        targets_raw[cell_idx, i] = trace
                        targets_dff[cell_idx, i] = trace_dff
                        targets_dfstdF[cell_idx, i] = dFstdF
                        # flu_dfstdF.append(dFstdF)
                        # flu_dff.append(trace_dff)

                # targets_dff.append(flu_dff)  # contains all individual dFF traces for all stim times
                # SLMTargets_stims_dffAvg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

                # targets_dfstdF.append(flu_dfstdF)
                # targets_dfstdF_avg.append(np.nanmean(flu_dfstdF, axis=0))

                # SLMTargets_stims_raw.append(flu)
                # targets_raw_avg.append(np.nanmean(flu, axis=0))

            targets_dff_avg = np.mean(targets_dff, axis=1)
            targets_dfstdF_avg = np.mean(targets_dfstdF, axis=1)
            targets_raw_avg = np.mean(targets_raw, axis=1)

            # ## plotting trace snippets for targets_dff to check data processing quality
            # pj.make_general_plot(data_arr=targets_dff_avg, ncols=1, nrows=1, figsize=(6,6), suptitle=f"Avg photostim response ({process}): {targets_dff_avg.shape[0]} targets from {self.metainfo['exptype']} {self.metainfo['animal prep.']} {self.metainfo['trial']}")

            print(f"|- returning targets stims array of shape: {targets_dff.shape[0]} targets, {targets_dff.shape[1]} stims, {targets_dff.shape[2]} frames")
            return targets_dff, targets_dff_avg, targets_dfstdF, targets_dfstdF_avg, targets_raw, targets_raw_avg

    def _parseNAPARMxml(self):

        print('\n-----parsing Naparm xml file...')

        print('loading NAPARM_xml_path:')
        NAPARM_xml_path = pj.path_finder(self.naparm_path, '.xml')[0]

        xml_tree = ET.parse(NAPARM_xml_path)
        root = xml_tree.getroot()

        title = root.get('Name')
        n_trials = int(root.get('Iterations'))

        inter_point_delay = 0
        spiral_duration = 20
        for elem in root:
            if int(elem[0].get('InterPointDelay')) > 0:
                inter_point_delay = int(elem[0].get('InterPointDelay'))
                spiral_duration = int(elem[0].get('Duration'))

        n_groups, n_reps, n_shots = [int(s) for s in re.findall(r'\d+', title)]

        print('Numbers of trials:', n_trials, '\nNumber of groups:', n_groups, '\nNumber of shots:', n_shots,
              '\nNumber of sequence reps:', n_reps, '\nInter-point delay:', inter_point_delay,
              '\nSpiral Duration (ms):', spiral_duration)

        # repetitions = int(root[1].get('Repetitions'))
        # print('Repetitions:', repetitions)

        self.n_groups = n_groups
        self.n_reps = n_reps
        self.n_shots = n_shots
        self.n_trials = n_trials
        self.inter_point_delay = inter_point_delay
        self.single_stim_dur = spiral_duration

    def _parseNAPARMgpl(self):

        print('\n-----parsing Naparm gpl file...')

        NAPARM_gpl_path = pj.path_finder(self.naparm_path, '.gpl')[0]
        print('loading NAPARM_gpl_path: ', NAPARM_gpl_path)

        xml_tree = ET.parse(NAPARM_gpl_path)
        root = xml_tree.getroot()

        for elem in root:
            if elem.get('Duration'):
                single_stim_dur = float(elem.get('Duration'))
                spiral_size = float(elem.get('SpiralSize'))
                print('Single stim dur (ms):', elem.get('Duration'))
                break

        for elem in root:
            if elem.get('SpiralSize'):
                spiral_size = float(elem.get('SpiralSize'))
                spiral_size = (spiral_size + 0.005155) / 0.005269  # hard-coded size of spiral from MATLAB code
                print('Spiral size .gpl file:', elem.get('SpiralSize'))
                print('spiral size (um):', int(spiral_size))
                break

        self.spiral_size = np.ceil(spiral_size)
        # self.single_stim_dur = single_stim_dur  # not sure why this was previously getting this value from here, but I'm now getting it from the xml file above

    def paqProcessing(self, **kwargs):

        print('\n-----processing paq file...')

        print('loading', self.paq_path)

        paq, _ = paq_read(self.paq_path, plot=True)
        self.paq_rate = paq['rate']

        # find frame times

        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = pj.threshold_detect(clock_voltage, 1)
        self.frame_clock = frame_clock
        plt.figure(figsize=(10, 5))
        plt.plot(clock_voltage)
        plt.plot(frame_clock, np.ones(len(frame_clock)), '.')
        plt.suptitle('frame clock from paq, with detected frame clock instances as scatter')
        sns.despine()
        plt.show()

        # find start and stop frame_clock times -- there might be multiple 2p imaging starts/stops in the paq trial (hence multiple frame start and end times)
        self.frame_start_times = [self.frame_clock[0]]  # initialize ls
        self.frame_end_times = []
        i = len(self.frame_start_times)
        for idx in range(1, len(self.frame_clock) - 1):
            if (self.frame_clock[idx + 1] - self.frame_clock[idx]) > 2e3:
                i += 1
                self.frame_end_times.append(self.frame_clock[idx])
                self.frame_start_times.append(self.frame_clock[idx + 1])
        self.frame_end_times.append(self.frame_clock[-1])

        # for frame in self.frame_clock[1:]:
        #     if (frame - self.frame_start_times[i - 1]) > 2e3:
        #         i += 1
        #         self.frame_start_times.append(frame)
        #         self.frame_end_times.append(self.frame_clock[np.where(self.frame_clock == frame)[0] - 1][0])
        # self.frame_end_times.append(self.frame_clock[-1])

        # handling cases where 2p imaging clock has been started/stopped >1 in the paq trial
        if len(self.frame_start_times) > 1:
            diff = [self.frame_end_times[idx] - self.frame_start_times[idx] for idx in
                    range(len(self.frame_start_times))]
            idx = diff.index(max(diff))
            self.frame_start_time_actual = self.frame_start_times[idx]
            self.frame_end_time_actual = self.frame_end_times[idx]
            self.frame_clock_actual = [frame for frame in self.frame_clock if
                                       self.frame_start_time_actual <= frame <= self.frame_end_time_actual]
        else:
            self.frame_start_time_actual = self.frame_start_times[0]
            self.frame_end_time_actual = self.frame_end_times[0]
            self.frame_clock_actual = self.frame_clock

        # find stim times
        stim_idx = paq['chan_names'].index(self.stim_channel)
        stim_volts = paq['data'][stim_idx, :]
        stim_times = pj.threshold_detect(stim_volts, 1)
        # self.stim_times = stim_times
        self.stim_start_times = stim_times
        print('# of stims found on %s: %s' % (self.stim_channel, len(self.stim_start_times)))

        # correct this based on txt file
        duration_ms = self.stim_dur
        frame_rate = self.fps / self.n_planes
        duration_frames = np.ceil((duration_ms / 1000) * frame_rate)
        self.stim_duration_frames = int(duration_frames)

        plt.figure(figsize=(10, 5))
        plt.plot(stim_volts)
        plt.plot(stim_times, np.ones(len(stim_times)), '.')
        plt.suptitle('stim times')
        sns.despine()
        plt.show()

        # find stim frames

        self.stim_start_frames = []

        for plane in range(self.n_planes):

            stim_start_frames = []

            for stim in stim_times:
                # the index of the frame immediately preceeding stim
                stim_start_frame = next(
                    i - 1 for i, sample in enumerate(frame_clock[plane::self.n_planes]) if sample - stim >= 0)
                stim_start_frames.append(stim_start_frame)

            self.stim_start_frames = stim_start_frames
            # self.stim_start_frames.append(np.array(stim_start_frames))  # recoded with slight improvement

            # # sanity check
            # assert max(self.stim_start_frames[0]) < self.raw[plane].shape[1] * self.n_planes

        if 'lfp' in kwargs.keys():
            if kwargs['lfp']:
                # find voltage (LFP recording signal) channel and save as lfp_signal attribute
                voltage_idx = paq['chan_names'].index('voltage')
                self.lfp_signal = paq['data'][voltage_idx]

    def photostimProcessing(self):

        '''
        remember that this is currently configured for only the interleaved photostim, not the really fast photostim of multi-groups
        '''

        self._parseNAPARMxml()
        self._parseNAPARMgpl()

        #         single_stim = self.single_stim_dur * self.n_shots
        #         total_single_stim = single_stim + self.inter_point_delay

        #         #self.stim_dur = total_single_stim - self.inter_point_delay

        #         total_multi_stim = total_single_stim * self.n_groups

        #         total_stim = total_multi_stim * self.n_reps

        #         self.stim_dur = total_stim - self.inter_point_delay

        ## PRAJAY EDITS HERE FOR PHOTOSTIMPROCESSING:
        # calculate duration (ms) of each individual trial (for a multi group stimulation trial)
        single_stim = self.single_stim_dur + self.inter_point_delay
        total_single_stim = single_stim * self.n_shots * self.n_groups * self.n_reps

        self.stim_dur = total_single_stim
        print('Single stim. Duration (ms): ', self.single_stim_dur)
        print('Total stim. Duration per trial (ms): ', self.stim_dur)


    def stimProcessing(self, stim_channel):

        self.stim_channel = stim_channel

        if self.stim_type == '2pstim':
            self.photostimProcessing()

    def cellStaProcessing(self, test='t_test'):

        if self.stim_start_frames:

            # this is the key parameter for the sta, how many frames before and after the stim onset do you want to use
            self.pre_frames = int(np.ceil(self.fps * 0.5))  # 500 ms pre-stim period
            self.post_frames = int(np.ceil(self.fps * 3))  # 3000 ms post-stim period

            # ls of cell pixel intensity values during each stim on each trial
            self.all_trials = []  # ls 1 = cells, ls 2 = trials, ls 3 = dff vector

            # the average of every trial
            self.stas = []  # ls 1 = cells, ls 2 = sta vector

            self.all_amplitudes = []
            self.sta_amplitudes = []

            self.t_tests = []
            self.wilcoxons = []

            for plane in range(self.n_planes):

                all_trials = []  # ls 1 = cells, ls 2 = trials, ls 3 = dff vector

                stas = []  # ls 1 = cells, ls 2 = sta vector

                all_amplitudes = []
                sta_amplitudes = []

                t_tests = []
                wilcoxons = []

                # loop through each cell
                for i, unit in enumerate(self.raw[plane]):

                    trials = []
                    amplitudes = []
                    df = []

                    # a flat ls of all observations before stim occured
                    pre_obs = []
                    # a flat ls of all observations after stim occured
                    post_obs = []

                    for stim in self.stim_start_frames[plane]:
                        # get baseline values from pre_stim_sec
                        pre_stim_f = unit[stim - self.pre_frames: stim]
                        baseline = np.mean(pre_stim_f)

                        # the whole trial and dfof using baseline
                        trial = unit[stim - self.pre_frames: stim + self.post_frames]
                        trial = [((f - baseline) / baseline) * 100 for f in trial]  # dff calc
                        trials.append(trial)

                        # calc amplitude of response
                        pre_f = trial[: self.pre_frames - 1]
                        pre_f = np.mean(pre_f)

                        avg_post_start = self.pre_frames + (self.stim_duration_frames + 1)
                        avg_post_end = avg_post_start + int(np.ceil(self.fps * 0.5))  # post-stim period of 500 ms

                        post_f = trial[avg_post_start: avg_post_end]
                        post_f = np.mean(post_f)
                        amplitude = post_f - pre_f
                        amplitudes.append(amplitude)

                        # append to flat lists
                        pre_obs.append(pre_f)
                        post_obs.append(post_f)

                    trials = np.array(trials)
                    all_trials.append(trials)

                    # average amplitudes across trials
                    amplitudes = np.array(amplitudes)
                    all_amplitudes.append(amplitudes)
                    sta_amplitude = np.mean(amplitudes, 0)
                    sta_amplitudes.append(sta_amplitude)

                    # average across all trials
                    sta = np.mean(trials, 0)
                    stas.append(sta)

                    # remove nans from flat lists
                    pre_obs = [x for x in pre_obs if ~np.isnan(x)]
                    post_obs = [x for x in post_obs if ~np.isnan(x)]

                    # t_test and man whit test pre and post stim (any other test could also be used here)
                    t_test = stats.ttest_rel(pre_obs, post_obs)
                    t_tests.append(t_test)

                    wilcoxon = stats.wilcoxon(pre_obs, post_obs)
                    wilcoxons.append(wilcoxon)

                self.all_trials.append(np.array(all_trials))
                self.stas.append(np.array(stas))

                self.all_amplitudes.append(np.array(all_amplitudes))
                self.sta_amplitudes.append(np.array(sta_amplitudes))

                self.t_tests.append(np.array(t_tests))
                self.wilcoxons.append(np.array(wilcoxons))

            plt.figure()
            plt.plot([avg_post_start] * 2, [-1000, 1000])
            plt.plot([avg_post_end] * 2, [-1000, 1000])
            plt.plot([self.pre_frames - 1] * 2, [-1000, 1000])
            plt.plot([0] * 2, [-1000, 1000])
            plt.plot(stas[5])
            plt.plot(stas[10])
            plt.plot(stas[15])
            plt.ylim([-100, 200])

            self.staSignificance(test)
            self.singleTrialSignificance()

    def staSignificance(self, test):

        self.sta_sig = []

        for plane in range(self.n_planes):

            # set this to true if you want to multiple comparisons correct for the number of cells
            multi_comp_correction = True
            if not multi_comp_correction:
                divisor = 1
            else:
                divisor = self.n_units[plane]

            if test == 't_test':
                p_vals = [t[1] for t in self.t_tests[plane]]
            if test == 'wilcoxon':
                p_vals = [t[1] for t in self.wilcoxons[plane]]

            if multi_comp_correction:
                print('performing t-test on cells with mutliple comparisons correction')
            else:
                print('performing t-test on cells without mutliple comparisons correction')

            sig_units = []

            for i, p in enumerate(p_vals):
                if p < (0.05 / divisor):
                    unit_index = self.cell_id[plane][i]
                    # print('stimulation has significantly changed fluoresence of s2p unit {}, its P value is {}'.format(unit_index, p))
                    sig_units.append(unit_index)  # significant units

            self.sta_sig.append(sig_units)

    def singleTrialSignificance(self):

        self.single_sig = []  # single trial significance value for each trial for each cell in each plane

        for plane in range(self.n_planes):

            single_sigs = []

            for cell, _ in enumerate(self.cell_id[plane]):

                single_sig = []

                for trial in range(self.n_trials):

                    pre_f_trial = self.all_trials[plane][cell][trial][: self.pre_frames]
                    std = np.std(pre_f_trial)

                    if np.absolute(self.all_amplitudes[plane][cell][trial]) >= 2 * std:
                        single_sig.append(True)
                    else:
                        single_sig.append(False)

                single_sigs.append(single_sig)

            self.single_sig.append(single_sigs)

    def _findTargetsAreas(self):

        '''
        Finds cells that have been targeted for optogenetic photostimulation using Naparm in all-optical type experiments.
        output: coordinates of targets, and circular areas of targets
        Note this is not done by target groups however. So all of the targets are just in one big ls.
        '''

        print('\n-----Loading up target coordinates...')

        self.n_targets = []
        self.target_coords = []
        self.target_areas = []

        # load naparm targets file for this experiment
        naparm_path = os.path.join(self.naparm_path, 'Targets')

        listdir = os.listdir(naparm_path)

        scale_factor = self.frame_x / 512

        ## All SLM targets
        for path in listdir:
            if 'AllFOVTargets' in path:
                target_file = path
        target_image = tf.imread(os.path.join(naparm_path, target_file))

        # # SLM group#1: FOVTargets_001
        # for path in listdir:
        #     if 'FOVTargets_001' in path:
        #         target_file_1 = path
        # target_image_slm_1 = tf.imread(os.path.join(naparm_path, target_file_1))
        #
        # # SLM group#2: FOVTargets_002
        # for path in listdir:
        #     if 'FOVTargets_002' in path:
        #         target_file_2 = path
        # target_image_slm_2 = tf.imread(os.path.join(naparm_path, target_file_2))

        ## idk if there a reason for this...but just keeping it here since Rob had used it for this analysis
        # n = np.array([[0, 0], [0, 1]])
        # target_image_scaled = np.kron(target_image, n)

        # target_image_scaled_1 = target_image_slm_1;
        # del target_image_slm_1
        # target_image_scaled_2 = target_image_slm_2;
        # del target_image_slm_2

        # use frame_x and frame_y to get bounding box of OBFOV inside the BAFOV, assume BAFOV always 1024x1024
        frame_x = self.frame_x
        frame_y = self.frame_y

        # if frame_x < 1024 or frame_y < 1024:
        #     pass
        # #             # bounding box coords
        # #             x1 = 511 - frame_x / 2
        # #             x2 = 511 + frame_x / 2
        # #             y1 = 511 - frame_y / 2
        # #             y2 = 511 + frame_y / 2
        #
        # #             # calc imaging galvo offset between BAFOV and t-series
        # #             zoom = self.zoom
        # #             scan_x = self.scan_x  # scan centre in V
        # #             scan_y = self.scan_y
        #
        # #             ScanAmp_X = 2.62 * 2
        # #             ScanAmp_Y = 2.84 * 2
        #
        # #             ScanAmp_V_FOV_X = ScanAmp_X / zoom
        # #             ScanAmp_V_FOV_Y = ScanAmp_Y / zoom
        #
        # #             scan_pix_y = ScanAmp_V_FOV_Y / 1024
        # #             scan_pix_x = ScanAmp_V_FOV_X / 1024
        #
        # #             offset_x = scan_x / scan_pix_x  # offset between image centres in pixels
        # #             offset_y = scan_y / scan_pix_y
        #
        # #             # offset the bounding box
        # #             x1, x2, y1, y2 = round(x1 + offset_x), round(x2 + offset_x), round(y1 - offset_y), round(y2 - offset_y)
        #
        # #             # crop the target image using the offset bounding box to get the targets in t-series imaging space
        # #             target_image_scaled = target_image_scaled[y1:y2, x1:x2]
        # #             tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        # else:
        #     #             image_8bit = pj.convert_to_8bit(target_image_scaled, np.unit8)
        #     #             tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), image_8bit)
        #     tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)

        targets = np.where(target_image > 0)
        # targets_1 = np.where(target_image_scaled_1 > 0)
        # targets_2 = np.where(target_image_scaled_2 > 0)

        targetCoordinates = list(zip(targets[1] * scale_factor, targets[0] * scale_factor))
        print('Number of targets:', len(targetCoordinates))

        # targetCoordinates_1 = ls(zip(targets_1[1], targets_1[0]))
        # print('Number of targets, SLM group #1:', len(targetCoordinates_1))
        #
        # targetCoordinates_2 = ls(zip(targets_2[1], targets_2[0]))
        # print('Number of targets, SLM group #2:', len(targetCoordinates_2))

        self.target_coords_all = targetCoordinates
        # self.target_coords_1 = targetCoordinates_1
        # self.target_coords_2 = targetCoordinates_2
        self.n_targets_total = len(targetCoordinates)

        ## specifying target areas in pixels to use for measuring responses of SLM targets
        radius_px = int(np.ceil(((self.spiral_size / 2) + 0) / self.pix_sz_x))
        print(f"spiral size: {self.spiral_size}um")
        print(f"pix sz x: {self.pix_sz_x}um")
        print("radius (in pixels): {:.2f}px".format(radius_px * self.pix_sz_x))

        target_areas = []
        for coord in targetCoordinates:
            target_area = ([item for item in pj.points_in_circle_np(radius_px, x0=coord[0], y0=coord[1])])
            target_areas.append(target_area)
        self.target_areas = target_areas

        ## target_areas that need to be excluded when filtering for nontarget cells
        radius_px = int(np.ceil(((self.spiral_size / 2) + 2.5) / self.pix_sz_x))
        print("radius of target exclusion zone (in pixels): {:.2f}px".format(radius_px * self.pix_sz_x))

        target_areas = []
        for coord in targetCoordinates:
            target_area = ([item for item in pj.points_in_circle_np(radius_px, x0=coord[0], y0=coord[1])])
            target_areas.append(target_area)
        self.target_areas_exclude = target_areas

        # # get areas for SLM group #1
        # target_areas_1 = []
        # for coord in targetCoordinates_1:
        #     target_area = ([item for item in pj.points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
        #     target_areas_1.append(target_area)
        # self.target_areas_1 = target_areas_1
        #
        # # get areas for SLM group #2
        # target_areas_2 = []
        # for coord in targetCoordinates_2:
        #     target_area = ([item for item in pj.points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
        #     target_areas_2.append(target_area)
        # self.target_areas_2 = target_areas_2

        # find targets by stim groups
        target_files = []
        for path in listdir:
            if 'FOVTargets_00' in path:
                target_files.append(path)

        self.target_coords = []
        counter = 0
        for slmgroup in target_files:
            target_image = tf.imread(os.path.join(naparm_path, slmgroup))
            targets = np.where(target_image > 0)
            targetCoordinates = list(zip(targets[1] * scale_factor, targets[0] * scale_factor))
            self.target_coords.append(targetCoordinates)
            print('Number of targets (in SLM group %s): ' % (counter + 1), len(targetCoordinates))
            counter += 1

        print('FIN. -----Loading up target coordinates-----')

    def _findTargetedS2pROIs(self, force_redo: bool = False, plot: bool = True):
        '''finding s2p cell ROIs that were also SLM targets (or more specifically within the target areas as specified by _findTargetAreas - include 15um radius from center coordinate of spiral)
        --- LAST UPDATED NOV 6 2021 - copied over from Rob's ---
        '''

        if force_redo:
            continu = True
        elif hasattr(self, 's2p_cell_targets'):
            print('skipped re-running of finding s2p targets from suite2p cell ls')
            continu = False
        else:
            continu = True

        if continu:
            '''
            Make a binary mask of the targets and multiply by an image of the cells
            to find cells that were targeted

            --- COPIED FROM ROB'S VAPE INTERAREAL_ANALYSIS.PY ON NOV 5 2021 ---
            '''

            print('searching for targeted cells in suite2p results... [Rob version]')
            ##### IDENTIFYING S2P ROIs THAT ARE WITHIN THE SLM TARGET SPIRAL AREAS
            # make all target area coords in to a binary mask
            targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
            target_areas = np.array(self.target_areas)
            targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1

            # make an image of every cell area, filled with the index of that cell
            cell_img = np.zeros_like(targ_img)

            cell_y = np.array(self.cell_x)
            cell_x = np.array(self.cell_y)

            for i, coord in enumerate(zip(cell_x, cell_y)):
                cell_img[coord] = i + 1

            # binary mask x cell image to get the cells that overlap with target areas
            targ_cell = cell_img * targ_img

            targ_cell_ids = np.unique(targ_cell)[1:] - 1  # correct the cell id due to zero indexing
            self.targeted_cells = np.zeros([self.n_units], dtype='bool')
            self.targeted_cells[targ_cell_ids] = True
            # self.s2p_cell_targets = [self.cell_id[i] for i, x in enumerate(self.targeted_cells) if x is True]  # get ls of s2p cells that were photostim targetted
            self.s2p_cell_targets = [self.cell_id[i] for i in np.where(self.targeted_cells)[
                0]]  # get ls of s2p cells that were photostim targetted

            self.n_targeted_cells = np.sum(self.targeted_cells)

            print('------- Search completed.')
            self.save()
            print('Number of targeted cells: ', self.n_targeted_cells)

            ##### IDENTIFYING S2P ROIs THAT ARE WITHIN THE EXCLUSION ZONES OF THE SLM TARGETS
            # make all target area coords in to a binary mask
            targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
            target_areas_exclude = np.array(self.target_areas_exclude)
            targ_img[target_areas_exclude[:, :, 1], target_areas_exclude[:, :, 0]] = 1

            # make an image of every cell area, filled with the index of that cell
            cell_img = np.zeros_like(targ_img)

            cell_y = np.array(self.cell_x)
            cell_x = np.array(self.cell_y)

            for i, coord in enumerate(zip(cell_x, cell_y)):
                cell_img[coord] = i + 1

            # binary mask x cell image to get the cells that overlap with target areas
            targ_cell = cell_img * targ_img

            targ_cell_ids = np.unique(targ_cell)[1:] - 1  # correct the cell id due to zero indexing
            self.exclude_cells = np.zeros([self.n_units], dtype='bool')
            self.exclude_cells[targ_cell_ids] = True
            self.s2p_cells_exclude = [self.cell_id[i] for i in np.where(self.exclude_cells)[0]]  # get ls of s2p cells that were photostim targetted

            self.n_exclude_cells = np.sum(self.exclude_cells)

            print('------- Search completed.')
            print(f"Number of exclude cells: {self.n_exclude_cells}")

            # define non targets from suite2p ROIs (exclude cells in the SLM targets exclusion - .s2p_cells_exclude)
            self.s2p_nontargets = [cell for cell in self.good_cells if
                                   cell not in self.s2p_cells_exclude]  ## exclusion of cells that are classified as s2p_cell_targets

            print(f"Number of good, s2p non_targets: {len(self.s2p_nontargets)}")
            self.save()

            if plot:
                from _utils_ import alloptical_plotting_utils as aoplot
                fig, ax = plt.subplots(figsize=[6, 6])
                fig, ax = aoplot.plot_cells_loc(expobj=self, cells=self.s2p_cell_targets, show=False, fig=fig, ax=ax,
                                                show_s2p_targets=True,
                                                title=f"s2p cell targets (red-filled) and target areas (white) {self.metainfo['trial']}/{self.metainfo['animal prep.']}",
                                                invert_y=True)

                targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
                target_areas = np.array(self.target_areas)
                targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1
                ax.imshow(targ_img, cmap='Greys_r', zorder=0)
                # for (x, y) in self.target_coords_all:
                #     ax.scatter(x=x, y=y, edgecolors='white', facecolors='none', linewidths=1.0)
                fig.show()

    def _findTargets_naparm(self):

        '''
        For gathering coordinates and circular areas of targets from naparm
        '''

        self.target_coords = []
        self.target_areas = []
        self.target_coords_all = []

        # load naparm targets file for this experiment
        naparm_path = os.path.join(self.naparm_path, 'Targets')

        listdir = os.listdir(naparm_path)

        targets_path = []
        for path in listdir:
            if 'FOVTargets_0' in path:
                targets_path.append(path)
        targets_path = sorted(targets_path)

        print(targets_path)

        target_groups = []
        scale_factor = self.frame_x / 512
        for target_tiff in targets_path:
            target_image_slm = tf.imread(os.path.join(naparm_path, target_tiff))

            targets = np.where(target_image_slm > 0)

            targetCoordinates = list(zip(targets[0] * scale_factor, targets[1] * scale_factor))
            target_groups.append(targetCoordinates)

        self.target_coords = target_groups
        for group in self.target_coords:
            for coord in group:
                self.target_coords_all.append(coord)

        self.n_targets = len(self.target_coords_all)

        radius = self.spiral_size / self.pix_sz_x

        target_areas = []
        for group in self.target_coords:
            a = []
            for target in group:
                target_area = ([item for item in pj.points_in_circle_np(radius, x0=target[0], y0=target[1])])
                a.append(target_area)
            target_areas.append(a)

        self.target_areas = target_areas

        print('Got targets...')

    def find_s2p_targets_naparm(self):
        '''finding s2p cells that were SLM targets - naparm, lots of different SLM groups version'''

        print('searching for targeted cells...')

        self.s2p_targets_naparm = []
        self.n_targeted_cells = 0
        for slm_group in self.target_coords:
            # print(' ')
            print('\nSLM Group # %s' % self.target_coords.index(slm_group))
            targeted_cells = []
            for cell in range(self.n_units):
                flag = 0
                for x, y in zip(self.cell_x[cell], self.cell_y[cell]):
                    if (y, x) in slm_group:
                        print('Cell # %s' % self.cell_id[cell], ('y%s' % y, 'x%s' % x))
                        flag = 1

                if flag == 1:
                    targeted_cells.append(1)
                else:
                    targeted_cells.append(0)

            s2p_cell_targets = [self.cell_id[i] for i, x in enumerate(targeted_cells) if
                                x == 1]  # get ls of s2p cells that were photostim targetted
            self.s2p_targets_naparm.append(s2p_cell_targets)
            print('Number of targeted cells:', len(s2p_cell_targets))
            self.n_targeted_cells += len(s2p_cell_targets)

        self.targeted_cells_all = [x for y in self.s2p_targets_naparm for x in y]

        print('Search completed.')
        print('Total number of targeted cells: ', self.n_targeted_cells)

    # other usefull functions for all-optical analysis

    def whiten_photostim_frame(self, tiff_path, save_as=''):
        im_stack = tf.imread(tiff_path, key=range(self.n_frames))

        frames_to_whiten = []
        for j in self.stim_start_frames:
            frames_to_whiten.append(j)

        im_stack_1 = im_stack
        a = np.full_like(im_stack_1[0], fill_value=0)
        a[0:100, 0:100] = 5000.
        for frame in frames_to_whiten:
            im_stack_1[frame - 3] = im_stack_1[frame - 3] + a
            im_stack_1[frame - 2] = im_stack_1[frame - 2] + a
            im_stack_1[frame - 1] = im_stack_1[frame - 1] + a

        frames_to_remove = []
        for j in self.stim_start_frames:
            for i in range(0,
                           self.stim_duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
                frames_to_remove.append(j + i)

        im_stack_1 = np.delete(im_stack, frames_to_remove, axis=0)

        tf.imwrite(save_as, im_stack_1, photometric='minisblack')

    def find_photostim_frames(self):
        """finds all photostim frames and saves them into the bad_frames attribute for the exp object"""
        print('\n-----calculating photostimulation frames...')
        print('# of photostim frames calculated per stim. trial: ', self.stim_duration_frames + 1)

        photostim_frames = []
        for j in self.stim_start_frames:
            for i in range(
                    self.stim_duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
                photostim_frames.append(j + i)

        self.photostim_frames = photostim_frames
        # print(photostim_frames)
        print('|-- Original # of frames:', self.n_frames, 'frames')
        print('|-- # of Photostim frames:', len(photostim_frames), 'frames')
        print('|-- Minus photostim. frames total:', self.n_frames - len(photostim_frames), 'frames')
        self.bad_frames = photostim_frames

    def append_bad_frames(self, bad_frames=[]):
        '''appends bad frames given as additional frames (e.g. seizure/CSD frames)'''

        if len(bad_frames) > 0:
            # self.seizure_frames = [i for i in bad_frames]
            self.bad_frames = list(np.unique([bad_frames + self.photostim_frames][0]))
            print('\nlen of bad frames total', len(self.bad_frames))
        else:
            self.seizure_frames = []

        print('len of seizures_frames:', len(self.seizure_frames))
        print('len of photostim_frames:', len(self.photostim_frames))

    def avg_stim_images(self, peri_frames: int = 100, stim_timings: list = None, save_img=False, to_plot=False, verbose=False):
        """
        Outputs (either by saving or plotting, or both) images from raw t-series TIFF for a trial around each individual
        stim timings.

        :param peri_frames:
        :param stim_timings:
        :param save_img:
        :param to_plot:
        :param force_redo:
        :param verbose:
        :return:
        """

        print('making stim images...')
        stim_timings = stim_timings if stim_timings else self.stim_start_frames
        if hasattr(self, 'stim_images'):
            x = [0 for stim in stim_timings if stim not in self.stim_images.keys()]
        else:
            self.stim_images = {}
            x = [0] * len(stim_timings)
        if 0 in x:
            tiffs_loc = '%s/*Ch3.tif' % self.tiff_path_dir
            tiff_path = glob.glob(tiffs_loc)[0]
            print('working on loading up %s tiff from: ' % self.metainfo['trial'], tiff_path)
            im_stack = tf.imread(tiff_path, key=range(self.n_frames))
            print('Processing seizures from experiment tiff (wait for all seizure comparisons to be processed), \n '
                  'total tiff shape: ', im_stack.shape)

        for stim in stim_timings:
            message = '|- stim # %s out of %s' % (stim_timings.index(stim), len(stim_timings))
            print(message, end='\r')
            if stim in self.stim_images.keys():
                avg_sub = self.stim_images[stim]
            else:
                if stim < peri_frames:
                    peri_frames = stim
                im_sub = im_stack[stim - peri_frames: stim + peri_frames]
                avg_sub = np.mean(im_sub, axis=0)
                self.stim_images[stim] = avg_sub

            if save_img:
                # save in a subdirectory under the ANALYSIS folder path from whence t-series TIFF came from
                save_path = self.analysis_save_path + 'avg_stim_images'
                save_path_stim = save_path + f'/{self.t_series_name}_stim-{stim}.tif'
                if os.path.exists(save_path):
                    print("saving stim_img tiff to... %s" % save_path_stim) if verbose else None
                    avg_sub8 = pj.convert_to_8bit(avg_sub, 0, 255)
                    tf.imwrite(save_path_stim,
                               avg_sub8, photometric='minisblack')
                else:
                    print('making new directory for saving images at:', save_path)
                    os.mkdir(save_path)
                    print("saving as... %s" % save_path_stim)
                    avg_sub8 = pj.convert_to_8bit(avg_sub, 0, 255)
                    tf.imwrite(save_path_stim,
                               avg_sub, photometric='minisblack')

            if to_plot:
                plt.imshow(avg_sub, cmap='gray')
                plt.suptitle('avg image from %s frames around stim_start_frame %s' % (peri_frames, stim))
                plt.show()  # just plot for now to make sure that you are doing things correctly so far


    def run_stamm_nogui(self, numDiffStims, startOnStim, everyXStims, preSeconds=0.75, postSeconds=1.25):
        """run STAmoviemaker for the expobj's trial"""
        qnap_path = os.path.expanduser('/home/pshah/mnt/qnap')

        ## data path
        movie_path = self.tiff_path
        sync_path = self.paq_path

        ## stamm save path
        stam_save_path = os.path.join(qnap_path, 'Analysis', self.metainfo['date'], 'STA_Movies',
                                      '%s_%s_%s' % (self.metainfo['date'],
                                                    self.metainfo['animal prep.'],
                                                    self.metainfo['trial']))
        os.makedirs(stam_save_path, exist_ok=True)

        ##
        assert os.path.exists(stam_save_path)

        print('QNAP_path:', qnap_path,
              '\ndata path:', movie_path,
              '\nsync path:', sync_path,
              '\nSTA movie save path:', stam_save_path)

        # define STAmm parameters
        frameRate = int(self.fps)

        arg_dict = {'moviePath': movie_path,  # hard-code this
                    'savePath': stam_save_path,
                    'syncFrameChannel': 'frame_clock',
                    'syncStimChannel': 'packio2markpoints',
                    'syncStartSec': 0,
                    'syncStopSec': 0,
                    'numDiffStims': numDiffStims,
                    'startOnStim': startOnStim,
                    'everyXStims': everyXStims,
                    'preSeconds': preSeconds,
                    'postSeconds': postSeconds,
                    'frameRate': frameRate,
                    'averageImageStart': 0.5,
                    'averageImageStop': 1.5,
                    'methodDF': False,
                    'methodDFF': True,
                    'methodZscore': False,
                    'syncPath': sync_path,
                    'zPlanes': 1,
                    'useStimOrder': False,
                    'stimOrder': [],
                    'useSingleTrials': False,
                    'doThreshold': False,
                    'threshold': 0,
                    'colourByTime': False,
                    'useCorrelationImage': False,
                    'blurHandS': False,
                    'makeMaxImage': True,
                    'makeColourImage': False
                    }

        # run STAmm
        STAMM.STAMovieMaker(arg_dict)

        # show the MaxResponseImage
        img = glob.glob(stam_save_path + '/*MaxResponseImage.tif')[0]
        pj.plot_single_tiff(img)

    # def _good_cells(self, min_radius_pix, max_radius_pix):
    #     '''
    #     This function filters each cell for two criteria. 1) at least 1 flu change greater than 2.5std above mean,
    #     and 2) a minimum cell radius (in pixels) of the given value.
    #
    #     :param min_radius_pix:
    #     :return: a ls of
    #     '''
    #
    #     good_cells = []
    #     for i in range(len(self.cell_id)):
    #         raw = self.raw[i]
    #         raw_ = ls(np.delete(raw, self.photostim_frames, None))
    #         raw_dff = normalize_dff(raw_)
    #         std = np.std(raw_dff, axis=0)
    #
    #         z = []
    #         avg_std = []
    #         for j in np.arange(len(raw_dff), step=4):
    #             avg = np.mean(raw_dff[j:j + 4])
    #             if avg > np.mean(raw_dff) + 2.5 * std:
    #                 z.append(j)
    #                 avg_std.append(avg)
    #
    #         radius = self.radius[i]
    #
    #         if len(z) > 0 and radius > min_radius_pix and radius < max_radius_pix:
    #             good_cells.append(self.cell_id[i])
    #
    #     print('# of good cells found: %s (out of %s ROIs) ' % (len(good_cells), len(self.cell_id)))
    #     self.good_cells = good_cells
    #
    # def _good_cells_jit(self, min_radius_pix, max_radius_pix):
    #     good_cells = []
    #     for i in range(len(self.cell_id)):
    #         radius = self.radius[i]
    #
    #         raw = self.raw[i]
    #         raw_ = ls(np.delete(raw, self.photostim_frames, None))
    #         raw_dff = normalize_dff(raw_)
    #         std = np.std(raw_dff, axis=0)
    #
    #         x = self.sliding_window_std(raw_dff, std)
    #
    #         if len(x) > 0 and radius > min_radius_pix and radius < max_radius_pix:
    #             good_cells.append(self.cell_id[i])
    #     print('# of good cells found: %s (out of %s ROIs)' % (len(good_cells), len(self.cell_id)))
    #     return good_cells

    @staticmethod
    def sliding_window_std(raw_dff, std):
        x = []
        # y = []
        for j in np.arange(len(raw_dff), step=4):
            avg = np.mean(raw_dff[j:j + 4])
            if avg > np.mean(
                    raw_dff) + 2.5 * std:  # if the avg of 4 frames is greater than the threshold then save the result
                x.append(j)
                # y.append(avg)
        return x

    # calculate reliability of photostim responsiveness of all of the targeted cells
    def get_SLMTarget_responses_dff(self, process: str, threshold=10, stims_to_use: list = None):
        """
        calculations of dFF responses to photostimulation of SLM Targets. Includes calculating reliability of slm targets,
        saving success stim locations, and saving stim response magnitudes as pandas dataframe.

        :param threshold: dFF threshold above which a response for a photostim trial is considered a success.
        :param stims_to_use: ls of stims to retrieve photostim trial dFF responses
        :return:
        """
        print(f'\n---------- Calculating {process} stim evoked responses (of SLM targets) [.1] ---------- ')
        if stims_to_use is None:
            stims_to_use = range(len(self.stim_start_frames))
            stims_idx = [self.stim_start_frames.index(stim) for stim in stims_to_use]
        elif stims_to_use:
            stims_idx = [self.stim_start_frames.index(stim) for stim in stims_to_use]
        else:
            AttributeError('no stims set to analyse [1.1]')

        # choose between .SLMTargets_stims_dff, or .SLMTargets_stims_dfstdF and .SLMTargets_stims_tracedFF for data to process
        if process == 'dF/prestimF':
            if hasattr(self, 'SLMTargets_stims_dff'):
                targets_traces = self.SLMTargets_stims_dff
                threshold=10
            else:
                AttributeError('no SLMTargets_stims_dff attr. [1.2]')
        elif process == 'dF/stdF':
            if hasattr(self, 'SLMTargets_stims_dfstdF'):
                targets_traces = self.SLMTargets_stims_dfstdF
                threshold = 0.3
            else:
                AttributeError('no SLMTargets_stims_dff attr. [1.2]')

        elif process == 'delta(trace_dFF)':
            if hasattr(self, 'SLMTargets_tracedFF_stims_dff'):
                if type(self.SLMTargets_tracedFF_stims_dff) == list:
                    self.SLMTargets_tracedFF_stims_dff, self.SLMTargets_tracedFF_stims_dffAvg, self.SLMTargets_tracedFF_stims_dfstdF, \
                    self.SLMTargets_tracedFF_stims_dfstdF_avg, self.SLMTargets_tracedFF_stims_raw, self.SLMTargets_tracedFF_stims_rawAvg = \
                        self.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=self.pre_stim,
                                                               post_stim=self.post_stim, stims=self.stim_start_frames)
                targets_traces = self.SLMTargets_tracedFF_stims_dff
                threshold=10
            else:
                AttributeError('no SLMTargets_tracedFF_stims_dff attr. [1.3]')
        else:
            ValueError('need to assign to process: dF/prestimF or dF/stdF or trace dFF')

        # initializing pandas df that collects responses of stimulations
        if hasattr(self, 'SLMTargets_stims_dff'):
            d = {}
            for stim in stims_idx:
                d[stim] = [None] * targets_traces.shape[0]
            df = pd.DataFrame(d, index=range(targets_traces.shape[0]))  # population dataframe
        else:
            AttributeError('no SLMTargets_stims_dff attr. [1.2]')

        # initializing pandas df for binary showing of success and fails (1= success, 0= fails)
        hits_slmtargets = {}  # to be converted in pandas df below - will contain 1 for every success stim, 0 for non success stims
        for stim in stims_idx:
            hits_slmtargets[stim] = [None] * len(self.slmtargets_ids)  # start with 0 for all stims
        hits_slmtargets_df = pd.DataFrame(hits_slmtargets, index=self.slmtargets_ids)  # population dataframe

        reliability_slmtargets = {}  # dict will be used to store the reliability results for each targeted cell

        # dFF response traces for successful photostim trials
        traces_dff_successes = {}
        cell_ids = df.index
        for target_idx in range(len(cell_ids)):
            traces_dff_successes_l = []
            success = 0
            counter = 0
            responses = []
            for stim_idx in stims_idx:
                dff_trace = targets_traces[target_idx][stim_idx]
                response_result = np.mean(dff_trace[self.pre_stim + self.stim_duration_frames + 1:
                                                    self.pre_stim + self.stim_duration_frames + self.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
                responses.append(round(response_result, 2))
                if response_result >= threshold:
                    success += 1
                    hits_slmtargets_df.loc[target_idx, stim_idx] = 1
                    traces_dff_successes_l.append(dff_trace)
                else:
                    hits_slmtargets_df.loc[target_idx, stim_idx] = 0

                df.loc[target_idx, stim_idx] = response_result
                counter += 1
            reliability_slmtargets[target_idx] = round(success / counter * 100., 2)
            traces_dff_successes[target_idx] = np.array(traces_dff_successes_l)

        return reliability_slmtargets, hits_slmtargets_df, df, traces_dff_successes

    # retrieves photostim avg traces for each SLM target, also calculates the reliability % for each SLM target
    def calculate_SLMTarget_SuccessStims(self, hits_slmtargets_df, process: str, stims_idx_l: list, exclude_stims_targets: dict = {}):
        """uses outputs of calculate_SLMTarget_responses_dff to calculate overall successrate of the specified stims

        :param hits_slmtargets_df: pandas dataframe of targets x stims where 1 denotes successful stim response (0 is failure)
        :param stims_idx_l: ls of stims to use for this function (useful when needing to filter out certain stims for in/out of sz)
        :param exclude_stims_targets: dictionary of stims (keys) where the values for each stim contains the targets that should be excluded from counting in the analysis of Success/failure of trial

        :return
            reliability_slmtargets: dict; reliability (% of successful stims) for each SLM target
            traces_SLMtargets_successes_avg: np.array; photostims avg traces for each SLM target (successful stims only)

        """

        print(
            f'\n---------- Calculating {process} stim success rates, and separating stims into successes and failures (of SLM targets) [1.] ----------')

        # choose between .SLMTargets_stims_dff, or .SLMTargets_stims_dfstdF and .SLMTargets_stims_tracedFF for data to process
        if process == 'dF/prestimF':
            if hasattr(self, 'SLMTargets_stims_dff'):
                targets_traces = self.SLMTargets_stims_dff
            else:
                AttributeError('no SLMTargets_stims_dff attr. [1.2]')
        elif process == 'dF/stdF':
            if hasattr(self, 'SLMTargets_stims_dfstdF'):
                targets_traces = self.SLMTargets_stims_dfstdF
            else:
                AttributeError('no SLMTargets_stims_dff attr. [1.2]')

        elif process == 'delta(trace_dFF)':
            if hasattr(self, 'SLMTargets_stims_dff'):
                targets_traces = self.SLMTargets_tracedFF_stims_dff
            else:
                AttributeError('no SLMTargets_tracedFF_stims_dff attr. [1.3]')
        else:
            ValueError('need to assign to process: dF/prestimF or dF/stdF or trace dFF')

        traces_SLMtargets_successes_avg_dict = {}
        traces_SLMtargets_failures_avg_dict = {}
        reliability_slmtargets = {}
        for target_idx in hits_slmtargets_df.index:
            traces_SLMtargets_successes_l = []
            traces_SLMtargets_failures_l = []
            success = 0
            counter = 0
            for stim_idx in stims_idx_l:
                if stim_idx in exclude_stims_targets.keys():
                    if target_idx not in exclude_stims_targets[stim_idx]:
                        continu_ = True
                    else:
                        continu_ = False
                else:
                    continu_ = True
                if continu_:
                    counter += 1
                    if hits_slmtargets_df.loc[target_idx, stim_idx] == 1:
                        success += 1
                        dff_trace = targets_traces[target_idx][stim_idx]  # grab the successful photostim trace based on the stim index
                        traces_SLMtargets_successes_l.append(dff_trace)
                    else:
                        success += 0
                        dff_trace = targets_traces[target_idx][stim_idx]
                        traces_SLMtargets_failures_l.append(
                            dff_trace)  # grab the failure photostim trace based on the stim index

            if counter > 0:
                reliability_slmtargets[target_idx] = round(success / counter * 100., 2)
            if success > 0:
                traces_SLMtargets_successes_avg_dict[target_idx] = np.mean(traces_SLMtargets_successes_l, axis=0)
            if success < counter:  # this helps protect against cases where a trial is 100% successful (and there's no failures).
                traces_SLMtargets_failures_avg_dict[target_idx] = np.mean(traces_SLMtargets_failures_l, axis=0)

        # # make plot of successes and failures - testing code stuff
        # pj.make_general_plot(data_arr=traces_SLMtargets_successes_l, v_span=(self.pre_stim, self.pre_stim + self.stim_duration_frames + 1), suptitle=f'Photostim. Successes - {process}')
        # pj.make_general_plot(data_arr=traces_SLMtargets_failures_l, v_span=(self.pre_stim, self.pre_stim + self.stim_duration_frames + 1), suptitle=f'Photostim. Failures - {process}')

        return reliability_slmtargets, traces_SLMtargets_successes_avg_dict, traces_SLMtargets_failures_avg_dict

    # def get_alltargets_stim_traces_norm(self, pre_stim_sec=15, post_stim_sec=200, filter_sz: bool = False, stims_idx_l: ls = None):
    #     """
    #     primary function to measure the dFF and dF/setdF traces for photostimulated targets.
    #     :param stims:
    #     :param targets_idx: integer for the index of target cell to process
    #     :param subselect_cells: ls of cells to subset from the overall set of traces (use in place of targets_idx if desired)
    #     :param pre_stim_sec: number of frames to use as pre-stim
    #     :param post_stim_sec: number of frames to use as post-stim
    #     :param filter_sz: whether to filter out stims that are occuring seizures
    #     :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
    #     """
    #
    #     if filter_sz:
    #         print('\n -- working on getting stim traces for cells inside sz boundary --')
    #
    #     if stims_idx_l is None:
    #         stim_timings = self.stim_start_frames
    #     else:
    #         stim_timings = [self.stim_start_frames[stim_idx] for stim_idx in stims_idx_l]
    #
    #     self.s2p_rois_nontargets = [cell for cell in self.cell_id if cell not in self.s2p_cell_targets]  # need to also detect (and exclude) ROIs that are within some radius of the SLM targets
    #     num_cells = len(self.s2p_rois_nontargets)
    #     targets_trace = self.raw
    #
    #     # collect photostim timed average dff traces of photostim nontargets
    #     nontargets_dff = np.zeros(
    #         [num_cells, len(self.stim_start_frames), pre_stim_sec + self.stim_duration_frames + post_stim_sec])
    #     # nontargets_dff_avg = np.zeros([num_cells, pre_stim_sec + post_stim_sec])
    #
    #     nontargets_dfstdF = np.zeros(
    #         [num_cells, len(self.stim_start_frames), pre_stim_sec + self.stim_duration_frames + post_stim_sec])
    #     # nontargets_dfstdF_avg = np.zeros([num_cells, pre_stim_sec + post_stim_sec])
    #
    #     nontargets_raw = np.zeros(
    #         [num_cells, len(self.stim_start_frames), pre_stim_sec + self.stim_duration_frames + post_stim_sec])
    #     # nontargets_raw_avg = np.zeros([num_cells, pre_stim_sec + post_stim_sec])
    #
    #
    #     for cell_idx in range(num_cells):
    #
    #         if filter_sz:
    #             if hasattr(self, 'slmtargets_szboundary_stim'):  ## change to ROIs in sz for each stim -- has this classification been done for non-targets ROIs?
    #                 flu = []
    #                 for stim in stim_timings:
    #                     if stim in self.slmtargets_szboundary_stim.keys():  # some stims dont have sz boundaries because of issues with their TIFFs not being made properly (not readable in Fiji), usually it is the first TIFF in a seizure
    #                         if cell_idx not in self.slmtargets_szboundary_stim[stim]:
    #                             flu.append(targets_trace[cell_idx][stim - pre_stim_sec: stim + self.stim_duration_frames + post_stim_sec])
    #             else:
    #                 flu = []
    #                 print('classifying of sz boundaries not completed for this expobj', self.metainfo['animal prep.'], self.metainfo['trial'])
    #             # flu = [targets_trace[cell_idx][stim - pre_stim_sec: stim + self.stim_duration_frames + post_stim_sec] for
    #             #        stim
    #             #        in stim_timings if
    #             #        stim not in self.seizure_frames]
    #         else:
    #             flu = [targets_trace[cell_idx][stim - pre_stim_sec: stim + self.stim_duration_frames + post_stim_sec] for
    #                    stim
    #                    in stim_timings]
    #
    #         # flu_dfstdF = []
    #         # flu_dff = []
    #         # flu = []
    #         if len(flu) > 0:
    #             for i in range(len(flu)):
    #                 trace = flu[i]
    #                 mean_pre = np.mean(trace[0:pre_stim_sec])
    #                 trace_dff = ((trace - mean_pre) / mean_pre) * 100
    #                 std_pre = np.std(trace[0:pre_stim_sec])
    #                 dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #
    #                 targets_raw[cell_idx, i] = trace
    #                 targets_dff[cell_idx, i] = trace_dff
    #                 targets_dfstdF[cell_idx, i] = dFstdF
    #                 # flu_dfstdF.append(dFstdF)
    #                 # flu_dff.append(trace_dff)
    #
    #         # targets_dff.append(flu_dff)  # contains all individual dFF traces for all stim times
    #         # targets_dff_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times
    #
    #         # targets_dfstdF.append(flu_dfstdF)
    #         # targets_dfstdF_avg.append(np.nanmean(flu_dfstdF, axis=0))
    #
    #         # SLMTargets_stims_raw.append(flu)
    #         # targets_raw_avg.append(np.nanmean(flu, axis=0))
    #
    #     targets_dff_avg = np.mean(targets_dff, axis=1)
    #     targets_dfstdF_avg = np.mean(targets_dfstdF, axis=1)
    #     targets_raw_avg = np.mean(targets_raw, axis=1)
    #
    #     print(targets_dff_avg.shape)
    #
    #     return targets_dff, targets_dff_avg, targets_dfstdF, targets_dfstdF_avg, targets_raw, targets_raw_avg

    def _makeNontargetsStimTracesArray(self, stim_timings, normalize_to='pre-stim', save=True):
        """
        primary function to retrieve photostimulation trial timed Fluorescence traces for non-targets (ROIs taken from suite2p).
        :param self: alloptical experiment object
        :param normalize_to: str; either "baseline" or "pre-stim" or "whole-trace"
        :return: plot of avg_dFF of 100 randomly selected nontargets
        """
        print('\nCollecting peri-stim traces ')

        # collect photostim timed average dff traces of photostim targets
        dff_traces = []
        dff_traces_avg = []

        dfstdF_traces = []
        dfstdF_traces_avg = []

        raw_traces = []
        raw_traces_avg = []

        for cell in self.s2p_nontargets:
            # print('considering cell # %s' % cell)
            cell_idx = self.cell_id.index(cell)
            flu_trials = [self.raw[cell_idx][stim - self.pre_stim: stim + self.stim_duration_frames + self.post_stim]
                          for stim in stim_timings]

            dff_trace = Utils.normalize_dff(self.raw[cell_idx],
                                      threshold_pct=50)  # normalize trace (dFF) to mean of whole trace

            if normalize_to == 'baseline':  # probably gonna ax this anyways
                flu_dff = []
                mean_spont_baseline = np.mean(self.baseline_raw[cell_idx])
                for i in range(len(flu_trials)):
                    trace_dff = ((flu_trials[i] - mean_spont_baseline) / mean_spont_baseline) * 100

                    # add nan if cell is inside sz boundary for this stim
                    if hasattr(self, 'slmtargets_szboundary_stim'):
                        if self.is_cell_insz(cell=cell, stim=stim_timings[i]):
                            trace_dff = [np.nan] * len(flu_trials[i])

                    flu_dff.append(trace_dff)

            elif normalize_to == 'whole-trace':
                print('s2p neu. corrected trace statistics: mean: %s (min: %s, max: %s, std: %s)' %
                      (np.mean(self.raw[cell_idx]), np.min(self.raw[cell_idx]), np.max(self.raw[cell_idx]),
                       np.std(self.raw[cell_idx], ddof=1)))
                # dfstdf_trace = (self.raw[cell_idx] - np.mean(self.raw[cell_idx])) / np.std(self.raw[cell_idx], ddof=1)  # normalize trace (dFstdF) to std of whole trace
                flu_dfstdF = []
                flu_dff = []
                flu_dff_ = [dff_trace[stim - self.pre_stim: stim + self.stim_duration_frames + self.post_stim] for
                            stim in stim_timings if
                            stim not in self.seizure_frames]

                for i in range(len(flu_dff_)):
                    trace = flu_dff_[i]
                    mean_pre = np.mean(trace[0:self.pre_stim])
                    trace_dff = trace - mean_pre  # correct dFF of this trial to mean of pre-stim dFF
                    std_pre = np.std(trace[0:self.pre_stim], ddof=1)
                    dFstdF = trace_dff / std_pre  # normalize dFF of this trial by std of pre-stim dFF

                    flu_dff.append(trace_dff)
                    flu_dfstdF.append(dFstdF)

            elif normalize_to == 'pre-stim':
                flu_dff = []
                flu_dfstdF = []
                # print('|- splitting trace by photostim. trials and correcting by pre-stim period')
                for i in range(len(flu_trials)):
                    trace = flu_trials[i]
                    mean_pre = np.mean(trace[0:self.pre_stim])

                    std_pre = np.std(trace[0:self.pre_stim], ddof=1)
                    # dFstdF = (((trace - mean_pre) / mean_pre) * 100) / std_pre  # make dF divided by std of pre-stim F trace
                    dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                    if mean_pre < 1:
                        # print('risky cell here at cell # %s, trial # %s, mean pre: %s [1.1]' % (cell, i+1, mean_pre))
                        trace_dff = [np.nan] * len(trace)
                        dFstdF = [np.nan] * len(
                            trace)  # - commented out to test if we need to exclude cells for this correction with low mean_pre since you're not dividing by a bad mean_pre value
                    else:
                        # trace_dff = ((trace - mean_pre) / mean_pre) * 100
                        trace_dff = Utils.normalize_dff(trace, threshold_val=mean_pre)
                        # std_pre = np.std(trace[0:self.pre_stim], ddof=1)
                        # # dFstdF = (((trace - mean_pre) / mean_pre) * 100) / std_pre  # make dF divided by std of pre-stim F trace
                        # dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                    # # add nan if cell is inside sz boundary for this stim -- temporarily commented out for a while
                    # if 'post' in self.metainfo['exptype']:
                    #     if hasattr(self, 'slmtargets_szboundary_stim'):
                    #         if self.is_cell_insz(cell=cell, stim=stim_timings[i]):
                    #             trace_dff = [np.nan] * len(trace)
                    #             dFstdF = [np.nan] * len(trace)
                    #     else:
                    #         AttributeError(
                    #             'no slmtargets_szboundary_stim attr, so classify cells in sz boundary hasnot been saved for this expobj')

                    flu_dff.append(trace_dff)
                    flu_dfstdF.append(dFstdF)

            else:
                TypeError('need to specify what to normalize to in get_targets_dFF (choose "baseline" or "pre-stim")')

            dff_traces.append(flu_dff)  # contains all individual dFF traces for all stim times
            dff_traces_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

            dfstdF_traces.append(flu_dfstdF)
            dfstdF_traces_avg.append(np.nanmean(flu_dfstdF, axis=0))

            raw_traces.append(flu_trials)
            raw_traces_avg.append(np.nanmean(flu_trials, axis=0))

        if normalize_to == 'baseline':
            print(
                '\nCompleted collecting pre to post stim traces -- normalized to spont imaging as baseline -- for %s cells' % len(
                    dff_traces_avg))
            self.dff_traces = dff_traces
            self.dff_traces_avg = dff_traces_avg
            # return dff_traces, dff_traces_avg
        elif normalize_to == 'pre-stim' or normalize_to == 'whole-trace':
            print(
                f'\nCompleted collecting pre to post stim traces -- normalized to pre-stim period or maybe whole-trace -- for {len(dff_traces_avg)} cells, {len(flu_trials)} stims')
            self.dff_traces = np.asarray(dff_traces)
            self.dff_traces_avg = np.asarray([i for i in dff_traces_avg])
            self.dfstdF_traces = np.asarray(dfstdF_traces)
            self.dfstdF_traces_avg = np.asarray([i for i in dfstdF_traces_avg])
            self.raw_traces = np.asarray(raw_traces)
            self.raw_traces_avg = np.asarray([i for i in raw_traces_avg])

        print('\nFinished collecting peri-stim traces ')

        self.save() if save else None

        # return dff_traces, dff_traces_avg, dfstdF_traces, dfstdF_traces_avg, raw_traces, raw_traces_avg

    def _trialProcessing_nontargets(expobj, normalize_to='pre-stim', save=True):
        '''
        Uses dfstdf traces for individual cells and photostim trials, calculate the mean amplitudes of response and
        statistical significance across all trials for all cells

        Inputs:
            plane             - imaging plane n
        '''

        print('\n----------------------------------------------------------------')
        print('running trial Processing for nontargets ')
        print('----------------------------------------------------------------')

        # make trial arrays from dff data shape: [cells x stims x frames]
        expobj._makeNontargetsStimTracesArray(stim_timings=expobj.stim_start_frames, normalize_to=normalize_to,
                                              save=False)

        # create parameters, slices, and subsets for making pre-stim and post-stim arrays to use in stats comparison
        # test_period = expobj.pre_stim_response_window_msec / 1000  # sec
        # expobj.test_frames = int(expobj.fps * test_period)  # test period for stats
        expobj.pre_stim_frames_test = np.s_[expobj.pre_stim - expobj.pre_stim_response_frames_window: expobj.pre_stim]
        stim_end = expobj.pre_stim + expobj.stim_duration_frames
        expobj.post_stim_frames_slice = np.s_[stim_end: stim_end + expobj.post_stim_response_frames_window]

        # mean pre and post stimulus (within post-stim response window) flu trace values for all cells, all trials
        expobj.analysis_array = expobj.dfstdF_traces  # NOTE: USING dF/stdF TRACES
        expobj.pre_array = np.mean(expobj.analysis_array[:, :, expobj.pre_stim_frames_test],
                                   axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
        expobj.post_array = np.mean(expobj.analysis_array[:, :, expobj.post_stim_frames_slice],
                                    axis=1)  # [cells x poststim frames] (avg'd taken over all stims)

        # ar2 = expobj.analysis_array[18, :, expobj.post_stim_frames_slice]
        # ar3 = ar2[~np.isnan(ar2).any(axis=1)]
        # assert np.nanmean(ar2) == np.nanmean(ar3)
        # expobj.analysis_array = expobj.analysis_array[~np.isnan(expobj.analysis_array).any(axis=1)]

        # measure avg response value for each trial, all cells --> return array with 3 axes [cells x response_magnitude_per_stim (avg'd taken over response window)]
        expobj.post_array_responses = []  ### this and the for loop below was implemented to try to root out stims with nan's but it's likley not necessary...
        for i in np.arange(expobj.analysis_array.shape[0]):
            a = expobj.analysis_array[i][~np.isnan(expobj.analysis_array[i]).any(axis=1)]
            responses = a.mean(axis=1)
            expobj.post_array_responses.append(responses)

        expobj.post_array_responses = np.mean(expobj.analysis_array[:, :, expobj.post_stim_frames_slice], axis=2)
        expobj.wilcoxons = expobj._runWilcoxonsTest()

        expobj.save() if save else None

    def _runWilcoxonsTest(expobj, array1=None, array2=None):

        if array1 is None and array2 is None:
            array1 = expobj.pre_array;
            array2 = expobj.post_array

        # check if the two distributions of flu values (pre/post) are different
        assert array1.shape == array2.shape, 'shapes for expobj.pre_array and expobj.post_array need to be the same for wilcoxon test'
        wilcoxons = np.empty(len(array1))  # [cell (p-value)]

        for cell in range(len(array1)):
            wilcoxons[cell] = stats.wilcoxon(array2[cell], array1[cell])[1]

        return wilcoxons

        # expobj.save() if save else None

    def _sigTestAvgResponse_nontargets(expobj, p_vals=None, alpha=0.1, save=True):
        """
        Uses the p values and a threshold for the Benjamini-Hochberg correction to return which
        cells are still significant after correcting for multiple significance testing
        """
        print('\n----------------------------------------------------------------')
        print('running statistical significance testing for nontargets response arrays ')
        print('----------------------------------------------------------------')

        # p_vals = expobj.wilcoxons
        sig_units = np.full_like(p_vals, False, dtype=bool)

        try:
            sig_units, _, _, _ = sm.stats.multitest.multipletests(p_vals, alpha=alpha, method='fdr_bh',
                                                                  is_sorted=False, returnsorted=False)
        except ZeroDivisionError:
            print('no cells responding')

        # # p values without bonferroni correction
        # no_bonf_corr = [i for i, p in enumerate(p_vals) if p < 0.05]
        # expobj.nomulti_sig_units = np.zeros(len(expobj.s2p_nontargets), dtype='bool')
        # expobj.nomulti_sig_units[no_bonf_corr] = True

        # expobj.save() if save else None

        # p values after bonferroni correction
        #         bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05 / expobj.n_units[plane]]
        #         sig_units = np.zeros(expobj.n_units[plane], dtype='bool')
        #         sig_units[bonf_corr] = True

        return sig_units

    # used for creating tiffs that remove artifacts from alloptical experiments with photostim artifacts
    def rm_artifacts_tiffs(expobj, tiffs_loc, new_tiffs):
        ### make a new tiff file (not for suite2p) with the first photostim frame whitened, and save new tiff
        print('\n-----making processed photostim .tiff from:')
        tiff_path = tiffs_loc
        print(tiff_path)
        im_stack = tf.imread(tiff_path, key=range(expobj.n_frames))
        print('Processing experiment tiff of shape: ', im_stack.shape)

        frames_to_whiten = []
        for j in expobj.stim_start_frames:
            frames_to_whiten.append(j)

        # number of photostim frames with artifacts
        frames_to_remove = []
        for j in expobj.stim_start_frames:
            for i in range(0,
                           expobj.stim_duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
                frames_to_remove.append(j + i)

        print('# of total photostim artifact frames:', len(frames_to_remove))

        im_stack_1 = im_stack
        a = np.full_like(im_stack_1[0], fill_value=0)
        a[0:100, 0:100] = 5000.
        for frame in frames_to_whiten:
            im_stack_1[frame - 3] = im_stack_1[frame - 3] + a
            im_stack_1[frame - 2] = im_stack_1[frame - 2] + a
            im_stack_1[frame - 1] = im_stack_1[frame - 1] + a
        print('Shape', im_stack_1.shape)

        im_stack_1 = np.delete(im_stack_1, frames_to_remove, axis=0)
        print('After delete shape artifactrem', im_stack_1.shape)

        save_path = (new_tiffs + "_artifactrm.tif")
        tf.imwrite(save_path, im_stack_1, photometric='minisblack')

        del im_stack_1

        # draw areas on top of im_stack_1 where targets are:
        im_stack_2 = im_stack
        print('Shape', im_stack_2.shape)

        for stim in range(expobj.n_groups):
            b = np.full_like(im_stack_2[0], fill_value=0)
            targets = expobj.target_areas[stim]
            for i in np.arange(len(targets)):
                for j in targets[i]:
                    b[j] = 5000

            all_stim_start_frames = []
            for stim_frame in expobj.stim_start_frames[stim::expobj.n_groups]:
                all_stim_start_frames.append(stim_frame)
            for frame in all_stim_start_frames:
                #         im_stack_2[frame-4] = im_stack_2[frame-4]+b
                #         im_stack_2[frame-3] = im_stack_2[frame-3]+b
                #        im_stack_2[frame-2] = im_stack_2[frame-2]+b
                im_stack_2[frame - 1] = im_stack_2[frame - 1] + b

        im_stack_2 = np.delete(im_stack_2, expobj.photostim_frames, axis=0)

        print('After delete shape targetcells', im_stack_2.shape)

        save_path = (new_tiffs + '_targetcells.tif')
        tf.imwrite(save_path, im_stack_2, photometric='minisblack')

        print('done saving to: ', save_path)

        del im_stack_2
        del im_stack

    def s2pMasks(expobj, s2p_path, cell_ids):
        os.chdir(s2p_path)
        stat = np.load('stat.npy', allow_pickle=True)
        ops = np.load('ops.npy', allow_pickle=True).item()
        iscell = np.load('iscell.npy', allow_pickle=True)
        mask_img = np.zeros((ops['Ly'], ops['Lx']), dtype='uint8')
        for n in range(0, len(iscell)):
            if n in cell_ids:
                ypix = stat[n]['ypix']
                xpix = stat[n]['xpix']
                mask_img[ypix, xpix] = 3000

        # s2p targets - all SLM targets
        targets_s2p_img = np.zeros((ops['Ly'], ops['Lx']), dtype='uint8')
        for n in range(0, len(iscell)):
            if n in expobj.s2p_cell_targets:
                ypix = stat[n]['ypix']
                xpix = stat[n]['xpix']
                targets_s2p_img[ypix, xpix] = 3000

        # # s2p targets - SLM group #1 targets
        # targets_s2p_img_1 = np.zeros((ops['Ly'], ops['Lx']), dtype='uint8')
        # for n in range(0, len(iscell)):
        #     if n in obj.s2p_cell_targets_1:
        #         ypix = stat[n]['ypix']
        #         xpix = stat[n]['xpix']
        #         targets_s2p_img_1[ypix, xpix] = 3000
        #
        # # s2p targets - SLM group #2 targets
        # targets_s2p_img_2 = np.zeros((ops['Ly'], ops['Lx']), dtype='uint8')
        # for n in range(0, len(iscell)):
        #     if n in obj.s2p_cell_targets_2:
        #         ypix = stat[n]['ypix']
        #         xpix = stat[n]['xpix']
        #         targets_s2p_img_2[ypix, xpix] = 3000

        return mask_img, targets_s2p_img,  # targets_s2p_img_1, targets_s2p_img_2

    def getTargetImage(obj):
        targ_img = np.zeros((obj.frame_x, obj.frame_y), dtype='uint8')
        # all FOV targets
        targ_areas = obj.target_areas
        for targ_area in targ_areas:
            for coord in targ_area:
                targ_img[coord[1], coord[0]] = 3000

        # targ_img_1 = np.zeros((obj.frame_x, obj.frame_y), dtype='uint8')
        # # FOV targets group #1
        # targ_areas = obj.target_areas_1
        # for targ_area in targ_areas:
        #     for coord in targ_area:
        #         targ_img_1[coord[1], coord[0]] = 3000
        #
        # targ_img_2 = np.zeros((obj.frame_x, obj.frame_y), dtype='uint8')
        # # FOV targets group #2
        # targ_areas = obj.target_areas_2
        # for targ_area in targ_areas:
        #     for coord in targ_area:
        #         targ_img_2[coord[1], coord[0]] = 3000

        return targ_img  # , targ_img_1, targ_img_2

    def s2pMaskStack(obj, pkl_list, s2p_path, parent_folder, force_redo: bool = False):
        """makes a TIFF stack with the s2p mean image, and then suite2p ROI masks for all cells detected, target cells, and also SLM targets as well?"""

        if force_redo:
            continu = True
        elif hasattr(obj, 's2p_cell_targets'):
            print('skipped re-making TIFF stack of finding s2p targets from suite2p cell ls')
            continu = False
        else:
            continu = True

        if continu:

            for pkl in pkl_list:
                expobj = obj

                print('Retrieving s2p masks for:', pkl, end='\r')

                # with open(pkl, 'rb') as f:
                #     expobj = pickle.load(f)

                # ls of cell ids to filter s2p masks by
                # cell_id_list = [ls(range(1, 99999)),  # all
                #                 expobj.photostim_r.cell_id[0],  # cells
                #                 [expobj.photostim_r.cell_id[0][i] for i, b in enumerate(expobj.photostim_r.cell_s1[0]) if
                #                  b == False],  # s2 cells
                #                 [expobj.photostim_r.cell_id[0][i] for i, b in enumerate(expobj.photostim_r.is_target) if
                #                  b == 1],  # pr cells
                #                 [expobj.photostim_s.cell_id[0][i] for i, b in enumerate(expobj.photostim_s.is_target) if
                #                  b == 1],  # ps cells
                #                 ]
                #
                cell_ids = expobj.cell_id

                # empty stack to fill with images
                stack = np.empty((0, expobj.frame_y, expobj.frame_x), dtype='uint8')

                s2p_path = s2p_path

                # mean image from s2p
                mean_img = obj.s2pMeanImage(s2p_path)
                mean_img = np.expand_dims(mean_img, axis=0)
                stack = np.append(stack, mean_img, axis=0)

                # mask images from s2p
                mask_img, targets_s2p_img = obj.s2pMasks(s2p_path=s2p_path, cell_ids=cell_ids)
                mask_img = np.expand_dims(mask_img, axis=0)
                targets_s2p_img = np.expand_dims(targets_s2p_img, axis=0)
                # targets_s2p_img_1 = np.expand_dims(targets_s2p_img_1, axis=0)
                # targets_s2p_img_2 = np.expand_dims(targets_s2p_img_2, axis=0)
                stack = np.append(stack, mask_img, axis=0)
                stack = np.append(stack, targets_s2p_img, axis=0)
                # stack = np.append(stack, targets_s2p_img_1, axis=0)
                # stack = np.append(stack, targets_s2p_img_2, axis=0)

                # # sta images
                # for file in os.listdir(stam_save_path):
                #     if all(s in file for s in ['AvgImage', expobj.photostim_r.tiff_path.split('/')[-1]]):
                #         pr_sta_img = tf.imread(os.path.join(stam_save_path, file))
                #         pr_sta_img = np.expand_dims(pr_sta_img, axis=0)
                #     elif all(s in file for s in ['AvgImage', expobj.photostim_s.tiff_path.split('/')[-1]]):
                #         ps_sta_img = tf.imread(os.path.join(stam_save_path, file))
                #         ps_sta_img = np.expand_dims(ps_sta_img, axis=0)

                # stack = np.append(stack, pr_sta_img, axis=0)
                # stack = np.append(stack, ps_sta_img, axis=0)

                # target images
                targ_img = obj.getTargetImage(expobj)
                targ_img = np.expand_dims(targ_img, axis=0)
                stack = np.append(stack, targ_img, axis=0)

                # targ_img_1 = np.expand_dims(targ_img_1, axis=0)
                # stack = np.append(stack, targ_img_1, axis=0)
                #
                # targ_img_2 = np.expand_dims(targ_img_2, axis=0)
                # stack = np.append(stack, targ_img_2, axis=0)

                # stack is now: mean_img, all_rois, all_cells, s2_cells, pr_cells, ps_cells,
                # (whisker,) pr_sta_img, ps_sta_img, pr_target_areas, ps_target_areas
                # c, x, y = stack.shape
                # stack.shape = 1, 1, c, x, y, 1  # dimensions in TZCYXS order

                x_pix = expobj.pix_sz_x
                y_pix = expobj.pix_sz_y

                save_path = os.path.join(parent_folder, pkl.split('/')[-1][:-4] + '_s2p_masks.tif')

                tf.imwrite(save_path, stack, photometric='minisblack')
                print('\ns2p ROI + photostim targets masks saved in TIFF to: ', save_path)


    #### plotting functions
    ### plot the location of all SLM targets, along with option for plotting the mean img of the current trial
    # @print_start_end_plot
    @plot_piping_decorator(figsize=(5, 5))
    def plot_SLMtargets_Locs(expobj, targets_coords: list = None, background: np.ndarray = None, fig=None, ax=None, **kwargs):
        """
        plot SLM target coordinate locations

        :param expobj:
        :param targets_coords: ls containing (x,y) coordinates of targets to plot
        :param background:
        :param kwargs:
        :return:
        """

        # if 'fig' in kwargs.keys():
        #     fig = kwargs['fig']
        #     ax = kwargs['ax']
        # else:
        #     if 'figsize' in kwargs.keys():
        #         fig, ax = plt.subplots(figsize=kwargs['figsize'])
        #     else:
        #         fig, ax = plt.subplots()

        if background is None:
            background = np.zeros((expobj.frame_x, expobj.frame_y), dtype='uint16')
            ax.imshow(background, cmap='gray')
        else:
            ax.imshow(background, cmap='gray')

        colors = pj.make_random_color_array(len(expobj.target_coords))
        if targets_coords is None:
            if len(expobj.target_coords) > 1:
                for i in range(len(expobj.target_coords)):
                    for (x, y) in expobj.target_coords[i]:
                        ax.scatter(x=x, y=y, edgecolors=colors[i], facecolors='none', linewidths=2.0)
            else:
                if 'edgecolors' in kwargs.keys():
                    edgecolors = kwargs['edgecolors']
                else:
                    edgecolors = 'yellowgreen'
                for (x, y) in expobj.target_coords_all:
                    ax.scatter(x=x, y=y, edgecolors=edgecolors, facecolors='none', linewidths=2.0)
        elif targets_coords:
            if 'edgecolors' in kwargs.keys():
                edgecolors = kwargs['edgecolors']
            else:
                edgecolors = 'yellowgreen'
            pj.plot_coordinates(coords=targets_coords, frame_x=expobj.frame_x, frame_y=expobj.frame_y,
                                edgecolors=edgecolors,
                                background=background, fig=fig, ax=ax)

        ax.margins(0)
        fig.tight_layout()

        if 'title' in kwargs.keys():
            if kwargs['title'] is not None:
                ax.set_title(kwargs['title'])
            else:
                pass
        else:
            ax.set_title('SLM targets location - %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))

        # if 'show' in kwargs.keys():
        #     if kwargs['show'] is True:
        #         fig.show()
        #     else:
        #         return fig, ax
        # else:
        #     fig.show()
        #
        # return fig, ax if 'fig' in kwargs.keys() else None

    # simple plot of the location of the given cell(s) against a black FOV
    # @print_start_end_plot
    @plot_piping_decorator(figsize=(5, 5))
    def plot_cells_loc(expobj, cells: list, title=None, background: np.array = None, scatter_only: bool = False, show_s2p_targets: bool = True,
                       color_float_list: list = None, cmap: str = 'Reds', invert_y=True, fig=None, ax=None, **kwargs):
        """
        plots an image of the FOV to show the locations of cells given in cells ls.
        :param background: either 2dim numpy array to use as the backsplash or None (where black backsplash will be created)
        :param expobj: alloptical or 2p imaging object
        :param edgecolor: str to specify edgecolor of the scatter plot for cells
        :param cells: ls of cells to plot
        :param title: str title for plot
        :param color_float_list: if given, it will be used to color the cells according a colormap
        :param cmap: cmap to be used in conjuction with the color_float_array argument
        :param show_s2p_targets: if True, then will prioritize coloring of cell points based on whether they were photostim targets
        :param kwargs: optional arguments
                invert_y: if True, invert the reverse the direction of the y axis
                show: if True, show the plot
                fig: a fig plt.subplots() instance, if provided use this fig for making figure
                ax: a ax plt.subplots() instance, if provided use this ax for plotting
        """

        # # if there is a fig and ax provided in the function call then use those, otherwise start anew
        # if 'fig' in kwargs.keys():
        #     fig = kwargs['fig']
        #     ax = kwargs['ax']
        # else:
        #     fig, ax = plt.subplots()

        x_list = []
        y_list = []
        for cell in cells:
            y, x = expobj.stat[expobj.cell_id.index(cell)]['med']
            x_list.append(x)
            y_list.append(y)

            if show_s2p_targets:
                if hasattr(expobj, 's2p_cell_targets'):
                    if cell in expobj.s2p_cell_targets:
                        color_ = '#F02A71'
                    else:
                        color_ = 'none'
                else:
                    color_ = 'none'
                ax.scatter(x=x, y=y, edgecolors=None, facecolors=color_, linewidths=0.8)
            elif color_float_list:
                # ax.scatter(x=x, y=y, edgecolors='none', c=color_float_list[cells.index(cell)], linewidths=0.8,
                #            cmap=cmap)
                pass
            else:
                if 'edgecolors' in kwargs.keys():
                    edgecolors = kwargs['edgecolors']
                else:
                    edgecolors = 'yellowgreen'
                ax.scatter(x=x, y=y, edgecolors=edgecolors, facecolors='none', linewidths=0.8)

        if color_float_list:
            ac = ax.scatter(x=x_list, y=y_list, edgecolors='none', c=color_float_list, linewidths=0.8,
                            cmap=cmap, zorder=1)

            plt.colorbar(ac, ax=ax)

        if not scatter_only:
            if background is None:
                black = np.zeros((expobj.frame_x, expobj.frame_y), dtype='uint16')
                ax.imshow(black, cmap='Greys_r', zorder=0)
                ax.set_xlim(0, expobj.frame_x)
                ax.set_ylim(0, expobj.frame_y)
            else:
                ax.imshow(background, cmap='Greys_r', zorder=0)

        if title is not None:
            plt.suptitle(title, wrap=True)

        if 'text' in kwargs.keys():
            if kwargs['text'] is not None:
                ax.text(0.99, 0.95, kwargs['text'],
                        verticalalignment='top', horizontalalignment='right',
                        transform=ax.transAxes, fontweight='bold',
                        color='white', fontsize=10)

        if 'hide_axis_labels' in kwargs.keys():
            ax.set_xticks(ticks=[])
            ax.set_xticklabels([])
            ax.set_yticks(ticks=[])
            ax.set_yticklabels([])

        if 'invert_y' in kwargs.keys():
            if kwargs['invert_y']:
                ax.invert_yaxis()

        # if 'show' in kwargs.keys():
        #     if kwargs['show'] is True:
        #         fig.show()
        #     else:
        #         pass
        # else:
        #     fig.show()
        #
        # return fig, ax if 'fig' in kwargs.keys() else None

    # plot to show s2p ROIs location, colored as specified
    def s2pRoiImage(expobj, save_fig: str = None):
        """
        plot to show the classification of each cell as the actual's filling in the cell's ROI pixels.

        :param expobj: expobj associated with data
        :param df: pandas dataframe (cell_id x stim frames)
        :param clim: color limits
        :param plot_target_coords: bool, if True plot the actual X and Y coords of all photostim cell targets
        :param save_fig: where to save the save figure (optional)
        :return:
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        if expobj.frame_x == 512:
            s = 0.003 * (1024 / expobj.frame_x * 4)
        else:
            s = 0.003
        ##### targets areas image
        targ_img = np.zeros([expobj.frame_x, expobj.frame_y], dtype='float')
        target_areas_exclude = np.array(expobj.target_areas_exclude)
        targ_img[target_areas_exclude[:, :, 1], target_areas_exclude[:, :, 0]] = 1
        x = np.asarray(list(range(expobj.frame_x)) * expobj.frame_y)
        y = np.asarray([i_y for i_y in range(expobj.frame_y) for i_x in range(expobj.frame_x)])
        img = targ_img.flatten()
        im_array = np.array([x, y], dtype=np.float)
        ax.scatter(im_array[0], im_array[1], c=img, cmap='gray', s=s, zorder=0, alpha=1)

        ##### suite2p ROIs areas image - nontargets
        for n in expobj.s2p_nontargets:
            idx = expobj.cell_id.index(n)
            ypix = expobj.stat[idx]['ypix']
            xpix = expobj.stat[idx]['xpix']
            ax.scatter(xpix, ypix, c='lightsteelblue', s=s, zorder=1, alpha=1)

        ##### suite2p ROIs areas image - exclude cells
        for n in expobj.s2p_cells_exclude:
            idx = expobj.cell_id.index(n)
            ypix = expobj.stat[idx]['ypix']
            xpix = expobj.stat[idx]['xpix']
            ax.scatter(xpix, ypix, c='yellow', s=s, zorder=2, alpha=1)

        ##### suite2p ROIs areas image - targeted cells
        for n in expobj.s2p_cell_targets:
            idx = expobj.cell_id.index(n)
            ypix = expobj.stat[idx]['ypix']
            xpix = expobj.stat[idx]['xpix']
            ax.scatter(xpix, ypix, c='red', s=s, zorder=3, alpha=1)

        ax.set_xlim([0, expobj.frame_x])
        ax.set_ylim([0, expobj.frame_y])
        plt.margins(x=0, y=0)
        plt.gca().invert_yaxis()
        # plt.gca().invert_xaxis()
        # fig.show()

        plt.suptitle(
            f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - s2p nontargets (blue), exclude (yellow), targets (red); target_areas (white)",
            y=0.97, fontsize=7)
        plt.show()
        Utils.save_figure(fig, save_path_suffix=f"{save_fig}") if save_fig else None

    # def create_anndata_SLMtargets(expobj):
    #     """
    #     Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial.
    #
    #     """
    #
    #     if expobj.dFF_SLMTargets or expobj.raw_SLMTargets:
    #         # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
    #         # build dataframe for obs_meta from SLM targets information
    #         obs_meta = pd.DataFrame(
    #             columns=['SLM group #', 'SLM target coord'], index=range(expobj.n_targets_total))
    #         for groupnum, targets in enumerate(expobj.target_coords):
    #             for target, coord in enumerate(targets):
    #                 obs_meta.loc[target, 'SLM group #'] = groupnum
    #                 obs_meta.loc[target, 'SLM target coord'] = coord
    #
    #         # build numpy array for multidimensional obs metadata
    #         obs_m = {'SLM targets areas': []}
    #         for groupnum, targets in enumerate(expobj.target_areas):
    #             for target, coord in enumerate(targets):
    #                 obs_m['SLM targets areas'] = groupnum
    #
    #         # SETUP THE VARIABLES ANNOTATIONS TO USE IN anndata
    #         # build dataframe for var annot's - based on stim_start_frames
    #         var_meta = pd.DataFrame(index=['wvfront in sz', 'seizure location'], columns=expobj.stim_start_frames)
    #         for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
    #             if 'pre' in expobj.exptype:
    #                 var_meta.loc['wvfront in sz', stim_frame] = False
    #                 var_meta.loc['seizure location', stim_frame] = None
    #             elif 'post' in expobj.exptype:
    #                 if stim_frame in expobj.stimsWithSzWavefront:
    #                     var_meta.loc['wvfront in sz', stim_frame] = True
    #                     var_meta.loc['seizure location', stim_frame] = '..not-set-yet..'
    #                 else:
    #                     var_meta.loc['wvfront in sz', stim_frame] = False
    #                     var_meta.loc['seizure location', stim_frame] = None
    #
    #         # BUILD LAYERS TO ADD TO anndata OBJECT
    #         layers = {'SLM Targets photostim responses (dFF)': expobj.dFF_SLMTargets
    #                   }
    #
    #         print(f"\n\----- CREATING annotated data object using AnnData:")
    #         __data_type = 'Registered Imaging Raw from SLM targets areas'
    #         adata = AnnotatedData2(X=expobj.raw_SLMTargets, obs=obs_meta, var=var_meta.T, obsm=obs_m,
    #                               layers=layers, data_label=__data_type)
    #
    #         print(f"\n{adata}")
    #         expobj.slmtargets_data = adata
    #     else:
    #         Warning(
    #             'did not create anndata. anndata creation only available if experiments were processed with suite2p and .paq file(s) provided for temporal synchronization')
