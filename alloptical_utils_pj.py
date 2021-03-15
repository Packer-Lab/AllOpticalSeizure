#### NOTE: THIS IS NOT CURRENTLY SETUP TO BE ABLE TO HANDLE MULTIPLE GROUPS/STIMS (IT'S REALLY ONLY FOR A SINGLE STIM TRIGGER PHOTOSTIM RESPONSES)

# TODO need to condense functions that are currently all calculating photostim responses
#      essentially you should only have to calculate the poststim respones for all cells (including targets) once.


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tifffile as tf
import csv
import bisect
import re
import glob
import pandas as pd
import itertools

from scipy import stats
import sys

sys.path.append('/home/pshah/Documents/code/')
from Vape.utils.paq2py import *
from Vape.utils.utils_funcs import *
from suite2p.run_s2p import run_s2p

import xml.etree.ElementTree as ET

from utils import funcs_pj as pjf
from utils.paq_utils import paq_read, frames_discard

import pickle

from numba import njit, jit


# %%
def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y


class alloptical():

    def __init__(self, paths, metainfo, stimtype):
        self.metainfo = metainfo
        self.tiff_path_dir = paths[0]
        self.tiff_path = paths[1]
        self.naparm_path = paths[2]
        self.paq_path = paths[3]

        self._parsePVMetadata()

        self.stim_type = stimtype

        ## CREATE THE APPROPRIATE ANALYSIS SUBFOLDER TO USE FOR SAVING ANALYSIS RESULTS TO
        self.analysis_save_path = self.tiff_path[:21] + 'Analysis/' + self.tiff_path_dir[26:]
        if os.path.exists(self.analysis_save_path):
            pass
        elif os.path.exists(self.analysis_save_path[:-17]):
            os.mkdir(self.analysis_save_path)
        elif os.path.exists(self.analysis_save_path[:-27]):
            os.mkdir(self.analysis_save_path[:-17])

        print('\ninitialized alloptical expobj of exptype and trial: \n', self.metainfo)

    def _getPVStateShard(self, path, key):

        '''
        Used in function PV metadata below
        '''

        value = []
        description = []
        index = []

        xml_tree = ET.parse(path)  # parse xml from a path
        root = xml_tree.getroot()  # make xml tree structure

        pv_state_shard = root.find('PVStateShard')  # find pv state shard element in root

        for elem in pv_state_shard:  # for each element in pv state shard, find the value for the specified key

            if elem.get('key') == key:

                if len(elem) == 0:  # if the element has only one subelement
                    value = elem.get('value')
                    break

                else:  # if the element has many subelements (i.e. lots of entries for that key)
                    for subelem in elem:
                        value.append(subelem.get('value'))
                        description.append(subelem.get('description'))
                        index.append(subelem.get('index'))
            else:
                for subelem in elem:  # if key not in element, try subelements
                    if subelem.get('key') == key:
                        value = elem.get('value')
                        break

            if value:  # if found key in subelement, break the loop
                break

        if not value:  # if no value found at all, raise exception
            raise Exception('ERROR: no element or subelement with that key')

        return value, description, index

    def _parsePVMetadata(self):

        print('\n-----parsing PV Metadata')

        tiff_path = self.tiff_path_dir
        path = []

        try:  # look for xml file in path, or two paths up (achieved by decreasing count in while loop)
            count = 2
            while count != 0 and not path:
                count -= 1
                for file in os.listdir(tiff_path):
                    if file.endswith('.xml'):
                        path = os.path.join(tiff_path, file)
                    if file.endswith('.env'):
                        env_path = os.path.join(tiff_path, file)
                tiff_path = os.path.dirname(tiff_path)

        except:
            raise Exception('ERROR: Could not find xml for this acquisition, check it exists')

        xml_tree = ET.parse(path)  # parse xml from a path
        root = xml_tree.getroot()  # make xml tree structure

        sequence = root.find('Sequence')
        acq_type = sequence.get('type')

        if 'ZSeries' in acq_type:
            n_planes = len(sequence.findall('Frame'))

        else:
            n_planes = 1

        frame_period = float(self._getPVStateShard(path, 'framePeriod')[0])
        fps = 1 / frame_period

        frame_x = int(self._getPVStateShard(path, 'pixelsPerLine')[0])

        frame_y = int(self._getPVStateShard(path, 'linesPerFrame')[0])

        zoom = float(self._getPVStateShard(path, 'opticalZoom')[0])

        scanVolts, _, index = self._getPVStateShard(path, 'currentScanCenter')
        for scanVolts, index in zip(scanVolts, index):
            if index == 'XAxis':
                scan_x = float(scanVolts)
            if index == 'YAxis':
                scan_y = float(scanVolts)

        pixelSize, _, index = self._getPVStateShard(path, 'micronsPerPixel')
        for pixelSize, index in zip(pixelSize, index):
            if index == 'XAxis':
                pix_sz_x = float(pixelSize)
            if index == 'YAxis':
                pix_sz_y = float(pixelSize)

        env_tree = ET.parse(env_path)
        env_root = env_tree.getroot()

        elem_list = env_root.find('TSeries')
        # n_frames = elem_list[0].get('repetitions') # Rob would get the n_frames from env file
        # change this to getting the last actual index from the xml file

        n_frames = root.findall('Sequence/Frame')[-1].get('index')

        self.fps = fps
        self.frame_x = frame_x
        self.frame_y = frame_y
        self.n_planes = n_planes
        self.pix_sz_x = pix_sz_x
        self.pix_sz_y = pix_sz_y
        self.scan_x = scan_x
        self.scan_y = scan_y
        self.zoom = zoom
        self.n_frames = int(n_frames)

        print('n planes:', n_planes,
              '\nn frames:', int(n_frames),
              '\nfps:', fps,
              '\nframe size (px):', frame_x, 'x', frame_y,
              '\nzoom:', zoom,
              '\npixel size (um):', pix_sz_x, pix_sz_y,
              '\nscan centre (V):', scan_x, scan_y
              )

    def s2pRun(self, ops, db, user_batch_size):

        '''run suite2p on the experiment object, using the attributes deteremined directly from the experiment object'''

        num_pixels = self.frame_x * self.frame_y
        sampling_rate = self.fps / self.n_planes
        diameter_x = 13 / self.pix_sz_x
        diameter_y = 13 / self.pix_sz_y
        diameter = int(diameter_x), int(diameter_y)
        batch_size = user_batch_size * (
                262144 / num_pixels)  # larger frames will be more RAM intensive, scale user batch size based on num pixels in 512x512 images

        if not db:
            db = {
                'data_path': [self.tiff_path],
                'fs': float(sampling_rate),
                'diameter': diameter,
                'batch_size': int(batch_size),
                'nimg_init': int(batch_size),
                'nplanes': self.n_planes
            }

        print(db)

        opsEnd = run_s2p(ops=ops, db=db)

    def cellAreas(self, x=None, y=None):

        '''not sure what this function does'''

        self.cell_area = []

        if x:
            for i, _ in enumerate(self.cell_id):
                if self.cell_med[i][1] < x:
                    self.cell_area.append(0)
                else:
                    self.cell_area.append(1)

        if y:
            for i, _ in enumerate(self.cell_id):
                if self.cell_med[i][1] < y:
                    self.cell_area.append(0)
                else:
                    self.cell_area.append(1)

    def s2pProcessing(self, s2p_path, subset_frames=None, subtract_neuropil=True, baseline_frames=[]):

        """processing of suite2p data on the experimental object"""

        self.cell_id = []
        self.n_units = []
        self.cell_plane = []
        self.cell_med = []
        self.cell_x = []
        self.cell_y = []
        self.raw = []
        self.mean_img = []
        self.radius = []

        if self.n_planes == 1:
            # s2p_path = os.path.join(self.tiff_path, 'suite2p', 'plane' + str(plane))
            FminusFneu, self.spks, self.stat = s2p_loader(s2p_path, subtract_neuropil)  # s2p_loader() is in utils_func
            ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()

            if subset_frames is None:
                self.raw = FminusFneu
            elif subset_frames is not None:
                self.raw = FminusFneu[:, subset_frames[0]:subset_frames[1]]
                self.spks = self.spks[:, subset_frames[0]:subset_frames[1]]
            if len(baseline_frames) > 0:
                self.baseline_raw = FminusFneu[:, baseline_frames[0]:baseline_frames[1]]
            self.mean_img = ops['meanImg']
            cell_id = []
            cell_plane = []
            cell_med = []
            cell_x = []
            cell_y = []
            radius = []

            for cell, s in enumerate(self.stat):
                cell_id.append(s['original_index'])  # stat is an np array of dictionaries!
                cell_med.append(s['med'])
                cell_x.append(s['xpix'])
                cell_y.append(s['ypix'])
                radius.append(s['radius'])

            self.cell_id = cell_id
            self.n_units = len(self.cell_id)
            self.cell_med = cell_med
            self.cell_x = cell_x
            self.cell_y = cell_y
            self.radius = radius

            num_units = FminusFneu.shape[0]

        else:
            for plane in range(self.n_planes):
                # s2p_path = os.path.join(self.tiff_path, 'suite2p', 'plane' + str(plane))
                FminusFneu, self.spks, self.stat = s2p_loader(s2p_path,
                                                              subtract_neuropil)  # s2p_loader() is in utils_func
                ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()

                self.raw.append(FminusFneu)
                self.mean_img.append(ops['meanImg'])
                cell_id = []
                cell_plane = []
                cell_med = []
                cell_x = []
                cell_y = []
                radius = []

                for cell, s in enumerate(self.stat):
                    cell_id.append(s['original_index'])  # stat is an np array of dictionaries!
                    cell_med.append(s['med'])
                    cell_x.append(s['xpix'])
                    cell_y.append(s['ypix'])
                    radius.append(s['radius'])

                self.cell_id.append(cell_id)
                self.n_units.append(len(self.cell_id[plane]))
                self.cell_med.append(cell_med)
                self.cell_x.append(cell_x)
                self.cell_y.append(cell_y)
                self.radius = radius

                num_units = FminusFneu.shape[0]
                cell_plane.extend([plane] * num_units)
                self.cell_plane.append(cell_plane)

    def _parseNAPARMxml(self):

        print('\n-----parsing Naparm xml file...')

        print('loading NAPARM_xml_path:')
        NAPARM_xml_path = pjf.path_finder(self.naparm_path, '.xml')[0]

        xml_tree = ET.parse(NAPARM_xml_path)
        root = xml_tree.getroot()

        title = root.get('Name')
        n_trials = int(root.get('Iterations'))

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

        NAPARM_gpl_path = pjf.path_finder(self.naparm_path, '.gpl')[0]
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
                print('Spiral size (um):', elem.get('SpiralSize'))
                break

        self.spiral_size = int(spiral_size)
        # self.single_stim_dur = single_stim_dur  # not sure why this was previously getting this value from here, but I'm now getting it from the xml file above

    def paqProcessing(self):

        print('\nloading', self.paq_path)

        paq, _ = paq_read(self.paq_path, plot=True)
        self.paq_rate = paq['rate']

        # find frame times

        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = pjf.threshold_detect(clock_voltage, 1)
        self.frame_clock = frame_clock
        plt.figure(figsize=(10, 5))
        plt.plot(clock_voltage)
        plt.plot(frame_clock, np.ones(len(frame_clock)), '.')
        sns.despine()
        plt.show()

        # find stim times

        stim_idx = paq['chan_names'].index(self.stim_channel)
        stim_volts = paq['data'][stim_idx, :]
        stim_times = pjf.threshold_detect(stim_volts, 1)
        self.stim_times = stim_times

        # correct this based on txt file
        duration_ms = self.stim_dur
        frame_rate = self.fps / self.n_planes
        duration_frames = np.ceil((duration_ms / 1000) * frame_rate)
        self.duration_frames = int(duration_frames)

        plt.figure(figsize=(10, 5))
        plt.plot(stim_volts)
        plt.plot(stim_times, np.ones(len(stim_times)), '.')
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

            self.stim_start_frames = np.array(stim_start_frames)
            # self.stim_start_frames.append(np.array(stim_start_frames))  # recoded with slight improvement

            # # sanity check
            # assert max(self.stim_start_frames[0]) < self.raw[plane].shape[1] * self.n_planes

        # find voltage channel and save as lfp_signal attribute
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

        self.paqProcessing()

    def stimProcessing(self, stim_channel):

        self.stim_channel = stim_channel

        if self.stim_type == '2pstim':
            self.photostimProcessing()

    def cellStaProcessing(self, test='t_test'):

        if self.stim_start_frames:

            # this is the key parameter for the sta, how many frames before and after the stim onset do you want to use
            self.pre_frames = int(np.ceil(self.fps * 0.5))  # 500 ms pre-stim period
            self.post_frames = int(np.ceil(self.fps * 3))  # 3000 ms post-stim period

            # list of cell pixel intensity values during each stim on each trial
            self.all_trials = []  # list 1 = cells, list 2 = trials, list 3 = dff vector

            # the average of every trial
            self.stas = []  # list 1 = cells, list 2 = sta vector

            self.all_amplitudes = []
            self.sta_amplitudes = []

            self.t_tests = []
            self.wilcoxons = []

            for plane in range(self.n_planes):

                all_trials = []  # list 1 = cells, list 2 = trials, list 3 = dff vector

                stas = []  # list 1 = cells, list 2 = sta vector

                all_amplitudes = []
                sta_amplitudes = []

                t_tests = []
                wilcoxons = []

                # loop through each cell
                for i, unit in enumerate(self.raw[plane]):

                    trials = []
                    amplitudes = []
                    df = []

                    # a flat list of all observations before stim occured
                    pre_obs = []
                    # a flat list of all observations after stim occured
                    post_obs = []

                    for stim in self.stim_start_frames[plane]:
                        # get baseline values from pre_stim
                        pre_stim_f = unit[stim - self.pre_frames: stim]
                        baseline = np.mean(pre_stim_f)

                        # the whole trial and dfof using baseline
                        trial = unit[stim - self.pre_frames: stim + self.post_frames]
                        trial = [((f - baseline) / baseline) * 100 for f in trial]  # dff calc
                        trials.append(trial)

                        # calc amplitude of response
                        pre_f = trial[: self.pre_frames - 1]
                        pre_f = np.mean(pre_f)

                        avg_post_start = self.pre_frames + (self.duration_frames + 1)
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

    def _findTargets(self):

        '''
        Finds cells that have been targeted for optogenetic photostimulation using Naparm in all-optical type experiments.
        output: coordinates of targets, and circular areas of targets
        Note this is not done by target groups however. So all of the targets are just in one big list.
        '''

        print('\n-----Loading up target coordinates...')

        self.n_targets = []
        self.target_coords = []
        self.target_areas = []

        # load naparm targets file for this experiment
        naparm_path = os.path.join(self.naparm_path, 'Targets')

        listdir = os.listdir(naparm_path)

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

        target_image_scaled = target_image;
        del target_image
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
        #     #             image_8bit = convert_to_8bit(target_image_scaled, np.unit8)
        #     #             tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), image_8bit)
        #     tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)

        targets = np.where(target_image_scaled > 0)
        # targets_1 = np.where(target_image_scaled_1 > 0)
        # targets_2 = np.where(target_image_scaled_2 > 0)

        targetCoordinates = list(zip(targets[1], targets[0]))
        print('Number of targets:', len(targetCoordinates))

        # targetCoordinates_1 = list(zip(targets_1[1], targets_1[0]))
        # print('Number of targets, SLM group #1:', len(targetCoordinates_1))
        #
        # targetCoordinates_2 = list(zip(targets_2[1], targets_2[0]))
        # print('Number of targets, SLM group #2:', len(targetCoordinates_2))

        self.target_coords = targetCoordinates
        # self.target_coords_1 = targetCoordinates_1
        # self.target_coords_2 = targetCoordinates_2
        self.n_targets_total = len(targetCoordinates)

        radius = self.spiral_size / self.pix_sz_x

        target_areas = []
        for coord in targetCoordinates:
            target_area = ([item for item in points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
            target_areas.append(target_area)
        self.target_areas = target_areas

        # # get areas for SLM group #1
        # target_areas_1 = []
        # for coord in targetCoordinates_1:
        #     target_area = ([item for item in points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
        #     target_areas_1.append(target_area)
        # self.target_areas_1 = target_areas_1
        #
        # # get areas for SLM group #2
        # target_areas_2 = []
        # for coord in targetCoordinates_2:
        #     target_area = ([item for item in points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
        #     target_areas_2.append(target_area)
        # self.target_areas_2 = target_areas_2

        print('Got targets...')

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
                target_area = ([item for item in points_in_circle_np(radius, x0=target[0], y0=target[1])])
                a.append(target_area)
            target_areas.append(a)

        self.target_areas = target_areas

        print('Got targets...')

    def s2p_targets(self):
        '''finding s2p cell ROIs that were also SLM targets'''

        self.is_target = []

        print('\nsearching for targeted cells...')

        # # Rob's new version of this, not really sure how he meant for it to be done, but it's not working
        # targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
        # target_areas = np.array(self.target_areas)
        # targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1
        #
        # cell_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
        # cell_x = np.array(self.cell_x)
        # cell_y = np.array(self.cell_y)
        # for i, coord in enumerate(zip(cell_x[0], cell_y[0])):
        #     cell_img[coord] = i + 1
        #
        # targ_cell = cell_img * targ_img
        #
        # targ_cell_ids = np.unique(targ_cell)[1:] - 1
        #
        # self.is_target = np.zeros([self.n_units], dtype='bool')
        # self.is_target[targ_cell_ids] = True
        #
        # self.n_targeted_cells = np.sum(self.is_target)
        #
        # self.s2p_cell_targets = [i for i, x in enumerate(self.is_target) if
        #                          x == True]  # get list of s2p cells that were photostim targetted

        ##### new version
        # s2p targets for all SLM targets
        for cell in range(self.n_units):
            flag = 0
            for x, y in zip(self.cell_x[cell], self.cell_y[cell]):
                if (x, y) in self.target_coords:
                    print((x, y))
                    flag = 1
                elif (x, y) in self.target_coords_all:
                    print((x, y))
                    flag = 1

            if flag == 1:
                self.is_target.append(1)
            else:
                self.is_target.append(0)

        self.n_targeted_cells = sum(self.is_target)
        self.s2p_cell_targets = [self.cell_id[i] for i, x in enumerate(self.is_target) if
                                 x == 1]  # get list of s2p cells that were photostim targetted

        # # s2p targets for SLM group #1
        # self.targeted_cells_1 = []
        # for cell in range(self.n_units):
        #     flag = 0
        #     for x, y in zip(self.cell_x[cell], self.cell_y[cell]):
        #         if (x, y) in self.target_coords_1:
        #             print('Target coordinate found (Group #1)', (x, y))
        #             flag = 1
        #
        #     if flag == 1:
        #         self.targeted_cells_1.append(1)
        #     else:
        #         self.targeted_cells_1.append(0)
        #
        # self.n_targeted_cells_1 = sum(self.targeted_cells_1)
        # self.s2p_cell_targets_1 = [self.cell_id[i] for i, x in enumerate(self.targeted_cells_1) if
        #                            x == 1]  # get list of s2p cells that were photostim targetted
        #
        # # s2p targets for SLM group #2
        # self.targeted_cells_2 = []
        # for cell in range(self.n_units):
        #     flag = 0
        #     for x, y in zip(self.cell_x[cell], self.cell_y[cell]):
        #         if (x, y) in self.target_coords_2:
        #             print('Target coordinate found (Group #2)', (x, y))
        #             flag = 1
        #
        #     if flag == 1:
        #         self.targeted_cells_2.append(1)
        #     else:
        #         self.targeted_cells_2.append(0)
        #
        # self.n_targeted_cells_2 = sum(self.targeted_cells_2)
        # self.s2p_cell_targets_2 = [self.cell_id[i] for i, x in enumerate(self.targeted_cells_2) if
        #                            x == 1]  # get list of s2p cells that were photostim targetted

        # ##### old version - pretty sure something is wrong with this code, the s2p cell targets this code finds don't make much sense
        # print('Searching for targeted cells in suite2p results...')
        # for cell in range(self.n_units):
        #     flag = 0
        #
        #     for x, y in zip(self.cell_x[cell], self.cell_y[cell]):
        #         for target in range(self.n_targets):
        #             for a, b in self.target_areas[target]:
        #                 if x == a and y == b:
        #                     flag = 1
        #
        #     if flag == 1:
        #         self.is_target.append(1)
        #     else:
        #         self.is_target.append(0)
        #
        # self.s2p_cell_targets = [i for i, x in enumerate(self.is_target) if
        #                     x == 1]  # get list of s2p cells that were photostim targetted
        #
        # self.n_targeted_cells = len(self.s2p_cell_targets)

        # self.s2p_cell_targets_groups = [self.s2p_cell_targets_1, self.s2p_cell_targets_2]

        print('------- Search completed.')
        print('Number of targeted cells: ', self.n_targeted_cells)
        print('Target cells found in suite2p: ', self.s2p_cell_targets, ' -- %s cells' % len(self.s2p_cell_targets))
        # print('Target cells SLM Group #1: ', self.s2p_cell_targets_1)
        # print('Target cells SLM Group #2: ', self.s2p_cell_targets_2)
        #

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
                                x == 1]  # get list of s2p cells that were photostim targetted
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
                           self.duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
                frames_to_remove.append(j + i)

        im_stack_1 = np.delete(im_stack, frames_to_remove, axis=0)

        tf.imwrite(save_as, im_stack_1, photometric='minisblack')

    def find_photostim_frames(self):
        '''finds all photostim frames and saves them into the bad_frames attribute for the exp object'''
        print('\n-----calculating photostimulation frames and adding to bad_frames.npy file...')
        print('# of photostim frames calculated per stim. trial: ', self.duration_frames + 1)

        photostim_frames = []
        for j in self.stim_start_frames:
            for i in range(
                    self.duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
                photostim_frames.append(j + i)

        self.photostim_frames = photostim_frames
        # print(photostim_frames)
        print('|\n -- Original # of frames:', self.n_frames, 'frames ///')
        print('|\n -- # of Photostim frames:', len(photostim_frames), 'frames ///')
        print('|\n -- Minus photostim. frames total:', self.n_frames - len(photostim_frames), 'frames ///')
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

    def save_pkl(self, pkl_path: str = None):
        if pkl_path is None:
            if hasattr(self, 'pkl_path'):
                pkl_path = self.pkl_path
            else:
                raise ValueError(
                    'pkl path for saving was not found in object attributes, please provide path to save to')
        else:
            self.pkl_path = pkl_path

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self, f)
        print("pkl saved to %s" % pkl_path)

    def save(self):
        self.save_pkl()

    def avg_stim_images(self, peri_frames: int = 100, stim_timings: list = [], save_img=False, to_plot=False):
        """
        Outputs (either by saving or plotting, or both) images from raw t-series TIFF for a trial around each individual
        stim timings.

        :param peri_frames:
        :param stim_timings:
        :param save_img:
        :param to_plot:
        :return:
        """

        if hasattr(self, 'stim_images'):
            x = [0 for stim in stim_timings if stim not in self.stim_images.keys()]
        else:
            self.stim_images = {}
            x = [0] * len(stim_timings)
        if 0 in x:
            tiffs_loc = '%s/*Ch3.tif' % self.tiff_path_dir
            tiff_path = glob.glob(tiffs_loc)[0]
            print('loading up %s tiff from: ' % self.metainfo['trial'], tiff_path)
            im_stack = tf.imread(tiff_path, key=range(self.n_frames))
            print('Processing seizures from experiment tiff (wait for all seizure comparisons to be processed), \n '
                  'total tiff shape: ', im_stack.shape)

        for stim in stim_timings:
            if stim in self.stim_images.keys():
                avg_sub = self.stim_images[stim]
            else:
                im_sub = im_stack[stim - peri_frames: stim + peri_frames]
                avg_sub = np.mean(im_sub, axis=0)
                self.stim_images[stim] = avg_sub

            if save_img:
                # save in a subdirectory under the ANALYSIS folder path from whence t-series TIFF came from
                save_path = self.tiff_path[:21] + 'Analysis/' + self.tiff_path_dir[26:] + '/avg_stim_images'
                save_path_stim = save_path + '/%s_%s_stim-%s.tif' % (
                    self.metainfo['date'], self.metainfo['trial'], stim)
                if os.path.exists(save_path):
                    print("saving stim_img tiff to... %s" % save_path_stim)
                    avg_sub8 = convert_to_8bit(avg_sub, np.uint8, 0, 255)
                    tf.imwrite(save_path_stim,
                               avg_sub8, photometric='minisblack')
                else:
                    print('made new directory for saving images at:', save_path)
                    os.mkdir(save_path)
                    print("saving as... %s" % save_path_stim)
                    avg_sub8 = convert_to_8bit(avg_sub, np.uint8, 0, 255)
                    tf.imwrite(save_path_stim,
                               avg_sub, photometric='minisblack')

            if to_plot:
                plt.imshow(avg_sub, cmap='gray')
                plt.suptitle('avg image from %s frames around stim_start_frame %s' % (peri_frames, stim))
                plt.show()  # just plot for now to make sure that you are doing things correctly so far

        if hasattr(self, 'pkl_path'):
            self.save_pkl()
        else:
            print('note: pkl not saved yet...')

    # def _good_cells(self, min_radius_pix, max_radius_pix):
    #     '''
    #     This function filters each cell for two criteria. 1) at least 1 flu change greater than 2.5std above mean,
    #     and 2) a minimum cell radius (in pixels) of the given value.
    #
    #     :param min_radius_pix:
    #     :return: a list of
    #     '''
    #
    #     good_cells = []
    #     for i in range(len(self.cell_id)):
    #         raw = self.raw[i]
    #         raw_ = list(np.delete(raw, self.photostim_frames, None))
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
    #         raw_ = list(np.delete(raw, self.photostim_frames, None))
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
    @njit
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


## this should technically be the biggest super class lol
class twopimaging():

    def __init__(self, tiff_path_dir, paq_path, suite2p_path=None, suite2p_run=False):
        '''

        :param paths: list of key paths (tiff_loc
        :param suite2p_path: path to the suite2p outputs (plan0 file? or ops file? not sure yet)
        :param suite2p_run: set to true if suite2p is already run for this trial
        '''

        self.tiff_path = tiff_path_dir
        self.paq_path = paq_path

        self._parsePVMetadata()

        if suite2p_run:
            self.suite2p_path = suite2p_path
            self.s2pProcessing(s2p_path=self.suite2p_path)

    def s2pProcessing(self, s2p_path, subtract_neuropil=True):

        '''processing of suite2p data on the experimental object?'''

        self.cell_id = []
        self.n_units = []
        self.cell_plane = []
        self.cell_med = []
        self.cell_x = []
        self.cell_y = []
        self.raw = []
        self.mean_img = []
        self.radius = []

        # s2p_path = os.path.join(self.tiff_path, 'suite2p', 'plane' + str(plane))
        FminusFneu, spks, stat = s2p_loader(s2p_path, subtract_neuropil)  # s2p_loader() is in utils_func
        ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()

        self.raw = FminusFneu
        self.spks = spks
        self.stat = stat
        self.mean_img = ops['meanImg']
        cell_id = []
        cell_plane = []
        cell_med = []
        cell_x = []
        cell_y = []
        radius = []

        for cell, s in enumerate(stat):
            cell_id.append(s['original_index'])  # stat is an np array of dictionaries!
            cell_med.append(s['med'])
            cell_x.append(s['xpix'])
            cell_y.append(s['ypix'])
            radius.append(s['radius'])

        self.cell_id = cell_id
        self.n_units = len(self.cell_id)
        self.cell_med = cell_med
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.radius = radius

    def _getPVStateShard(self, path, key):

        '''
        Used in function PV metadata below
        '''

        value = []
        description = []
        index = []

        xml_tree = ET.parse(path)  # parse xml from a path
        root = xml_tree.getroot()  # make xml tree structure

        pv_state_shard = root.find('PVStateShard')  # find pv state shard element in root

        for elem in pv_state_shard:  # for each element in pv state shard, find the value for the specified key

            if elem.get('key') == key:

                if len(elem) == 0:  # if the element has only one subelement
                    value = elem.get('value')
                    break

                else:  # if the element has many subelements (i.e. lots of entries for that key)
                    for subelem in elem:
                        value.append(subelem.get('value'))
                        description.append(subelem.get('description'))
                        index.append(subelem.get('index'))
            else:
                for subelem in elem:  # if key not in element, try subelements
                    if subelem.get('key') == key:
                        value = elem.get('value')
                        break

            if value:  # if found key in subelement, break the loop
                break

        if not value:  # if no value found at all, raise exception
            raise Exception('ERROR: no element or subelement with that key')

        return value, description, index

    def _parsePVMetadata(self):

        tiff_path = self.tiff_path
        path = []

        try:  # look for xml file in path, or two paths up (achieved by decreasing count in while loop)
            count = 2
            while count != 0 and not path:
                count -= 1
                for file in os.listdir(tiff_path):
                    if file.endswith('.xml'):
                        path = os.path.join(tiff_path, file)
                    if file.endswith('.env'):
                        env_path = os.path.join(tiff_path, file)
                tiff_path = os.path.dirname(tiff_path)

        except:
            raise Exception('ERROR: Could not find xml for this acquisition, check it exists in %s' % tiff_path)

        xml_tree = ET.parse(path)  # parse xml from a path
        root = xml_tree.getroot()  # make xml tree structure

        sequence = root.find('Sequence')
        acq_type = sequence.get('type')

        if 'ZSeries' in acq_type:
            n_planes = len(sequence.findall('Frame'))

        else:
            n_planes = 1

        frame_period = float(self._getPVStateShard(path, 'framePeriod')[0])
        fps = 1 / frame_period

        frame_x = int(self._getPVStateShard(path, 'pixelsPerLine')[0])

        frame_y = int(self._getPVStateShard(path, 'linesPerFrame')[0])

        zoom = float(self._getPVStateShard(path, 'opticalZoom')[0])

        scanVolts, _, index = self._getPVStateShard(path, 'currentScanCenter')
        for scanVolts, index in zip(scanVolts, index):
            if index == 'XAxis':
                scan_x = float(scanVolts)
            if index == 'YAxis':
                scan_y = float(scanVolts)

        pixelSize, _, index = self._getPVStateShard(path, 'micronsPerPixel')
        for pixelSize, index in zip(pixelSize, index):
            if index == 'XAxis':
                pix_sz_x = float(pixelSize)
            if index == 'YAxis':
                pix_sz_y = float(pixelSize)

        env_tree = ET.parse(env_path)
        env_root = env_tree.getroot()

        elem_list = env_root.find('TSeries')
        # n_frames = elem_list[0].get('repetitions') # Rob would get the n_frames from env file
        # change this to getting the last actual index from the xml file

        n_frames = root.findall('Sequence/Frame')[-1].get('index')

        self.fps = fps
        self.frame_x = frame_x
        self.frame_y = frame_y
        self.n_planes = n_planes
        self.pix_sz_x = pix_sz_x
        self.pix_sz_y = pix_sz_y
        self.scan_x = scan_x
        self.scan_y = scan_y
        self.zoom = zoom
        self.n_frames = int(n_frames)

        print('n planes:', n_planes,
              '\nn frames:', int(n_frames),
              '\nfps:', fps,
              '\nframe size (px):', frame_x, 'x', frame_y,
              '\nzoom:', zoom,
              '\npixel size (um):', pix_sz_x, pix_sz_y,
              '\nscan centre (V):', scan_x, scan_y
              )


class Post4ap(alloptical):
    # TODO fix the superclass definitions and heirarchy which might require more rejigging of the code you are running later on as well

    def __init__(self, paths, metainfo, stimtype):
        alloptical.__init__(self, paths, metainfo, stimtype)
        print('\ninitialized Post4ap expobj of exptype and trial: %s, %s, %s' % (self.metainfo['exptype'],
                                                                                 self.metainfo['trial'],
                                                                                 self.metainfo['date']))

    def _subselect_sz_tiffs(self, onsets, offsets):
        """subselect raw tiff movie over all seizures as marked by onset and offsets. save under analysis path for object.
        Note that the onsets and offsets definitions may vary, so check exactly what was used in those args."""

        print('\n-----Making raw sz movies by cropping original raw tiff')
        if hasattr(self, 'analysis_save_path'):
            pass
        else:
            raise ValueError(
                'need to add the analysis_save_path attr before using this function -- this is where it will save to')

        print('reading in seizure trial from: ', self.tiff_path, '\n')
        stack = tf.imread(self.tiff_path)

        # subselect raw tiff movie over all seizures as marked by LFP onset and offsets
        for on, off in zip(onsets, offsets):
            select_frames = (on, off);
            print('cropping sz frames', select_frames)
            save_as = self.analysis_save_path + '/%s_%s_subselected_%s_%s.tif' % (self.metainfo['date'],
                                                                                  self.metainfo['trial'],
                                                                                  select_frames[0], select_frames[1])
            subselect_tiff(tiff_stack=stack, select_frames=select_frames, save_as=save_as)
        print('\ndone. saved to:', self.analysis_save_path)

    def collect_seizures_info(self, seizures_info_array=None, discard_all=True):
        print('\ncollecting information about seizures...')
        self.seizures_info_array = seizures_info_array  # path to the matlab array containing paired measurements of seizures onset and offsets

        # retrieve seizure onset and offset times from the seizures info array input
        paq = paq_read(file_path=self.paq_path, plot=False)

        # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
        # NOTE: the output of all of the following function is in dimensions of the FRAME CLOCK (not official paq clock time)
        if seizures_info_array is not None:
            print('-- using matlab array to collect seizures %s: ' % seizures_info_array)
            bad_frames, self.seizure_frames, self.seizure_lfp_onsets, self.seizure_lfp_offsets = frames_discard(
                paq=paq[0], input_array=seizures_info_array, total_frames=self.n_frames, discard_all=discard_all)
        else:
            print('-- no matlab array given to use for finding seizures.')
            bad_frames = frames_discard(paq=paq[0], input_array=seizures_info_array, total_frames=self.n_frames,
                                        discard_all=discard_all)

        print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
        print('|\n -- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])
        self.append_bad_frames(
            bad_frames=bad_frames)  # here only need to append the bad frames to the expobj.bad_frames property

        if len(self.bad_frames) > 0:
            np.save('%s/bad_frames.npy' % self.tiff_path[:-35],
                    self.bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p
            print('***Saving a total of ', len(self.bad_frames),
                  'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy***')

        print('now creating raw movies for each sz as well (saved to the /Analysis folder')
        self._subselect_sz_tiffs(onsets=self.seizure_lfp_onsets, offsets=self.seizure_lfp_offsets)

    def find_closest_sz_frames(self):
        """finds time from the closest seizure onset on LFP (-ve values for forthcoming, +ve for past)
        FOR each photostim timepoint"""

        self.closest_sz = {'stim': [], 'closest sz on (frames)': [], 'closest sz off (frames)': [],
                           'closest sz (instance)': []}
        for stim in self.stim_start_frames:
            differences_on = stim - self.seizure_lfp_onsets
            differences_off = stim - self.seizure_lfp_offsets

            # some math to figure out the closest seizure on and off frames from the list of sz LFP stamps and current stim time
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

    def avg_seizure_images(self, baseline_tiff: str = '', frames_last: int = 0):
        """
        used to make averaged images of all seizures contained within an individual expobj trial. the averaged images
        are also subtracted from baseline_tiff image to give a difference image that should highlight the seizure well.

        :param baseline_tiff: path to the baseline tiff file to use
        :param frames_last: use to specify the tail of the seizure frames for images.
        :return:
        """
        if baseline_tiff == '':
            raise Exception(
                'please provide a baseline tiff path to use for this trial -- usually the spont imaging trials of the same experiment')

        print('First loading up and plotting baseline (comparison) tiff from: ', baseline_tiff)
        im_stack_base = tf.imread(baseline_tiff, key=range(5000))  # reading in just the first 5000 frames of the spont
        avg_baseline = np.mean(im_stack_base, axis=0)
        plt.imshow(avg_baseline, cmap='gray')
        plt.suptitle('avg 5000 frames baseline from %s' % baseline_tiff[-35:])
        plt.show()

        tiffs_loc = '%s/*Ch3.tif' % self.tiff_path_dir
        tiff_path = glob.glob(tiffs_loc)[0]
        print('loading up post4ap tiff from: ', tiff_path)
        im_stack = tf.imread(tiff_path, key=range(self.n_frames))
        print('Processing seizures from experiment tiff (wait for all seizure comparisons to be processed), \n '
              'total tiff shape: ', im_stack.shape)
        avg_sub_l = []
        im_sub_l = []
        im_diff_l = []
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

            im_diff = avg_sub - avg_baseline
            plt.imshow(im_diff, cmap='gray')
            plt.suptitle('diff of seizure from %s to %s frames' % (sz_on, sz_off))
            plt.show()  # just plot for now to make sure that you are doing things correctly so far

            avg_sub_l.append(avg_sub)
            im_sub_l.append(im_sub)
            im_diff_l.append(im_diff)

            # how to calculate the dominant direction? do you need to look at the seizure throughout its whole length
            # or can you just take the mean image of the seizure duration and then use that mean img to make the dominant direction measurement

        return avg_sub_l, im_sub_l, im_diff_l

    def _InOutSz(self, cell, cell_med: list, sz_border_path: str, to_plot=False):
        """
        Returns True if the given cell's location is inside the seizure boundary which is defined as the coordinates
        given in the .csv sheet.

        :param cell_med: from stat['med'] of the cell
        :param sz_border_path: path to the csv file generated by ImageJ macro for the seizure boundary
        :param to_plot: make plot showing the boundary start, end and the location of the cell in question
        :return: bool

        # examples
        cell_med = expobj.stat[0]['med']
        sz_border_path = "/home/pshah/mnt/qnap/Analysis/2020-12-18/2020-12-18_t-013/boundary_csv/2020-12-18_t-013_post 4ap all optical trial_stim-9222.tif_border.csv"
        InOutSz(cell_med, sz_border_path)
        """

        y = cell_med[0]
        x = cell_med[1]

        # for path in os.listdir(sz_border_path):
        #     if all(s in path for s in ['.csv', self.sheet_name]):
        #         csv_path = os.path.join(sz_border_path, path)

        xline = []
        yline = []
        with open(sz_border_path) as csv_file:
            csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
            for row in csv_file:
                xline.append(int(float(row['xcoords'])))
                yline.append(int(float(row['ycoords'])))

        # assumption = line is monotonic
        line_argsort = np.argsort(yline)
        xline = np.array(xline)[line_argsort]
        yline = np.array(yline)[line_argsort]

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
        #     # pjf.plot_cell_loc(self, cells=[cell], show=False)
        #     plt.scatter(x=xline[0], y=yline[0])
        #     plt.scatter(x=xline[1], y=yline[1])
        #     # plt.show()

        if np.sign(d) == np.sign(ds1):
            return True
        elif np.sign(d) == np.sign(ds2):
            return False
        else:
            return False

    def classify_cells_sz(self, sz_border_path, to_plot=True, title=None, flip=False):
        """
        going to use Rob's suggestions to define boundary of the seizure in ImageJ and then read in the ImageJ output,
        and use this to classify cells as in seizure or out of seizure in a particular image (which will relate to stim time).

        :param sz_border_path: str; path to the .csv containing the points specifying the seizure border for a particular stim image
        :param to_plot: make plot showing the boundary start, end and the location of the cell in question
        :param title:
        :param flip: use True if the seizure orientation is from bottom right to top left.
        :return in_sz = list; containing the cell_ids of cells that are classified inside the seizure area
        """

        in_sz = []
        out_sz = []
        for cell, s in enumerate(self.stat):
            x = self._InOutSz(cell=cell, cell_med=s['med'], sz_border_path=sz_border_path, to_plot=to_plot)

            if x is True:
                in_sz.append(cell)
            elif x is False:
                out_sz.append(cell)

            if to_plot:  # plot the sz boundary points
                xline = []
                yline = []
                with open(sz_border_path) as csv_file:
                    csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
                    for row in csv_file:
                        xline.append(int(float(row['xcoords'])))
                        yline.append(int(float(row['ycoords'])))
                # assumption = line is monotonic
                line_argsort = np.argsort(yline)
                xline = np.array(xline)[line_argsort]
                yline = np.array(yline)[line_argsort]

                # pjf.plot_cell_loc(self, cells=[cell], show=False)
                plt.scatter(x=xline[0], y=yline[0])
                plt.scatter(x=xline[1], y=yline[1])
                # plt.show()

        if flip:
            # pass
            in_sz_2 = in_sz
            in_sz = out_sz
            out_sz = in_sz_2

        if to_plot:
            pjf.plot_cell_loc(self, cells=in_sz, title=title)
            # plt.show()  # the indiviual cells were plotted in ._InOutSz

        return in_sz


## Rob's functions for generating some important commonly used image types.
def s2pMeanImage(s2p_path):
    os.chdir(s2p_path)

    ops = np.load('ops.npy', allow_pickle=True).item()

    mean_img = ops['meanImg']

    mean_img = np.array(mean_img, dtype='uint16')

    return mean_img


def s2pMasks(obj, s2p_path, cell_ids):
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
        if n in obj.s2p_cell_targets:
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


def s2pMaskStack(obj, pkl_list, s2p_path, parent_folder):
    '''makes a TIFF stack with the s2p mean image, and then suite2p ROI masks for all cells detected, target cells, and also SLM targets as well?'''

    for pkl in pkl_list:
        expobj = obj

        print('Retrieving s2p masks for:', pkl, '             ', end='\r')

        # with open(pkl, 'rb') as f:
        #     expobj = pickle.load(f)

        # list of cell ids to filter s2p masks by
        # cell_id_list = [list(range(1, 99999)),  # all
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
        mean_img = s2pMeanImage(s2p_path)
        mean_img = np.expand_dims(mean_img, axis=0)
        stack = np.append(stack, mean_img, axis=0)

        # mask images from s2p
        mask_img, targets_s2p_img = s2pMasks(obj=expobj, s2p_path=s2p_path, cell_ids=cell_ids)
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
        targ_img = getTargetImage(expobj)
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
        print('\ns2p ROI + photostim targets masks saved to: ', save_path)


# other functions written by me

# TODO need to transfer alot of these functions to methods

def save_pkl(expobj, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("pkl saved to %s" % pkl_path)


# PRE-PROCESSING FUNCTIONS
@njit
def moving_average(a, n=4):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


@njit
def _good_cells(cell_ids, raws, photostim_frames, radiuses, std_thresh, min_radius_pix, max_radius_pix):
    good_cells = []
    len_cell_ids = len(cell_ids)
    for i in range(len_cell_ids):

        # print(i, " out of ", len(cell_ids), " cells")
        raw = raws[i]
        raw_ = np.delete(raw, photostim_frames)
        raw_dff = normalize_dff_jit(raw_)  # note that this function is defined in this file a little further down
        std_ = raw_dff.std()

        raw_dff_ = moving_average(raw_dff, n=4)

        thr = np.mean(raw_dff) + std_thresh * std_
        e = np.where(raw_dff_ > thr)
        # y = raw_dff_[e]

        radius = radiuses[i]

        if len(e[0]) > 0 and radius > min_radius_pix and radius < max_radius_pix:
            good_cells.append(cell_ids[i])

        if i % 100 == 0:  # print the progress once every 100 cell iterations
            print(i, " out of ", len_cell_ids, " cells done")

        # if i == 465:  # just using this here if ever need to check back with specific cells if function seems to be misbehaving
        #     print(e, len(e[0]), thr)

    print('# of good cells found: ', len(good_cells), ' (out of ', len_cell_ids, ' ROIs)')
    return good_cells


def get_targets_stim_traces_norm(expobj, normalize_to='', pre_stim=10, post_stim=200):
    """
    primary function to measure the dFF traces for photostimulated targets.
    :param expobj:
    :param normalize_to: str; either "baseline" or "pre-stim"
    :param pre_stim: number of frames to use as pre-stim
    :param post_stim: number of frames to use as post-stim
    :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
    """
    stim_timings = expobj.stim_start_frames
    targeted_cells = [cell for cell in expobj.s2p_cell_targets if cell in expobj.good_cells]

    # collect photostim timed average dff traces of photostim targets
    targets_dff = []
    targets_dff_avg = []

    targets_dfstdF = []
    targets_dfstdF_avg = []

    targets_raw = []
    targets_raw_avg = []
    for cell in targeted_cells:
        # print('considering cell # %s' % cell)
        if cell in expobj.cell_id:
            cell_idx = expobj.cell_id.index(cell)
            flu = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
                   stim not in expobj.seizure_frames]

            flu_dfstdF = []
            flu_dff = []
            if normalize_to == 'baseline':
                mean_spont_baseline = np.mean(expobj.baseline_raw[cell_idx])
                for trace in flu:
                    trace_dff = ((trace - mean_spont_baseline) / mean_spont_baseline) * 100
                    flu_dff.append(trace_dff)
            elif normalize_to == 'pre-stim':
                for trace in flu:
                    mean_pre = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean_pre) / mean_pre) * 100

                    std_pre = np.std(trace[0:pre_stim])
                    dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                    flu_dfstdF.append(dFstdF)
                    flu_dff.append(trace_dff)
            else:
                TypeError('need to specify what to normalize to in get_targets_dFF (choose "baseline" or "pre-stim")')

            targets_dff.append(flu_dff)  # contains all individual dFF traces for all stim times
            targets_dff_avg.append(np.mean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

            targets_dfstdF.append(flu_dfstdF)
            targets_dfstdF_avg.append(np.mean(flu_dfstdF, axis=0))

            targets_raw.append(flu)
            targets_raw_avg.append(np.mean(flu, axis=0))

    if normalize_to == 'baseline':
        return targets_dff, targets_dff_avg
    elif normalize_to == 'pre-stim':
        return targets_dff, targets_dff_avg, targets_dfstdF, targets_dfstdF_avg, targets_raw, targets_raw_avg


def _good_photostim_cells(expobj, std_thresh=1, dff_threshold=None, pre_stim=10,
                          post_stim=200, to_plot=None, use_raw=False):
    '''
    make sure to specify std threshold to use for filtering
    the pre-stim and post-stim args specify which pre-stim and post-stim frames to consider for filtering
    '''
    expobj.good_photostim_cells = []
    expobj.good_photostim_cells_responses = []
    expobj.good_photostim_cells_stim_responses_dF_stdF = []
    expobj.good_photostim_cells_stim_responses_dFF = []
    total = 0  # use to tally up how many cells across all groups are filtered in
    total_considered = 0  # use to tally up how many cells were looked at for their photostim response.

    stim_timings = expobj.stim_start_frames
    targeted_cells = [cell for cell in expobj.s2p_cell_targets if cell in expobj.good_cells]

    # SELECT PHOTOSTIMULATION TARGET CELLS WHO FIRE >1*std ABOVE PRE-STIM MEAN dF
    good_photostim_responses = {}
    good_photostim_cells = []
    good_targets_dF_stdF = []
    good_targets_dff = []
    std_thresh = std_thresh
    if to_plot is not None:
        fig = plt.figure(figsize=(5, 15))
        axes = fig.subplots(5, 1)

    for cell in targeted_cells:
        if use_raw:  ## CHANGING A LOT OF THINGS HERE BE CAREFUL!!!!!!
            trace = expobj.targets_raw_avg[
                targeted_cells.index(cell)]  # trace = averaged raw trace across all photostims. for this cell
            x_ = 'raw'
        else:
            trace = expobj.targets_dff_avg[
                targeted_cells.index(cell)]  # trace = averaged dff trace across all photostims. for this cell
            x_ = 'dFF'
        pre_stim_trace = trace[:pre_stim]
        mean_pre = np.mean(pre_stim_trace)
        std_pre = np.std(pre_stim_trace)
        # mean_post = np.mean(post_stim_trace[:10])
        dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
        # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
        response = np.mean(trace[
                           pre_stim + expobj.duration_frames:pre_stim + 3 * expobj.duration_frames])  # calculate the dF over pre-stim mean F response within the response window

        if to_plot is not None:
            if cell in targeted_cells[:to_plot]:
                idx = targeted_cells[:to_plot].index(cell)
                axes[idx].plot(trace)
                axes[idx].axhspan(mean_pre + 0 * std_pre, mean_pre + std_thresh * std_pre, facecolor='0.25')
                axes[idx].axvspan(pre_stim + expobj.duration_frames, pre_stim + 3 * expobj.duration_frames,
                                  facecolor='0.25')
                axes[idx].title.set_text('Average trace (%s) across all photostims - cell #%s' % (x_, cell))

        # post_stim_trace = trace[pre_stim + expobj.duration_frames:post_stim]
        if dff_threshold is None:
            thresh_ = mean_pre + std_thresh * std_pre
        else:
            thresh_ = mean_pre + dff_threshold  # need to triple check before using
        if response > thresh_:  # test if the response passes threshold
            good_photostim_responses[cell] = response
            good_photostim_cells.append(cell)
            good_targets_dF_stdF.append(dF_stdF)
            good_targets_dff.append(trace)
            print('Cell #%s - %s post-stim: %s (threshold value = %s)' % (cell, x_, response, thresh_))

    if to_plot is not None:
        plt.show()

    expobj.good_photostim_cells.append(good_photostim_cells)
    expobj.good_photostim_cells_responses.append(good_photostim_responses)
    expobj.good_photostim_cells_stim_responses_dF_stdF.append(good_targets_dF_stdF)
    expobj.good_photostim_cells_stim_responses_dFF.append(good_targets_dff)

    if dff_threshold:
        print('[dFF threshold of %s percent]' % dff_threshold)
    elif dff_threshold is None:
        print('[std threshold of %s std]' % std_thresh)

    print('\n%s cells out of %s s2p target cells selected above threshold' % (
        len(good_photostim_cells), len(targeted_cells)))
    total += len(good_photostim_cells)
    total_considered += len(targeted_cells)

    expobj.good_photostim_cells_all = [y for x in expobj.good_photostim_cells for y in x]
    print('Total number of good photostim responsive cells found: %s (out of %s s2p photostim target cells)' % (
        total, total_considered))


def normalize_dff_baseline(arr, baseline_array):
    """normalize given array (cells x time) to the mean of the spont baseline value for each cell.
    :param arr: pandas df or 1-dimensional array
    :param baseline_array: pandas df or 1-dimensionary array
    """

    if arr.ndim == 1:
        mean_ = abs(baseline_array.mean())
        new_array = (arr - mean_) / mean_ * 100
        return new_array
    elif len(arr) > 1:
        new_array = np.empty_like(arr)
        new_array_df = pd.DataFrame(new_array, columns=arr.columns, index=arr.index)
        for i in arr.index:
            mean_ = baseline_array.loc[str(i)].mean()
            new_array_df.loc[str(i)] = (arr.loc[str(i)] - mean_) / mean_ * 100
        return new_array_df


def normalize_dff(arr, threshold=20):
    """normalize given array (cells x time) to the mean of the fluorescence values below given threshold"""

    if arr.ndim == 1:
        a = np.percentile(arr, threshold)
        mean_ = abs(arr[arr < a].mean())
        new_array = (arr - mean_) / mean_ * 100
        # print(mean)
    else:
        new_array = np.empty_like(arr)
        for i in range(len(arr)):
            a = np.percentile(arr[i], threshold)
            mean_ = np.mean(arr[i][arr[i] < a])
            new_array[i] = arr[i] / abs(mean_) * 100

            if np.isnan(new_array[i]).any() == True:
                print('Warning:')
                print('Cell %d: contains nan' % (i + 1))
                print('      Mean of the sub-threshold for this cell: %s' % mean_)

    return new_array


@jit
def normalize_dff_jit(arr, threshold=20):
    """normalize given array (cells x time) to the mean of the fluorescence values below given threshold"""

    if arr.ndim == 1:
        a = np.percentile(arr, threshold)
        mean_ = abs(arr[arr < a].mean())
        new_array = (arr - mean_) / mean_ * 100
        # print(mean)
    else:
        new_array = np.empty_like(arr)
        for i in range(len(arr)):
            a = np.percentile(arr[i], threshold)
            mean_ = np.mean(arr[i][arr[i] < a])
            new_array[i] = arr[i] / abs(mean_) * 100

            if np.isnan(new_array[i]).any() == True:
                print('Warning:')
                print('Cell %d: contains nan' % (i + 1))
                print('      Mean of the sub-threshold for this cell: %s' % mean_)

    return new_array


def make_tiff_stack(tiff_paths, save_as=''):
    """
    read in a bunch of tiffs and stack them together, and save the output as the save_as
    - make sure that the save_as variable is a .tif file path to where the tif should be saved
    - make sure to change dtype as necessary
    - make sure that tiff_paths is a glob type path, where you can * for many tiffs
    e.g.: '/home/pshah/mnt/qnap/Data/2020-02-25/t05/*.tif'
    """

    sorted_paths = sorted(glob.glob(tiff_paths))

    num_frames = len(sorted_paths)

    with tf.TiffWriter(save_as, bigtiff=True) as tif:
        for i, frame in enumerate(sorted_paths):
            with tf.TiffFile(frame, multifile=False) as input_tif:
                data = input_tif.asarray()
            tif.save(data)
            msg = ' -- Writing frame: ' + str(i + 1) + ' out of ' + str(num_frames)
            print(msg, end='\r')


def convert_to_8bit(img, target_type_min=0, target_type_max=255):
    """
    :param img:
    :param target_type:
    :param target_type_min:
    :param target_type_max:
    :return:
    """
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(np.uint8)
    return new_img


def downsample_tiff(tiff_path, save_as=None):
    print('working on... %s' % tiff_path)

    # open tiff file
    stack = tf.imread(tiff_path)
    resolution = stack.shape[1]

    # downsample to 8-bit
    stack8 = np.full_like(stack, fill_value=0)
    for frame in np.arange(stack.shape[0]):
        stack8[frame] = convert_to_8bit(stack[frame], np.uint8, 0, 255)

    # grouped average by specified interval
    group_by = 4
    num_frames = stack8.shape[0] // group_by
    avgd_stack = np.empty((num_frames, resolution, resolution), dtype='uint16')
    # avgd_stack = np.empty((num_frames, resolution, resolution), dtype='uint8')
    frame_count = np.arange(0, stack8.shape[0], group_by)

    for i in np.arange(num_frames):
        frame = frame_count[i]
        avgd_stack[i] = np.mean(stack8[frame:frame + group_by], axis=0)

    if save_as == None:
        save_as = tiff_path[:-4] + '_downsampled.tif'

    # write output
    print("saving as... %s" % save_as)
    tf.imwrite(save_as,
               avgd_stack, photometric='minisblack')

    return


def subselect_tiff(tiff_stack, select_frames, save_as):
    stack_cropped = tiff_stack[select_frames[0]:select_frames[1]]

    stack8 = convert_to_8bit(stack_cropped)

    tf.imwrite(save_as, stack8, photometric='minisblack')


# STATISTICS AND OTHER ANALYSES FUNCTIONS

# calculate correlation across all cells
def corrcoef_array(array):
    '''This is a horribly slow function, modify to use numpy.corrcoef() which way faster'''
    df = pd.DataFrame(array)
    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[str(col_a) + '__' + str(col_b)] = stats.pearsonr(df.loc[:, col_a], df.loc[:, col_b])

    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    corr = result['PCC'].mean()

    print('Correlation coefficient: %.2f' % corr)

    return corr, result


# calculate reliability of photostim responsiveness of all of the targeted cells (found in s2p output)
def calculate_reliability(expobj, dfstdf_threshold=None, dff_threshold=None, pre_stim=10,
                          post_stim=200, filter_for_sz=False):
    """calculates the percentage of successful photoresponsive trials for each targeted cell, where success is post
     stim response over the dff_threshold. the filter_for_sz argument is set to True when needing to filter out stim timings
     that occured when the cell was classified as inside the sz boundary."""

    reliability = {}  # dict will be used to store the reliability results for each targeted cell
    targets_dff_all_stimtrials = {}  # dict will contain the peri-stim dFF for each cell by the cell_idx
    stim_timings = expobj.stim_start_frames

    if filter_for_sz:
        # assert list(stim_timings) == list(expobj.cells_sz_stim.keys())  # dont really need this assertion because you wont necessarily always look at the sz boundary for all stims every trial
        stim_timings = expobj.cells_sz_stim.keys()
        if dff_threshold:
            threshold = round(dff_threshold)
            df = expobj.dff_all_cells
        elif dfstdf_threshold:
            threshold = dfstdf_threshold
            df = expobj.dfstdf_all_cells
        else:
            raise Exception("need to specify either dff_threshold or dfstdf_threshold value to use")

        # if you need to filter out cells based on their location inside the sz or not, you need a different approach
        # where you are for looping over each stim and calculating the reliability of cells like that. BUT the goal is still to collect reliability values by cell.

        for cell in expobj.s2p_cell_targets:
            # print('considering cell # %s' % cell)
            if cell in expobj.cell_id:
                cell_idx = expobj.cell_id.index(cell)

                # going to start using the pandas photostim response df
                stims_to_use = [str(stim) for stim in stim_timings if cell not in expobj.cells_sz_stim[
                    stim]]  # select only the stim times where the cell IS NOT inside the sz boundary
                counter = len(stims_to_use)
                responses = df.loc[
                    cell, stims_to_use]  # collect the appropriate responses for the current cell at the selected stim times
                success = sum(i >= threshold for i in responses)

                reliability[cell] = success / counter * 100.

    else:
        # TODO need to update code to utilize the main responses pandas df for the expobj
        if dff_threshold:
            threshold = round(dff_threshold)
            dff = True
        elif dfstdf_threshold:
            threshold = dfstdf_threshold
            dff = False
        else:
            raise Exception("need to specify either dff_threshold or dfstdf_threshold value to use")
        for cell in expobj.s2p_cell_targets:
            # print('considering cell # %s' % cell)
            if cell in expobj.cell_id:
                cell_idx = expobj.cell_id.index(cell)
                # collect a trace of prestim and poststim raw fluorescence for each stim time
                flu_all_stims = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
                                 stim not in expobj.seizure_frames]  # REMOVING ALL STIMS FROM SEIZURE FRAMES

                success = 0
                counter = 0
                for trace in flu_all_stims:
                    counter += 1
                    # calculate dFF (noramlized to pre-stim) for each trace
                    pre_stim_mean = np.mean(trace[0:pre_stim])
                    if dff:
                        response_trace = ((trace - pre_stim_mean) / pre_stim_mean) * 100
                    elif not dff:
                        std_pre = np.std(trace[0:expobj.pre_stim])
                        response_trace = ((trace - pre_stim_mean) / std_pre) * 100

                    # calculate if the current trace beats dff_threshold for calculating reliability (note that this happens over a specific window just after the photostim)
                    response = np.mean(response_trace[
                                       pre_stim + expobj.duration_frames:pre_stim + 3 * expobj.duration_frames])  # calculate the dF over pre-stim mean F response within the response window
                    if response >= threshold:
                        success += 1

                reliability[cell] = success / counter * 100.
    print(reliability)
    print("avg reliability is: %s (calc. over %s stims)" % (round(np.mean(list(reliability.values())), 2), counter))
    return reliability


# calculate the dFF responses of the non-targeted cells, create a pandas df of the post-stim dFF responses of all cells

def all_cell_responses_dff(expobj, normalize_to=''):
    d = {}
    # d['group'] = [int(expobj.good_photostim_cells.index(x)) for x in expobj.good_photostim_cells for y in x]
    d['group'] = ['non-target'] * (len(expobj.good_cells))  # start with all cell in the non-targets group
    for stim in expobj.stim_start_frames:
        d['%s' % stim] = [None] * len(expobj.good_cells)
    df = pd.DataFrame(d, index=expobj.good_cells)  # population dataframe

    risky_cells = []
    for cell in np.unique([expobj.good_cells + expobj.s2p_cell_targets]):

        if cell in expobj.s2p_cell_targets:
            group = 'photostim target'
            move_forward = True
        elif cell in expobj.good_cells:
            group = 'non-target'
            move_forward = True
        else:
            move_forward = False

        if move_forward:
            if normalize_to == 'baseline':
                mean_base = np.mean(expobj.baseline_raw_df.loc[str(cell), :])
                for stim in expobj.stim_start_frames:
                    cell_idx = expobj.cell_id.index(cell)
                    trace = expobj.raw[cell_idx][
                            stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                    trace_dff = ((trace - mean_base) / abs(mean_base)) * 100
                    response = np.mean(trace_dff[
                                       expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 3 * expobj.duration_frames])
                    df.at[cell, '%s' % stim] = round(response, 3)
                    df.at[cell, 'group'] = group
                if mean_base < 50:
                    risky_cells.append(cell)

            elif normalize_to == 'pre-stim':
                mean_pre_list = []
                for stim in expobj.stim_start_frames:
                    cell_idx = expobj.cell_id.index(cell)
                    trace = expobj.raw[cell_idx][
                            stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                    mean_pre = np.mean(trace[0:expobj.pre_stim]);
                    mean_pre_list.append(mean_pre)
                    trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
                    response = np.mean(trace_dff[
                                       expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 3 * expobj.duration_frames])
                    df.at[cell, '%s' % stim] = round(response, 3)
                    df.at[cell, 'group'] = group
                if np.mean(mean_pre_list) < 50:
                    risky_cells.append(cell)
            else:
                raise Exception('use either normalize_to = "baseline" or "pre-stim"')

                # how to solve the issue of very very large dFF values being caused by very small mean_pre_stim values?
                # option #1) just remove all cells whose average dFF values are above 500 or 1000 %
                # option #2) remove all cells who average mean pre values are less than some number (e.g 100)
                # option #3) if the mean_pre_stim value is less than 10, just make it nan for this stim trial for this cell

    # # getting rid of the for loop below that was used previously for more hard coded photostim target groups
    # for group in cell_groups:
    #     # hard coded number of stim. groups as the 0 and 1 in the list of this for loop
    #     if group == 'non-targets':
    #         for stim in expobj.stim_start_frames:
    #             cells = [i for i in expobj.good_cells if i not in expobj.s2p_cell_targets]
    #             for cell in cells:
    #                 cell_idx = expobj.cell_id.index(cell)
    #                 trace = expobj.raw[cell_idx][
    #                         stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre))  * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim])
    #                 # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #     elif 'photostim target' in group:
    #         cells = expobj.s2p_cell_targets
    #         for stim in expobj.stim_start_frames:
    #             for cell in cells:
    #                 cell_idx = expobj.cell_id.index(cell)
    #                 trace = expobj.raw[cell_idx][
    #                         stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim])
    #                 # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 3 * expobj.duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #                 df.at[cell, 'group'] = group

    print('Completed gathering dFF responses to photostim for %s cells' % len(
        np.unique([expobj.good_cells + expobj.s2p_cell_targets])))
    print('risky cells (with low Flu values to normalize with): ', risky_cells)

    return df


def all_cell_responses_dFstdF(expobj):
    # TODO need to confirm that this code works properly
    # normalizing post stim dF response to PRE-STIM std F

    all_cells_stim_traces_dF_stdF_avg = []

    d = {}
    # d['group'] = [int(expobj.good_photostim_cells.index(x)) for x in expobj.good_photostim_cells for y in x]
    d['group'] = ['non-target'] * (len(expobj.good_cells))  # start with all cell in the non-targets group
    for stim in expobj.stim_start_frames:
        d['%s' % stim] = [None] * len(expobj.good_cells)
    df = pd.DataFrame(d, index=expobj.good_cells)
    # population dataframe

    for cell in np.unique([expobj.good_cells + expobj.s2p_cell_targets]):
        if cell in expobj.s2p_cell_targets:
            group = 'photostim target'
            move_forward = True
        elif cell in expobj.good_cells:
            group = 'non-target'
            move_forward = True
        else:
            move_forward = False

        if move_forward:
            stim_traces_dF_stdF = []
            for stim in expobj.stim_start_frames:
                cell_idx = expobj.cell_id.index(cell)
                trace = expobj.raw[cell_idx][
                        stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                mean_pre = np.mean(trace[0:expobj.pre_stim])
                std_pre = np.std(trace[0:expobj.pre_stim])
                dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                stim_traces_dF_stdF.append(dF_stdF)
                response = np.mean(
                    dF_stdF[expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])

                df.at[cell, '%s' % stim] = round(response, 4)
                df.at[cell, 'group'] = group
            all_cells_stim_traces_dF_stdF_avg.append(np.mean(stim_traces_dF_stdF, axis=0))

    # # getting rid of the for loop below that was used previously for hard coded photostim target groups
    # for group in cell_groups:
    #     # hard coded number of stim. groups as the 0 and 1 in the list of this for loop
    #     if group == 'non-targets':
    #         for stim in expobj.stim_start_frames:
    #             cells = [i for i in expobj.good_cells if i not in expobj.s2p_cell_targets]
    #             for cell in cells:
    #                 cell_idx = expobj.cell_id.index(cell)
    #                 trace = expobj.raw[cell_idx][
    #                         stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre))  * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim])
    #                 # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #     elif 'photostim target' in group:
    #         cells = expobj.s2p_cell_targets
    #         for stim in expobj.stim_start_frames:
    #             for cell in cells:
    #                 cell_idx = expobj.cell_id.index(cell)
    #                 trace = expobj.raw[cell_idx][
    #                         stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim])
    #                 # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 3 * expobj.duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #                 df.at[cell, 'group'] = group

    print('Completed gathering dF/stdF responses to photostim for %s cells' % len(
        np.unique([expobj.good_cells + expobj.s2p_cell_targets])))

    return df


# 2 functions for plotting photostimulation timed DFF, baseline DFF considered as pre-stim period
def plot_photostim_avg(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None):
    flu_avg = np.median(dff_array, axis=0)
    std = np.std(dff_array, axis=0)
    ci = 1.960 * (std / np.sqrt(len(
        dff_array)))  # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
    x = list(range(-pre_stim, post_stim))
    y = flu_avg

    fig, ax = plt.subplots()
    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)  # plot confidence interval
    ax.axvspan(0, stim_duration, alpha=0.2, color='red')
    ax.plot(x, y)
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    fig.suptitle(title, y=0.95)
    plt.show()


def plot_photostim_(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None):
    x = list(range(-pre_stim, post_stim))
    fig, ax = plt.subplots()
    ax.axvspan(0, stim_duration, alpha=0.2, color='red')
    for cell_trace in dff_array:
        ax.plot(x, cell_trace, linewidth='0.5')
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    fig.suptitle(title, y=0.95)
    plt.show()


## kept in utils.funcs_pj
def plot_single_tiff(tiff_path: str, title: str = None):
    """
    plots an image of a single tiff frame after reading using tifffile.
    :param tiff_path: path to the tiff file
    :param title: give a string to use as title (optional)
    :return: imshow plot
    """
    stack = tf.imread(tiff_path, key=0)
    plt.imshow(stack, cmap='gray')
    if title is not None:
        plt.suptitle(title)
    plt.show()

#### archive

# other useful functions written by me
# paq2py by Llyod Russel
# def paq_read(file_path=None, plot=False):
#     """
#     Read PAQ file (from PackIO) into python
#     Lloyd Russell 2015
#     Parameters
#     ==========
#     file_path : str, optional
#         full path to file to read in. if none is supplied a load file dialog
#         is opened, buggy on mac osx - Tk/matplotlib. Default: None.
#     plot : bool, optional
#         plot the data after reading? Default: False.
#     Returns
#     =======
#     data : ndarray
#         the data as a m-by-n array where m is the number of channels and n is
#         the number of datapoints
#     chan_names : list of str
#         the names of the channels provided in PackIO
#     hw_chans : list of str
#         the hardware lines corresponding to each channel
#     units : list of str
#         the units of measurement for each channel
#     rate : int
#         the acquisition sample rate, in Hz
#     """
#
#     # file load gui
#     if file_path is None:
#         import Tkinter
#         import tkFileDialog
#         root = Tkinter.Tk()
#         root.withdraw()
#         file_path = tkFileDialog.askopenfilename()
#         root.destroy()
#
#     # open file
#     fid = open(file_path, 'rb')
#
#     # get sample rate
#     rate = int(np.fromfile(fid, dtype='>f', count=1))
#
#     # get number of channels
#     num_chans = int(np.fromfile(fid, dtype='>f', count=1))
#
#     # get channel names
#     chan_names = []
#     for i in range(num_chans):
#         num_chars = int(np.fromfile(fid, dtype='>f', count=1))
#         chan_name = ''
#         for j in range(num_chars):
#             chan_name = chan_name + chr(np.fromfile(fid, dtype='>f', count=1))
#         chan_names.append(chan_name)
#
#     # get channel hardware lines
#     hw_chans = []
#     for i in range(num_chans):
#         num_chars = int(np.fromfile(fid, dtype='>f', count=1))
#         hw_chan = ''
#         for j in range(num_chars):
#             hw_chan = hw_chan + chr(np.fromfile(fid, dtype='>f', count=1))
#         hw_chans.append(hw_chan)
#
#     # get acquisition units
#     units = []
#     for i in range(num_chans):
#         num_chars = int(np.fromfile(fid, dtype='>f', count=1))
#         unit = ''
#         for j in range(num_chars):
#             unit = unit + chr(np.fromfile(fid, dtype='>f', count=1))
#         units.append(unit)
#
#     # get data
#     temp_data = np.fromfile(fid, dtype='>f', count=-1)
#     num_datapoints = int(len(temp_data)/num_chans)
#     data = np.reshape(temp_data, [num_datapoints, num_chans]).transpose()
#
#     # close file
#     fid.close()
#
#     # plot
#     if plot:
#         # import matplotlib
#         # matplotlib.use('QT4Agg')
#         import matplotlib.pylab as plt
#         f, axes = plt.subplots(num_chans, 1, sharex=True, figsize=(10,num_chans), frameon=False)
#         for idx, ax in enumerate(axes):
#             ax.plot(data[idx])
#             ax.set_xlim([0, num_datapoints-1])
#             ax.set_ylim([data[idx].min()-1, data[idx].max()+1])
#             # ax.set_ylabel(units[idx])
#             ax.set_title(chan_names[idx])
#         plt.tight_layout()
#         plt.show()
#
#     return {"data": data,
#             "chan_names": chan_names,
#             "hw_chans": hw_chans,
#             "units": units,
#             "rate": rate,
#             "num_datapoints": num_datapoints}
