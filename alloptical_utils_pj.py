#### NOTE: THIS IS NOT CURRENTLY SETUP TO BE ABLE TO HANDLE MULTIPLE GROUPS/STIMS (IT'S REALLY ONLY FOR A SINGLE STIM TRIGGER PHOTOSTIM RESPONSES)

# TODO need to condense functions that are currently all calculating photostim responses
#      essentially you should only have to calculate the poststim respones for all cells (including targets) once to avoid redundancy, and
#      more importantly to avoid risk of calculating it differently at various stages.


import re
import glob
import pandas as pd
import itertools

import os
import sys

from utils.funcs_pj import SaveDownsampledTiff, subselect_tiff, make_tiff_stack, convert_to_8bit

sys.path.append('/home/pshah/Documents/code/')
# from Vape.utils.paq2py import *
from matplotlib.colors import ColorConverter
from Vape.utils.utils_funcs import *
from Vape.utils import STAMovieMaker_noGUI as STAMM
import scipy.stats as stats
import statsmodels.api
import statsmodels as sm
from suite2p.run_s2p import run_s2p
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xml.etree.ElementTree as ET
import tifffile as tf
import csv
import warnings
import bisect

from utils import funcs_pj as pj
from utils.paq_utils import paq_read, frames_discard
import alloptical_plotting_utils as aoplot
import pickle

from numba import njit


# %%
def import_expobj(aoresults_map_id: str = None, trial: str = None, prep: str = None, date: str = None, pkl_path: str = None, verbose: bool = True,
                  do_processing: bool = False):

    """
    primary function for importing of saved expobj files saved pickel files.

    :param aoresults_map_id:
    :param trial:
    :param prep:
    :param date:
    :param pkl_path:
    :param verbose:
    :param do_processing: whether to do extra misc. processing steps that are the end of the importing code here.
    :return:
    """

    if aoresults_map_id is not None:
        if 'pre' in aoresults_map_id:
            exp_type = 'pre'
        elif 'post' in aoresults_map_id:
            exp_type = 'post'
        id = aoresults_map_id.split(' ')[1][0]
        if len(allopticalResults.trial_maps[exp_type][id]) > 1:
            num_ = int(re.search(r"\d", aoresults_map_id)[0])
        else:
            num_ = 0
        prep, trial = allopticalResults.trial_maps[exp_type][id][num_].split(' ')

    if pkl_path is None:
        if date is None:
            date = allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, trial), 'date'].values[0]
        pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)



    if not os.path.exists(pkl_path):
        raise Exception('pkl path NOT found: ' + pkl_path)
    else:
        if trial is not None and date is not None:
            print('\n-----------------------------------------------------------------------------------------------------------',
                  '\nimporting expobj for "%s, %s" from: %s ... ' % (date, trial, pkl_path)) if verbose else None
    with open(pkl_path, 'rb') as f:
        expobj = pickle.load(f)
        experiment = '%s: %s, %s, %s' % (
            expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'],
            expobj.metainfo['comments'])
        print('|- DONE IMPORT of %s' % experiment) if verbose else None

    if pkl_path is not None:
        if expobj.pkl_path != pkl_path:
            expobj.pkl_path = pkl_path
            print('updated expobj.pkl_path ', pkl_path)
            expobj.analysis_save_path = expobj.pkl_path[:-20]
            print('updated expobj.analysis_save_path ', expobj.analysis_save_path)
            expobj.save()

    ### roping in some extraneous processing steps if there's expobj's that haven't completed for them
    if not hasattr(expobj, 'paq_rate') or not hasattr(expobj, 'frame_start_time_actual'):
        print('\n-running paqProcessing to update paq attr.s in expobj')
        expobj.paqProcessing()
        expobj.save_pkl()

    if not hasattr(expobj, 'post_stim_response_frames_window'):
        expobj.post_stim_response_window_msec = 500  # msec
        expobj.post_stim_response_frames_window = int(expobj.fps * expobj.post_stim_response_window_msec / 1000)

    # other misc. things you want to do when importing expobj -- should be temp code basically - not essential for actual importing of expobj
    if do_processing:
        if expobj.metainfo['animal prep.'] not in expobj.analysis_save_path:
            expobj.analysis_save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s" % (date, expobj.metainfo['animal prep.'], date, trial)
            # expobj.analysis_save_path = expobj.analysis_save_path + '/' + expobj.metainfo['animal prep.'] + '/' + expobj.metainfo['date'] + '_' + expobj.metainfo['trial']

        if not expobj.pre_stim == int(1.0 * expobj.fps):
            print('updating expobj.pre_stim_sec to 1 sec')
            expobj.pre_stim = int(1.0 * expobj.fps)

        if not hasattr(expobj, 'pre_stim_response_frames_window'):
            expobj.pre_stim_response_window_msec = 500  # msec
            expobj.pre_stim_response_frames_window = int(expobj.fps * expobj.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)

        if not hasattr(expobj, 'subset_frames') and hasattr(expobj, 'curr_trial_frames'):
            expobj.subset_frames = expobj.curr_trial_frames
            expobj.save()

        if not hasattr(expobj, 'neuropil'):
            # running s2p Processing
            expobj.s2pProcessing(s2p_path=expobj.s2p_path, subset_frames=expobj.curr_trial_frames, subtract_neuropil=True,
                                 baseline_frames=expobj.baseline_frames, force_redo=True)

        expobj.save()

    return expobj, experiment

def import_resultsobj(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        print(f"\nimporting resultsobj from: {pkl_path} ... " )
        resultsobj = pickle.load(f)
        print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
    return resultsobj


class TwoPhotonImaging:

    def __init__(self, tiff_path_dir, tiff_path, paq_path, metainfo, analysis_save_path, suite2p_path=None, suite2p_run=False,
                 save_downsampled_tiff: bool = False, quick=False):
        """
        :param suite2p_path: path to the suite2p outputs (plane0 file? or ops file? not sure yet)
        :param suite2p_run: set to true if suite2p is already run for this trial
        """

        print('***** CREATING NEW TwoPhotonImaging with the following metainfo: ', metainfo)

        self.tiff_path_dir = tiff_path_dir
        self.tiff_path = tiff_path
        self.paq_path = paq_path
        self.metainfo = metainfo
        self.analysis_save_path = analysis_save_path

        # create analysis save path location
        if os.path.exists(analysis_save_path):
            self.analysis_save_path = analysis_save_path
        else:
            self.analysis_save_path = analysis_save_path
            print('making analysis save folder at: \n  %s' % self.analysis_save_path)
            os.makedirs(self.analysis_save_path)

        # create pkl path and save expobj to pkl object
        pkl_path = "%s/%s_%s.pkl" % (self.analysis_save_path, metainfo['date'], metainfo['trial'])  # specify path in Analysis folder to save pkl object
        self.save_pkl(pkl_path=pkl_path)

        # if os.path.exists(self.analysis_save_path):
        #     pass
        # elif os.path.exists(self.analysis_save_path[:-17]):
        #     print('making analysis save folder at: \n  %s' % self.analysis_save_path)
        #     os.mkdir(self.analysis_save_path)
        # else:
        #     raise Exception('cannot find save folder path at: ', self.analysis_save_path[:-17])

        # elif os.path.exists(self.analysis_save_path[:-27]):
        #     print('making analysis save folder at: \n  %s \n and %s' % (self.analysis_save_path[:-17], self.analysis_save_path))
        #     os.mkdir(self.analysis_save_path[:-17])
        #     os.mkdir(self.analysis_save_path)

        self._parsePVMetadata()
        if not quick:
            stack = self.mean_raw_flu_trace(save_pkl=True) if not quick else None
        if save_downsampled_tiff:
            SaveDownsampledTiff(stack=stack, save_as=analysis_save_path + '/%s_%s_downsampled.tif' % (
            metainfo['date'], metainfo['trial']))  # specify path in Analysis folder to save pkl object')

        if suite2p_run:
            self.suite2p_path = suite2p_path
            self.s2pProcessing(s2p_path=self.suite2p_path)

        self.save_pkl(pkl_path=pkl_path)

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
            raise Exception('ERROR: Could not find or load xml for this acquisition from %s' % tiff_path)

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

    def s2pProcessing(self, s2p_path, subset_frames=None, subtract_neuropil=True, baseline_frames=[],
                      force_redo: bool = False, save=True):
        """processing of suite2p data on the experimental object"""

        if force_redo:
            continu = True
        elif hasattr(self, 'stat'):
            print('skipped re-processing suite2p data for current trial')
            continu = False
        else:
            continu = True

        if continu:

            self.cell_id = []
            self.n_units = []
            self.cell_plane = []
            self.cell_med = []
            self.cell_x = []
            self.cell_y = []
            self.raw = []
            self.mean_img = []
            self.radius = []
            self.s2p_path = s2p_path
            self.subset_frames = subset_frames

            if self.n_planes == 1:
                # s2p_path = os.path.join(self.tiff_path, 'suite2p', 'plane' + str(plane))
                FminusFneu, spks, self.stat, neuropil = s2p_loader(s2p_path,
                                                              subtract_neuropil)  # s2p_loader() is in Vape/utils_func
                ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()

                if self.subset_frames is None:
                    self.raw = FminusFneu
                    self.neuropil = neuropil
                    self.spks = spks
                elif self.subset_frames is not None:
                    self.raw = FminusFneu[:, self.subset_frames[0]:self.subset_frames[1]]
                    self.spks = spks[:, self.subset_frames[0]:self.subset_frames[1]]
                    self.neuropil = neuropil[:, self.subset_frames[0]:self.subset_frames[1]]
                if len(self.baseline_frames) > 0:
                    self.baseline_raw = FminusFneu[:, self.baseline_frames[0]:self.baseline_frames[1]]
                self.mean_img = ops['meanImg']
                cell_id = []
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
                    FminusFneu, self.spks, self.stat, self.neuro = s2p_loader(s2p_path,
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

            print('|- Loaded %s cells, recorded for %s secs' % (self.raw.shape[0], round(self.raw.shape[1]/self.fps, 2)))

            self.save() if save else None

    def _good_cells(self, cell_ids: list, raws: np.ndarray, photostim_frames: list, std_thresh: int, radiuses: list = None,
                    min_radius_pix: int = 2.5, max_radius_pix: int = 10, save=True):
        """
        This function is used for filtering for "good" cells based on detection of Flu deflections that are above some std threshold based on std_thresh.
        Note: a moving averaging window of 4 frames is used to find Flu deflections above std threshold.

        :param cell_ids: list of cell ids to iterate over
        :param raws: raw flu values corresponding to the cell list in cell_ids
        :param photostim_frames: frames to delete from the raw flu traces across all cells (because they are photostim frames)
        :param std_thresh: std. factor above which to define reliable Flu events
        :param radiuses: radiuses of the s2p ROI mask of all cells in the same order as cell_ids
        :param min_radius_pix:
        :param max_radius_pix:
        :return:
            good_cells: list of cell_ids that had atleast 1 event above the std_thresh
            events_loc_cells: dictionary containing locations for each cell_id where the moving averaged Flu trace passes the std_thresh
            flu_events_cells: dictionary containing the dff Flu value corresponding to events_loc_cells
            stds = dictionary containing the dFF trace std value for each cell_id
        """

        print('\n----------------------------------------------------------------')
        print('running finding of good cells for suite2p ROIs ')
        print('----------------------------------------------------------------')

        good_cells = []
        events_loc_cells = {}
        flu_events_cells = {}
        stds = {}  # collect the std values for all filtered cells used later for identifying high and low std cells
        for i in range(len(cell_ids)):
            cell_id = cell_ids[i]

            if i % 100 == 0:  # print the progress once every 100 cell iterations
                print(i, " out of ", len(cell_ids), " cells done", end='\r')

            # print(i, " out of ", len(cell_ids), " cells")
            raw = raws[i]
            if np.mean(raw) > 1:  # exclude all negative raw traces and very small mean raw traces
                raw_ = np.delete(raw, photostim_frames)
                raw_dff = normalize_dff_jit(
                    raw_)  # note that this function is defined in this file a little further down
                std_ = raw_dff.std()

                raw_dff_ = moving_average(raw_dff, n=4)

                thr = np.mean(raw_dff) + std_thresh * std_
                events = np.where(raw_dff_ > thr)
                flu_values = raw_dff[events]

                if radiuses is not None:
                    radius = radiuses[i]
                    if len(events[0]) > 0 and radius > min_radius_pix and radius < max_radius_pix:
                        events_loc_cells[cell_id] = events
                        flu_events_cells[cell_id] = flu_values
                        stds[cell_id] = std_
                        good_cells.append(cell_id)
                elif len(events[0]) > 0:
                    events_loc_cells[cell_id] = events
                    flu_events_cells[cell_id] = flu_values
                    good_cells.append(cell_id)
                    stds[cell_id] = std_

                # if i == 465:  # just using this here if ever need to check back with specific cells if function seems to be misbehaving
                #     print(events, len(events[0]), thr)

        print('# of good cells found: ', len(good_cells), ' (out of ', len(cell_ids), ' ROIs)')
        self.good_cells = good_cells
        self.save() if save else None

        return good_cells, events_loc_cells, flu_events_cells, stds


    def paqProcessing(self, lfp=False):

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
        self.frame_start_times = [self.frame_clock[0]]  # initialize list
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

        if lfp:
            # find voltage (LFP recording signal) channel and save as lfp_signal attribute
            voltage_idx = paq['chan_names'].index('voltage')
            self.lfp_signal = paq['data'][voltage_idx]

    def mean_raw_flu_trace(self, plot: bool = False, save_pkl: bool = True):
        print('\n-----collecting mean raw flu trace from tiff file...')
        print(self.tiff_path)
        if len(self.tiff_path) == 2:  # this is to account for times when two channel imaging might be used for one t-series
            tiff_path = self.tiff_path[0]
        else:
            tiff_path = self.tiff_path
        im_stack = tf.imread(tiff_path, key=range(self.n_frames))
        print('|- Loaded experiment tiff of shape: ', im_stack.shape)

        self.meanFluImg = np.mean(im_stack, axis=0)
        self.meanRawFluTrace = np.mean(np.mean(im_stack, axis=1), axis=1)

        if save_pkl:
            if hasattr(self, 'pkl_path'):
                self.save_pkl(pkl_path=self.pkl_path)
            else:
                print('pkl file not saved yet because .pkl_path attr not found')

        if plot:
            aoplot.plotMeanRawFluTrace(expobj=self, stim_span_color=None, x_axis='frames', figsize=[20, 3],
                                       title='Mean raw Flu trace -')
        return im_stack

    def plot_single_frame_tiff(self, frame_num: int = 0, title: str = None):
        """
        plots an image of a single specified tiff frame after reading using tifffile.
        :param frame_num: frame # from 2p imaging tiff to show (default is 0 - i.e. the first frame)
        :param title: give a string to use as title (optional)
        :return: imshow plot
        """
        stack = tf.imread(self.tiff_path, key=frame_num)
        plt.imshow(stack, cmap='gray')
        if title is not None:
            plt.suptitle(title)
        else:
            plt.suptitle('frame num: %s' % frame_num)
        plt.show()
        return stack

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
        print("\n\t -- pkl saved to %s -- " % pkl_path)

    def save(self):
        self.save_pkl()


class alloptical(TwoPhotonImaging):

    def __init__(self, paths, metainfo, stimtype, quick=False):
        # self.metainfo = metainfo
        self.slmtargets_sz_stim = None
        self.stim_type = stimtype
        self.naparm_path = paths[2]
        self.paq_path = paths[3]


        assert os.path.exists(self.naparm_path)
        assert os.path.exists(self.paq_path)

        self.seizure_frames = []

        TwoPhotonImaging.__init__(self, tiff_path_dir=paths[0], tiff_path=paths[1], paq_path=paths[3],
                                  metainfo=metainfo, analysis_save_path=paths[4],
                                  suite2p_path=None, suite2p_run=False, quick=quick)


        # self.tiff_path_dir = paths[0]
        # self.tiff_path = paths[1]


        # self._parsePVMetadata()

        ## CREATE THE APPROPRIATE ANALYSIS SUBFOLDER TO USE FOR SAVING ANALYSIS RESULTS TO

        print('\ninitialized alloptical expobj of exptype and trial: \n', self.metainfo)

        self.stimProcessing(stim_channel='markpoints2packio')
        self._findTargetsAreas()
        self.find_photostim_frames()

        self.pre_stim = int(1.0 * self.fps)  # length of pre stim trace collected (in frames)
        self.post_stim = int(3.0 * self.fps)  # length of post stim trace collected (in frames)
        self.post_stim_response_window_msec = 500  # msec
        self.post_stim_response_frames_window = int(self.fps * self.pre_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)
        self.pre_stim_response_window_msec = 500  # msec
        self.pre_stim_response_frames_window = int(self.fps * self.pre_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)

        self.save()

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

    def subset_frames_current_trial(self, to_suite2p, trial, baseline_trials, force_redo: bool = False, save=True):

        if force_redo:
            continu = True
            print('re-collecting subset frames of current trial')
        elif hasattr(self, 'curr_trial_frames'):
            print('skipped re-collecting subset frames of current trial')
            continu = False
        else:
            continu = True
            print('collecting subset frames of current trial')

        if continu:
            # determine which frames to retrieve from the overall total s2p output
            total_frames_stitched = 0
            curr_trial_frames = None
            self.baseline_frames = [0, 0]
            for t in to_suite2p:
                pkl_path_2 = self.pkl_path[:-26] + t + '/' + self.metainfo['date'] + '_' + t + '.pkl'
                with open(pkl_path_2, 'rb') as f:
                    _expobj = pickle.load(f)
                    # import suite2p data
                total_frames_stitched += _expobj.n_frames
                if t == trial:
                    self.curr_trial_frames = [total_frames_stitched - _expobj.n_frames, total_frames_stitched]
                if t in baseline_trials:
                    self.baseline_frames[1] = total_frames_stitched

            print('baseline frames: ', self.baseline_frames)
            print('current trial frames: ', self.curr_trial_frames)

            self.save() if save else None

    def stitch_reg_tiffs(self, force_crop: bool = False, force_stack: bool = False):
        start = self.curr_trial_frames[0] // 2000  # 2000 because that is the batch size for suite2p run
        end = self.curr_trial_frames[1] // 2000 + 1

        tif_path_save = self.analysis_save_path + 'reg_tiff_%s.tif' % self.metainfo['trial']
        tif_path_save2 = self.analysis_save_path + 'reg_tiff_%s_r.tif' % self.metainfo['trial']
        reg_tif_folder = self.s2p_path + '/reg_tif/'
        reg_tif_list = os.listdir(reg_tif_folder)
        reg_tif_list.sort()
        sorted_paths = [reg_tif_folder + tif for tif in reg_tif_list][start:end + 1]

        print(tif_path_save)
        print(sorted_paths)

        if os.path.exists(tif_path_save):
            if force_stack:
                make_tiff_stack(sorted_paths, save_as=tif_path_save)
            else:
                pass
        else:
            make_tiff_stack(sorted_paths, save_as=tif_path_save)

        if not os.path.exists(tif_path_save2) or force_crop:
            with tf.TiffWriter(tif_path_save2, bigtiff=True) as tif:
                with tf.TiffFile(tif_path_save, multifile=False) as input_tif:
                    print('cropping registered tiff')
                    data = input_tif.asarray()
                    print('shape of stitched tiff: ', data.shape)
                reg_tif_crop = data[self.curr_trial_frames[0] - start * 2000: self.curr_trial_frames[1] - (
                        self.curr_trial_frames[0] - start * 2000)]
                print('saving cropped tiff ', reg_tif_crop.shape)
                tif.save(reg_tif_crop)

    def raw_traces_from_targets(self, force_redo: bool = False, save: bool = True):

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
            targets_trace_full = np.zeros([len(self.target_coords_all), (end - start) * 2000], dtype='float32')
            counter = 0
            for i in range(start, end):
                tif_path_save2 = self.s2p_path + '/reg_tif/' + reg_tif_list[i]
                with tf.TiffFile(tif_path_save2, multifile=False) as input_tif:
                    print('|- reading tiff: %s' % tif_path_save2)
                    data = input_tif.asarray()

                targets_trace = np.zeros([len(self.target_coords_all), data.shape[0]], dtype='float32')
                for coord in range(len(self.target_coords_all)):
                    target_areas = np.array(self.target_areas)  # TODO update this so that it doesn't include the extra exclusion zone
                    x = data[:, target_areas[coord, :, 1], target_areas[coord, :, 0]]  # = 1
                    targets_trace[coord] = np.mean(x, axis=1)

                targets_trace_full[:, (i - start) * 2000: ((i - start) * 2000) + data.shape[
                    0]] = targets_trace  # iteratively write to each successive segment of the targets_trace array based on the length of the reg_tiff that is read in.

                mean_img_stack[counter] = np.mean(data, axis=0)
                counter += 1

            # final part, crop to the *exact* frames for current trial
            self.raw_SLMTargets = targets_trace_full[:,
                                  self.curr_trial_frames[0] - start * 2000: self.curr_trial_frames[1] - (start * 2000)]


            self.meanFluImg_registered = np.mean(mean_img_stack, axis=0)

            self.save() if save else None

    def get_alltargets_stim_traces_norm(self, targets_idx: int = None, subselect_cells: list = None, pre_stim=15,
                                        post_stim=200, filter_sz: bool = False, stims: list = None):
        """
        primary function to measure the dFF and dF/setdF traces for photostimulated targets.
        :param stims:
        :param targets_idx: integer for the index of target cell to process
        :param subselect_cells: list of cells to subset from the overall set of traces (use in place of targets_idx if desired)
        :param pre_stim: number of frames to use as pre-stim
        :param post_stim: number of frames to use as post-stim
        :param filter_sz: whether to filter out stims that are occuring seizures
        :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
        """
        if filter_sz:
            print('\n -- working on getting stim traces for cells inside sz boundary --')

        if stims is None:
            stim_timings = self.stim_start_frames
        else:
            stim_timings = stims

        if subselect_cells:
            num_cells = len(self.raw_SLMTargets[subselect_cells])
            targets_trace = self.raw_SLMTargets[subselect_cells]
        else:
            num_cells = len(self.raw_SLMTargets)
            targets_trace = self.raw_SLMTargets

        # collect photostim timed average dff traces of photostim targets
        targets_dff = np.zeros(
            [num_cells, len(self.stim_start_frames), pre_stim + self.stim_duration_frames + post_stim])
        # targets_dff_avg = np.zeros([num_cells, pre_stim_sec + post_stim_sec])

        targets_dfstdF = np.zeros(
            [num_cells, len(self.stim_start_frames), pre_stim + self.stim_duration_frames + post_stim])
        # targets_dfstdF_avg = np.zeros([num_cells, pre_stim_sec + post_stim_sec])

        targets_raw = np.zeros(
            [num_cells, len(self.stim_start_frames), pre_stim + self.stim_duration_frames + post_stim])
        # targets_raw_avg = np.zeros([num_cells, pre_stim_sec + post_stim_sec])

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
                trace_dff = ((trace - mean_pre) / mean_pre) * 100
                std_pre = np.std(trace[0:pre_stim])
                dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                targets_raw[targets_idx, i] = trace
                targets_dff[targets_idx, i] = trace_dff
                targets_dfstdF[targets_idx, i] = dFstdF
            return targets_raw[targets_idx], targets_dff[targets_idx], targets_dfstdF[targets_idx]

        else:
            for cell_idx in range(num_cells):
                if subselect_cells:
                    print('collecting stim traces for cell %s' % subselect_cells[cell_idx])
                else:
                    pass
                    # print('collecting stim traces for cell # %s out of %s' % (cell_idx + 1, num_cells))
                if filter_sz:
                    if self.slmtargets_sz_stim is not None and hasattr(self, 'slmtargets_sz_stim'):
                        flu = []
                        for stim in stim_timings:
                            if stim in self.slmtargets_sz_stim.keys():  # some stims dont have sz boundaries because of issues with their TIFFs not being made properly (not readable in Fiji), usually it is the first TIFF in a seizure
                                if cell_idx not in self.slmtargets_sz_stim[stim]:
                                    flu.append(targets_trace[cell_idx][stim - pre_stim: stim + self.stim_duration_frames + post_stim])
                    else:
                        flu = []
                        print('classifying of sz boundaries not completed for this expobj', self.metainfo['animal prep.'], self.metainfo['trial'])
                    # flu = [targets_trace[cell_idx][stim - pre_stim_sec: stim + self.stim_duration_frames + post_stim_sec] for
                    #        stim
                    #        in stim_timings if
                    #        stim not in self.seizure_frames]
                else:
                    flu = [targets_trace[cell_idx][stim - pre_stim: stim + self.stim_duration_frames + post_stim] for
                           stim
                           in stim_timings]

                # flu_dfstdF = []
                # flu_dff = []
                # flu = []
                if len(flu) > 0:
                    for i in range(len(flu)):
                        trace = flu[i]
                        mean_pre = np.mean(trace[0:pre_stim])
                        trace_dff = ((trace - mean_pre) / mean_pre) * 100
                        std_pre = np.std(trace[0:pre_stim])
                        dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                        targets_raw[cell_idx, i] = trace
                        targets_dff[cell_idx, i] = trace_dff
                        targets_dfstdF[cell_idx, i] = dFstdF
                        # flu_dfstdF.append(dFstdF)
                        # flu_dff.append(trace_dff)

                # targets_dff.append(flu_dff)  # contains all individual dFF traces for all stim times
                # targets_dff_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

                # targets_dfstdF.append(flu_dfstdF)
                # targets_dfstdF_avg.append(np.nanmean(flu_dfstdF, axis=0))

                # SLMTargets_stims_raw.append(flu)
                # targets_raw_avg.append(np.nanmean(flu, axis=0))

            targets_dff_avg = np.mean(targets_dff, axis=1)
            targets_dfstdF_avg = np.mean(targets_dfstdF, axis=1)
            targets_raw_avg = np.mean(targets_raw, axis=1)

            print(targets_dff_avg.shape)

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
        self.frame_start_times = [self.frame_clock[0]]  # initialize list
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

        self.paqProcessing(lfp=True)

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
        Note this is not done by target groups however. So all of the targets are just in one big list.
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
        #     #             image_8bit = convert_to_8bit(target_image_scaled, np.unit8)
        #     #             tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), image_8bit)
        #     tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)

        targets = np.where(target_image > 0)
        # targets_1 = np.where(target_image_scaled_1 > 0)
        # targets_2 = np.where(target_image_scaled_2 > 0)

        targetCoordinates = list(zip(targets[1] * scale_factor, targets[0] * scale_factor))
        print('Number of targets:', len(targetCoordinates))

        # targetCoordinates_1 = list(zip(targets_1[1], targets_1[0]))
        # print('Number of targets, SLM group #1:', len(targetCoordinates_1))
        #
        # targetCoordinates_2 = list(zip(targets_2[1], targets_2[0]))
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
        print(f"spiral size: {self.spiral_size}um")
        print(f"pix sz x: {self.pix_sz_x}um")
        print("radius (in pixels): {:.2f}px".format(radius_px * self.pix_sz_x))

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

        print('Got targets...')

    def _findTargetedS2pROIs(self, force_redo: bool = False, plot: bool = True):
        '''finding s2p cell ROIs that were also SLM targets (or more specifically within the target areas as specified by _findTargetAreas - include 15um radius from center coordinate of spiral)
        --- LAST UPDATED NOV 6 2021 - copied over from Rob's ---
        '''

        if force_redo:
            continu = True
        elif hasattr(self, 's2p_cell_targets'):
            print('skipped re-running of finding s2p targets from suite2p cell list')
            continu = False
        else:
            continu = True

        if continu:
            '''
            Make a binary mask of the targets and multiply by an image of the cells
            to find cells that were targeted

            --- COPIED FROM ROB'S VAPE INTERAREAL_ANALYSIS.PY ON NOV 5 2021 ---
            '''

            print('searching for targeted cells... [Rob version]')
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
            # self.s2p_cell_targets = [self.cell_id[i] for i, x in enumerate(self.targeted_cells) if x is True]  # get list of s2p cells that were photostim targetted
            self.s2p_cell_targets = [self.cell_id[i] for i in np.where(self.targeted_cells)[0]]  # get list of s2p cells that were photostim targetted

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
            self.s2p_cells_exclude = [self.cell_id[i] for i in np.where(self.exclude_cells)[0]]  # get list of s2p cells that were photostim targetted

            self.n_exclude_cells = np.sum(self.exclude_cells)

            print('------- Search completed.')
            self.save()
            print('Number of exclude cells: ', self.n_exclude_cells)

            if plot:
                fig, ax = plt.subplots(figsize=[6, 6])
                fig, ax = aoplot.plot_cells_loc(self, cells=self.s2p_cell_targets, show=False, fig=fig, ax=ax,
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

    def avg_stim_images(self, peri_frames: int = 100, stim_timings: list = [], save_img=False, to_plot=False,
                        verbose=False, force_redo=False):
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

        if force_redo:
            continu = True
        elif hasattr(self, 'avgstimimages_r'):
            if self.avgstimimages_r is True:
                continu = False
            else:
                continu = True
        else:
            continu = True

        if continu:
            print('making stim images...')
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
                    save_path_stim = save_path + '/%s_%s_stim-%s.tif' % (
                        self.metainfo['date'], self.metainfo['trial'], stim)
                    if os.path.exists(save_path):
                        print("saving stim_img tiff to... %s" % save_path_stim) if verbose else None
                        avg_sub8 = convert_to_8bit(avg_sub, 0, 255)
                        tf.imwrite(save_path_stim,
                                   avg_sub8, photometric='minisblack')
                    else:
                        print('making new directory for saving images at:', save_path)
                        os.mkdir(save_path)
                        print("saving as... %s" % save_path_stim)
                        avg_sub8 = convert_to_8bit(avg_sub, 0, 255)
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

            self.avgstimimages_r = True

        else:
            print('skipping remaking of avg stim images')

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

    # calculate reliability of photostim responsiveness of all of the targeted cells (found in s2p output)
    def calculate_SLMTarget_responses_dff(self, threshold=10, stims_to_use: list = None):
        """
        calculations of dFF responses to photostimulation of SLM Targets. Includes calculating reliability of slm targets,
        saving success stim locations, and saving stim response magnitudes as pandas dataframe.

        :param threshold: dFF threshold above which a response for a photostim trial is considered a success.
        :param stims_to_use: list of stims to retrieve photostim trial dFF responses
        :return:
        """
        if stims_to_use is None:
            stims_to_use = range(len(self.stim_start_frames))
            stims_idx = [self.stim_start_frames.index(stim) for stim in stims_to_use]
        elif stims_to_use:
            stims_idx = [self.stim_start_frames.index(stim) for stim in stims_to_use]
        else:
            AssertionError('no stims set to analyse [1]')

        # initializing pandas df that collects responses of stimulations
        if hasattr(self, 'SLMTargets_stims_dff'):
            d = {}
            for stim in stims_idx:
                d[stim] = [None] * self.SLMTargets_stims_dff.shape[0]
            df = pd.DataFrame(d, index=range(self.SLMTargets_stims_dff.shape[0]))  # population dataframe
        else:
            AssertionError('no SLMTargets_stims_dff attr. [2]')

        # initializing pandas df for binary showing of success and fails (1= success, 0= fails)
        hits_slmtargets = {}  # to be converted in pandas df below - will contain 1 for every success stim, 0 for non success stims
        for stim in stims_idx:
            hits_slmtargets[stim] = [None] * self.SLMTargets_stims_dff.shape[0]  # start with 0 for all stims
        hits_slmtargets_df = pd.DataFrame(hits_slmtargets,
                                          index=range(self.SLMTargets_stims_dff.shape[0]))  # population dataframe

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
                dff_trace = self.SLMTargets_stims_dff[target_idx][stim_idx]
                response_result = np.mean(dff_trace[self.pre_stim + self.stim_duration_frames + 1:
                                                    self.pre_stim + self.stim_duration_frames +
                                                    self.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
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
    def calculate_SLMTarget_SuccessStims(self, hits_df, stims_idx_l: list, exclude_stims_targets: dict = {}):
        """uses outputs of calculate_SLMTarget_responses_dff to calculate overall successrate of the specified stims

        :param hits_df: pandas dataframe of targets x stims where 1 denotes successful stim response (0 is failure)
        :param stims_idx_l: list of stims to use for this function (useful when needing to filter out certain stims for in/out of sz)
        :param exclude_stims_targets: dictionary of stims (keys) where the values for each stim contains the targets that should be excluded from counting in the analysis of Success/failure of trial

        :return
            reliability_slmtargets: dict; reliability (% of successful stims) for each SLM target
            traces_SLMtargets_successes_avg: np.array; photostims avg traces for each SLM target (successful stims only)

        """

        traces_SLMtargets_successes_avg_dict = {}
        traces_SLMtargets_failures_avg_dict = {}
        reliability_slmtargets = {}
        for target_idx in hits_df.index:
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
                    if hits_df.loc[target_idx, stim_idx] == 1:
                        success += 1
                        dff_trace = self.SLMTargets_stims_dff[target_idx][stim_idx]
                        traces_SLMtargets_successes_l.append(dff_trace)
                    else:
                        success += 0
                        dff_trace = self.SLMTargets_stims_dff[target_idx][stim_idx]
                        traces_SLMtargets_failures_l.append(dff_trace)

            if counter > 0:
                reliability_slmtargets[target_idx] = round(success / counter * 100., 2)
            if success > 0:
                traces_SLMtargets_successes_avg_dict[target_idx] = np.mean(traces_SLMtargets_successes_l, axis=0)
            if success < counter:  # this helps protect against cases where a trial is 100% successful (and there's no failures).
                traces_SLMtargets_failures_avg_dict[target_idx] = np.mean(traces_SLMtargets_failures_l, axis=0)

        return reliability_slmtargets, traces_SLMtargets_successes_avg_dict, traces_SLMtargets_failures_avg_dict


    # def get_alltargets_stim_traces_norm(self, pre_stim_sec=15, post_stim_sec=200, filter_sz: bool = False, stims_idx_l: list = None):
    #     """
    #     primary function to measure the dFF and dF/setdF traces for photostimulated targets.
    #     :param stims:
    #     :param targets_idx: integer for the index of target cell to process
    #     :param subselect_cells: list of cells to subset from the overall set of traces (use in place of targets_idx if desired)
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
    #     self.s2p_rois_nontargets = [cell for cell in self.cell_id if cell not in self.s2p_cell_targets]  # TODO need to also detect (and exclude) ROIs that are within some radius of the SLM targets
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
    #             if hasattr(self, 'slmtargets_sz_stim'):  ## TODO change to ROIs in sz for each stim -- has this classification been done for non-targets ROIs?
    #                 flu = []
    #                 for stim in stim_timings:
    #                     if stim in self.slmtargets_sz_stim.keys():  # some stims dont have sz boundaries because of issues with their TIFFs not being made properly (not readable in Fiji), usually it is the first TIFF in a seizure
    #                         if cell_idx not in self.slmtargets_sz_stim[stim]:
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
            flu_trials = [self.raw[cell_idx][stim - self.pre_stim: stim + self.stim_duration_frames + self.post_stim] for stim in stim_timings]


            dff_trace = normalize_dff(self.raw[cell_idx], threshold_pct=50)  # normalize trace (dFF) to mean of whole trace


            if normalize_to == 'baseline':  # probably gonna ax this anyways
                flu_dff = []
                mean_spont_baseline = np.mean(self.baseline_raw[cell_idx])
                for i in range(len(flu_trials)):
                    trace_dff = ((flu_trials[i] - mean_spont_baseline) / mean_spont_baseline) * 100

                    # add nan if cell is inside sz boundary for this stim
                    if hasattr(self, 'slmtargets_sz_stim'):
                        if self.is_cell_insz(cell=cell, stim=stim_timings[i]):
                            trace_dff = [np.nan] * len(flu_trials[i])

                    flu_dff.append(trace_dff)

            elif normalize_to == 'whole-trace':
                print('s2p neu. corrected trace statistics: mean: %s (min: %s, max: %s, std: %s)' %
                      (np.mean(self.raw[cell_idx]), np.min(self.raw[cell_idx]), np.max(self.raw[cell_idx]), np.std(self.raw[cell_idx], ddof=1)))
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
                        dFstdF = [np.nan] * len(trace)  # - commented out to test if we need to exclude cells for this correction with low mean_pre since you're not dividing by a bad mean_pre value
                    else:
                        # trace_dff = ((trace - mean_pre) / mean_pre) * 100
                        trace_dff = normalize_dff(trace, threshold_val=mean_pre)
                        # std_pre = np.std(trace[0:self.pre_stim], ddof=1)
                        # # dFstdF = (((trace - mean_pre) / mean_pre) * 100) / std_pre  # make dF divided by std of pre-stim F trace
                        # dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                    # # add nan if cell is inside sz boundary for this stim -- temporarily commented out for a while
                    # if 'post' in self.metainfo['exptype']:
                    #     if hasattr(self, 'slmtargets_sz_stim'):
                    #         if self.is_cell_insz(cell=cell, stim=stim_timings[i]):
                    #             trace_dff = [np.nan] * len(trace)
                    #             dFstdF = [np.nan] * len(trace)
                    #     else:
                    #         AttributeError(
                    #             'no slmtargets_sz_stim attr, so classify cells in sz boundary hasnot been saved for this expobj')

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

        # define non targets from suite2p ROIs (exclude cells in the SLM targets exclusion - .s2p_cells_exclude)
        expobj.s2p_nontargets = [cell for cell in expobj.good_cells if cell not in expobj.s2p_cells_exclude]  ## exclusion of cells that are classified as s2p_cell_targets


        # make trial arrays from dff data shape: [cells x stims x frames]
        expobj._makeNontargetsStimTracesArray(stim_timings=expobj.stim_start_frames, normalize_to=normalize_to, save=False)


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
        expobj._runWilcoxonsTest(save=False)

        expobj.save() if save else None


    def _runWilcoxonsTest(expobj, array1=None, array2=None, save=True):

        if array1 is None and array2 is None:
            array1 = expobj.pre_array; array2 = expobj.post_array

        # check if the two distributions of flu values (pre/post) are different
        assert array1.shape == array2.shape, 'shapes for expobj.pre_array and expobj.post_array need to be the same for wilcoxon test'
        wilcoxons = np.empty(len(array1))  # [cell (p-value)]

        for cell in range(len(array1)):
            wilcoxons[cell] = stats.wilcoxon(array2[cell], array1[cell])[1]

        expobj.wilcoxons = wilcoxons

        expobj.save() if save else None


    def _sigTestAvgResponse_nontargets(expobj, alpha=0.1, save=True):
        """
        Uses the p values and a threshold for the Benjamini-Hochberg correction to return which
        cells are still significant after correcting for multiple significance testing
        """
        print('\n----------------------------------------------------------------')
        print('running statistical significance testing for nontargets response arrays ')
        print('----------------------------------------------------------------')

        p_vals = expobj.wilcoxons
        expobj.sig_units = np.full_like(p_vals, False, dtype=bool)

        try:
            expobj.sig_units, _, _, _ = sm.stats.multitest.multipletests(p_vals, alpha=alpha, method='fdr_bh',
                                                                         is_sorted=False, returnsorted=False)
        except ZeroDivisionError:
            print('no cells responding')

        # p values without bonferroni correction
        no_bonf_corr = [i for i, p in enumerate(p_vals) if p < 0.05]
        expobj.nomulti_sig_units = np.zeros(len(expobj.s2p_nontargets), dtype='bool')
        expobj.nomulti_sig_units[no_bonf_corr] = True

        expobj.save() if save else None

        # p values after bonferroni correction
        #         bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05 / expobj.n_units[plane]]
        #         sig_units = np.zeros(expobj.n_units[plane], dtype='bool')
        #         sig_units[bonf_corr] = True


class Post4ap(alloptical):

    def __init__(self, paths, metainfo, stimtype, discard_all):
        alloptical.__init__(self, paths, metainfo, stimtype)
        print('\ninitialized Post4ap expobj of exptype and trial: %s, %s, %s' % (self.metainfo['exptype'],
                                                                                 self.metainfo['trial'],
                                                                                 self.metainfo['date']))

        # collect information about seizures
        self.collect_seizures_info(seizures_lfp_timing_matarray=paths[5], discard_all=discard_all)

        self.save()

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
            subselect_tiff(tiff_stack=stack, select_frames=select_frames, save_as=save_as)
        print('\ndone. saved to:', self.analysis_save_path)

    def collect_seizures_info(self, seizures_lfp_timing_matarray=None, discard_all=True):
        print('\ncollecting information about seizures...')
        if seizures_lfp_timing_matarray is not None:
            self.seizures_lfp_timing_matarray = seizures_lfp_timing_matarray  # path to the matlab array containing paired measurements of seizures onset and offsets

        assert self.seizures_lfp_timing_matarray is not None

        # retrieve seizure onset and offset times from the seizures info array input
        paq = paq_read(file_path=self.paq_path, plot=False)

        # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
        # NOTE: the output of all of the following function is in dimensions of the FRAME CLOCK (not official paq clock time)
        if seizures_lfp_timing_matarray is not None:
            print('-- using matlab array to collect seizures %s: ' % seizures_lfp_timing_matarray)
            bad_frames, self.seizure_frames, self.seizure_lfp_onsets, self.seizure_lfp_offsets = frames_discard(
                paq=paq[0], input_array=self.seizures_lfp_timing_matarray, total_frames=self.n_frames,
                discard_all=discard_all)
        else:
            print('-- no matlab array given to use for finding seizures.')
            bad_frames = frames_discard(paq=paq[0], input_array=seizures_lfp_timing_matarray,
                                        total_frames=self.n_frames,
                                        discard_all=discard_all)
            self.seizure_frames = []
            self.seizure_lfp_onsets = []
            self.seizure_lfp_offsets = []

        print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
        print('|- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])
        self.append_bad_frames(
            bad_frames=bad_frames)  # here only need to append the bad frames to the expobj.bad_frames property

        if hasattr(self, 'seizures_lfp_timing_matarray'):
            print('\n|-now creating raw movies for each sz as well (saved to the /Analysis folder) ... ')
            self.subselect_tiffs_sz(onsets=self.seizure_lfp_onsets, offsets=self.seizure_lfp_offsets,
                                    on_off_type='lfp_onsets_offsets')

            print('\n|-now classifying photostims at phases of seizures ... ')
            self.stims_in_sz = [stim for stim in self.stim_start_frames if stim in self.seizure_frames]
            self.stims_out_sz = [stim for stim in self.stim_start_frames if stim not in self.seizure_frames]

            # self.stims_bf_sz = [self.stim_start_frames[self.stim_start_frames.index(sz_start) - 1] for sz_start in self.seizure_lfp_onsets]

            self.stims_bf_sz = [stim for stim in self.stim_start_frames
                                for sz_start in self.seizure_lfp_onsets
                                if 0 < (
                                        sz_start - stim) < 10 * self.fps]  # select stims that occur within 5 seconds before of the sz onset
            self.stims_af_sz = [stim for stim in self.stim_start_frames
                                for sz_start in self.seizure_lfp_offsets
                                if 0 < -1 * (
                                        sz_start - stim) < 10 * self.fps]  # select stims that occur within 5 seconds afterof the sz offset
            print(' \n|- stims_in_sz:', self.stims_in_sz, ' \n|- stims_out_sz:', self.stims_out_sz,
                  ' \n|- stims_bf_sz:', self.stims_bf_sz, ' \n|- stims_af_sz:', self.stims_af_sz)
            aoplot.plot_lfp_stims(self)
        self.save_pkl()

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

            if baseline_tiff is None:
                raise Exception(
                    'please provide a baseline tiff path to use for this trial -- usually the spont imaging trials of the same experiment')

            print('First loading up and plotting baseline (comparison) tiff from: ', baseline_tiff)
            im_stack_base = tf.imread(baseline_tiff,
                                      key=range(5000))  # reading in just the first 5000 frames of the spont
            avg_baseline = np.mean(im_stack_base, axis=0)
            plt.imshow(avg_baseline, cmap='gray')
            plt.suptitle('avg 5000 frames baseline from %s' % baseline_tiff[-35:], wrap=True)
            plt.show()

            tiffs_loc = '%s/*Ch3.tif' % self.tiff_path_dir
            tiff_path = glob.glob(tiffs_loc)[0]
            print('loading up post4ap tiff from: ', tiff_path)
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
                SaveDownsampledTiff(stack=im_sub, save_as=self.analysis_save_path + '%s_%s_sz%s_downsampled.tiff' % (self.metainfo['date'], self.metainfo['trial'], counter))

                self.meanszimages_r = True

            self.avg_sub_list = avg_sub_list
        else:
            print('skipping remaking of mean sz images')

    def _InOutSz(self, cell_med: list, sz_border_path: str, to_plot=False):
        """
        Returns True if the given cell's location is inside the seizure boundary which is defined as the coordinates
        given in the .csv sheet.

        :param cell_med: from stat['med'] of the cell (stat['med'] refers to a suite2p results obj); the y and x (respectively) coordinates
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

    def classify_cells_sz_bound(self, sz_border_path, stim, to_plot=True, title=None, flip=False, fig=None, ax=None, text=None):
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
        for _, s in enumerate(self.stat):
            in_seizure = self._InOutSz(cell_med=s['med'], sz_border_path=sz_border_path, to_plot=to_plot)

            if in_seizure is True:
                in_sz.append(s['original_index'])  # this is the s2p cell id
            elif in_seizure is False:
                out_sz.append(s['original_index'])  # this is the s2p cell id

        if flip:
            # pass
            in_sz_2 = in_sz
            in_sz_final = out_sz
            out_sz_final = in_sz_2
        else:
            in_sz_final = in_sz
            out_sz_final = out_sz

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

            # pj.plot_cell_loc(self, cells=[cell], show=False)
            # plot sz boundary points
            if fig is None:
                fig, ax = plt.subplots(figsize=[5,5])

            ax.scatter(x=xline[0], y=yline[0], facecolors='#1A8B9D')
            ax.scatter(x=xline[1], y=yline[1], facecolors='#B2D430')
            # fig.show()

            # plot SLM targets in sz boundary
            # coords_to_plot = [s['med'] for cell, s in enumerate(self.stat) if cell in in_sz_final]
            # read in avg stim image to use as the background
            avg_stim_img_path = '%s/%s_%s_stim-%s.tif' % (self.analysis_save_path + 'avg_stim_images', self.metainfo['date'], self.metainfo['trial'], stim)
            bg_img = tf.imread(avg_stim_img_path)
            fig, ax = aoplot.plot_cells_loc(self, cells=in_sz_final, fig=fig, ax=ax, title=title, show=False, background=bg_img, cmap='gray', text=text)

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
            return in_sz_final, out_sz_final, fig, ax
        else:
            return in_sz_final, out_sz_final

    def classify_slmtargets_sz_bound(self, sz_border_path, stim, to_plot=True, title=None, flip=False, fig=None, ax=None):
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
        for cell in range(len(self.target_coords_all)):
            x = self._InOutSz(cell_med=[self.target_coords_all[cell][1], self.target_coords_all[cell][0]], sz_border_path=sz_border_path, to_plot=to_plot)

            if x is True:
                in_sz.append(cell)
            elif x is False:
                out_sz.append(cell)

        if flip:
            in_sz_2 = in_sz
            in_sz = out_sz
            out_sz = in_sz_2


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

            # pj.plot_cell_loc(self, cells=[cell], show=False)
            # plot sz boundary points
            if fig is None:
                fig, ax = plt.subplots(figsize=[5,5])

            ax.scatter(x=xline[0], y=yline[0], facecolors='#1A8B9D')
            ax.scatter(x=xline[1], y=yline[1], facecolors='#B2D430')
            # fig.show()

            # plot SLM targets in sz boundary
            coords_to_plot = [self.target_coords_all[cell] for cell in in_sz]
            # read in avg stim image to use as the background
            avg_stim_img_path = '%s/%s_%s_stim-%s.tif' % (self.analysis_save_path + 'avg_stim_images', self.metainfo['date'], self.metainfo['trial'], stim)
            bg_img = tf.imread(avg_stim_img_path)
            fig, ax = aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot, fig=fig, ax=ax, cells=in_sz, title=title,
                                                  show=False, background=bg_img)
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
        if hasattr(self, 'slmtargets_sz_stim'):
            if stim in self.slmtargets_sz_stim.keys():
                if cell in self.slmtargets_sz_stim[stim]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            # return False  # not all expobj will have the sz boundary classes attr so for those just assume no seizure
            raise Exception(
                'cannot check for cell inside sz boundary because cell sz classification hasnot been performed yet')


class OnePhotonStim(TwoPhotonImaging):
    def __init__(self, data_path_base, date, animal_prep, trial, metainfo, analysis_save_path_base: str = None):
        paqs_loc = '%s%s_%s_%s.paq' % (
            data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
        tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
        tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
        self.pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (
            date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
        # paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
        new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved

        # make the necessary Analysis saving subfolder as well
        if analysis_save_path_base is None:
            analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]
        else:
            analysis_save_path = analysis_save_path_base + tiffs_loc_dir[-16:]

        # if os.path.exists(self.analysis_save_path):
        #     pass
        # elif os.path.exists(self.analysis_save_path[:-17]):
        #     os.mkdir(self.analysis_save_path)
        # elif os.path.exists(self.analysis_save_path[:-27]):
        #     os.mkdir(self.analysis_save_path[:-17])
        #     os.mkdir(self.analysis_save_path)

        print('\n-----Processing trial # %s-----' % trial)

        paths = [tiffs_loc_dir, tiffs_loc, paqs_loc]
        # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

        self.tiff_path_dir = paths[0]
        self.tiff_path = paths[1]
        self.paq_path = paths[2]
        TwoPhotonImaging.__init__(self, self.tiff_path_dir, self.tiff_path, self.paq_path, metainfo=metainfo,
                                  save_downsampled_tiff=True, analysis_save_path=analysis_save_path, quick=False)
        self.paqProcessing()

        # add all frames as bad frames incase want to include this trial in suite2p run
        paq = paq_read(file_path=self.paq_path, plot=False)
        self.bad_frames = frames_discard(paq=paq[0], input_array=None, total_frames=self.n_frames, discard_all=True)



        self.save_pkl(pkl_path=self.pkl_path)

        print('\n-----DONE OnePhotonStim init of trial # %s-----' % trial)


    def paqProcessing(self, **kwargs):

        print('\n-----processing paq file for 1p photostim...')

        print('loading', self.paq_path)

        paq, _ = paq_read(self.paq_path, plot=True)
        self.paq_rate = paq['rate']

        frame_rate = self.fps / self.n_planes

        # if 'shutter_loopback' in paq['chan_names']:
        #     ans = input('shutter_loopback in this paq found, should we continue')
        #     if ans is True or 'Yes':
        #         pass
        #     else:
        #         raise Exception('need to write code for using the shutter loopback')

        # find frame_clock times
        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = pj.threshold_detect(clock_voltage, 1)
        self.frame_clock = frame_clock

        # find start and stop frame_clock times -- there might be multiple 2p imaging starts/stops in the paq trial (hence multiple frame start and end times)
        self.frame_start_times = [self.frame_clock[0]]  # initialize list
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

        plt.figure(figsize=(50, 2))
        plt.plot(clock_voltage)
        plt.plot(frame_clock, np.ones(len(frame_clock)), '.', color='orange')
        plt.plot(self.frame_clock_actual, np.ones(len(self.frame_clock_actual)), '.', color='red')
        plt.suptitle('frame clock from paq, with detected frame clock instances as scatter')
        plt.show()

        # find 1p stim times
        opto_loopback_chan = paq['chan_names'].index('opto_loopback')
        stim_volts = paq['data'][opto_loopback_chan, :]
        stim_times = pj.threshold_detect(stim_volts, 1)

        self.stim_times = stim_times
        self.stim_start_times = [self.stim_times[0]]  # initialize list
        self.stim_end_times = []
        i = len(self.stim_start_times)
        for stim in self.stim_times[1:]:
            if (stim - self.stim_start_times[i - 1]) > 1e5:
                i += 1
                self.stim_start_times.append(stim)
                self.stim_end_times.append(self.stim_times[np.where(self.stim_times == stim)[0] - 1][0])
        self.stim_end_times.append(self.stim_times[-1])

        print("\nNumber of 1photon stims found: ", len(self.stim_start_times))

        plt.figure(figsize=(50, 2))
        plt.plot(stim_volts)
        plt.plot(stim_times, np.ones(len(stim_times)), '.')
        plt.suptitle('1p stims from paq, with detected 1p stim instances as scatter')
        plt.xlim([stim_times[0] - 2e3, stim_times[-1] + 2e3])
        plt.show()

        # find all stim frames
        self.stim_frames = []
        for plane in range(self.n_planes):
            for stim in range(len(self.stim_start_times)):
                stim_frames_ = [frame for frame, t in enumerate(self.frame_clock_actual[plane::self.n_planes]) if
                                self.stim_start_times[stim] - 100 / self.paq_rate <= t <= self.stim_end_times[
                                    stim] + 100 / self.paq_rate]

                self.stim_frames.append(stim_frames_)

        # if >1 1p stims per trial, find the start of all 1p trials
        self.stim_start_frames = [stim_frames[0] for stim_frames in self.stim_frames if len(stim_frames) > 0]
        self.stim_end_frames = [stim_frames[-1] for stim_frames in self.stim_frames if len(stim_frames) > 0]
        self.stim_duration_frames = int(np.mean(
            [self.stim_end_frames[idx] - self.stim_start_frames[idx] for idx in range(len(self.stim_start_frames))]))

        print("\nStim duration of 1photon stim: %s frames (%s ms)" % (self.stim_duration_frames, round(self.stim_duration_frames / self.fps * 1000)))


        # i = len(self.stim_start_frames)
        # for stim in self.stim_frames[1:]:
        #     if (stim - self.stim_start_frames[i-1]) > 100:
        #         i += 1
        #         self.stim_start_frames.append(stim)

        # find shutter loopback frames
        if 'shutter_loopback' in paq['chan_names']:
            shutter_idx = paq['chan_names'].index('shutter_loopback')
            shutter_voltage = paq['data'][shutter_idx, :]

            shutter_times = np.where(shutter_voltage > 4)
            self.shutter_times = shutter_times[0]
            self.shutter_frames = []
            self.shutter_start_frames = []
            self.shutter_end_frames = []
            for plane in range(self.n_planes):
                shutter_frames_ = [frame for frame, t in enumerate(self.frame_clock_actual[plane::self.n_planes]) if
                                   t in self.shutter_times]
                self.shutter_frames.append(shutter_frames_)

                shutter_start_frames = [shutter_frames_[0]]
                shutter_end_frames = []
                i = len(shutter_start_frames)
                for frame in shutter_frames_[1:]:
                    if (frame - shutter_start_frames[i - 1]) > 5:
                        i += 1
                        shutter_start_frames.append(frame)
                        shutter_end_frames.append(shutter_frames_[shutter_frames_.index(frame) - 1])
                shutter_end_frames.append(shutter_frames_[-1])
                self.shutter_start_frames.append(shutter_start_frames)
                self.shutter_end_frames.append(shutter_end_frames)

        # # sanity check
        # assert max(self.stim_start_frames[0]) < self.raw[plane].shape[1] * self.n_planes

        # find voltage channel and save as lfp_signal attribute
        voltage_idx = paq['chan_names'].index('voltage')
        self.lfp_signal = paq['data'][voltage_idx]

    def collect_seizures_info(self, seizures_lfp_timing_matarray=None, discard_all=True):
        print('\ncollecting information about seizures...')
        self.seizures_lfp_timing_matarray = seizures_lfp_timing_matarray  # path to the matlab array containing paired measurements of seizures onset and offsets

        # retrieve seizure onset and offset times from the seizures info array input
        paq = paq_read(file_path=self.paq_path, plot=False)

        # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
        # NOTE: the output of all of the following function is in dimensions of the FRAME CLOCK (not official paq clock time)
        if seizures_lfp_timing_matarray is not None:
            print('-- using matlab array to collect seizures %s: ' % seizures_lfp_timing_matarray)
            bad_frames, self.seizure_frames, self.seizure_lfp_onsets, self.seizure_lfp_offsets = frames_discard(
                paq=paq[0], input_array=seizures_lfp_timing_matarray, total_frames=self.n_frames,
                discard_all=discard_all)
        else:
            print('-- no matlab array given to use for finding seizures.')
            self.seizure_frames = []
            bad_frames = frames_discard(paq=paq[0], input_array=seizures_lfp_timing_matarray,
                                        total_frames=self.n_frames,
                                        discard_all=discard_all)

        print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
        print('|- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])

        if seizures_lfp_timing_matarray is not None:
            # print('|-now creating raw movies for each sz as well (saved to the /Analysis folder) ... ')
            # self.subselect_tiffs_sz(onsets=self.seizure_lfp_onsets, offsets=self.seizure_lfp_offsets,
            #                         on_off_type='lfp_onsets_offsets')

            print('|-now classifying photostims at phases of seizures ... ')
            self.stims_in_sz = [stim for stim in self.stim_start_frames if stim in self.seizure_frames]
            self.stims_out_sz = [stim for stim in self.stim_start_frames if stim not in self.seizure_frames]
            self.stims_bf_sz = [stim for stim in self.stim_start_frames
                                  for sz_start in self.seizure_lfp_onsets
                                  if -2 * self.fps < (
                                              sz_start - stim) < 2 * self.fps]  # select stims that occur within 2 seconds before of the sz onset
            self.stims_af_sz = [stim for stim in self.stim_start_frames
                                  for sz_start in self.seizure_lfp_offsets
                                  if -2 * self.fps < -1 * (
                                              sz_start - stim) < 2 * self.fps]  # select stims that occur within 2 seconds afterof the sz offset
            print(' \n|- stims_in_sz:', self.stims_in_sz, ' \n|- stims_out_sz:', self.stims_out_sz,
                  ' \n|- stims_bf_sz:', self.stims_bf_sz, ' \n|- stims_af_sz:', self.stims_af_sz)

        else:
            print('|- No matlab measurement array given so setting all stims as outside of sz ... ')
            self.stims_in_sz = []
            self.stims_out_sz = [stim for stim in self.stim_start_frames if stim not in self.seizure_frames]
            self.stims_bf_sz = []
            self.stims_af_sz = []

        aoplot.plot_lfp_stims(self, x_axis='time')
        self.save_pkl()


class onePstim(TwoPhotonImaging):
    def __init__(self, data_path_base, date, animal_prep, trial, metainfo, analysis_save_path_base: str = None):
        paqs_loc = '%s%s_%s_%s.paq' % (
            data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
        tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
        tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
        self.pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (
            date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
        # paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
        new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved

        # make the necessary Analysis saving subfolder as well
        if analysis_save_path_base is None:
            analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]
        else:
            analysis_save_path = analysis_save_path_base + tiffs_loc_dir[-16:]

        # if os.path.exists(self.analysis_save_path):
        #     pass
        # elif os.path.exists(self.analysis_save_path[:-17]):
        #     os.mkdir(self.analysis_save_path)
        # elif os.path.exists(self.analysis_save_path[:-27]):
        #     os.mkdir(self.analysis_save_path[:-17])
        #     os.mkdir(self.analysis_save_path)

        print('\n-----Processing trial # %s-----' % trial)

        paths = [tiffs_loc_dir, tiffs_loc, paqs_loc]
        # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

        self.tiff_path_dir = paths[0]
        self.tiff_path = paths[1]
        self.paq_path = paths[2]
        TwoPhotonImaging.__init__(self, self.tiff_path_dir, self.tiff_path, self.paq_path, metainfo=metainfo,
                                  save_downsampled_tiff=True, analysis_save_path=analysis_save_path, quick=False)
        self.paqProcessing()

        # add all frames as bad frames incase want to include this trial in suite2p run
        paq = paq_read(file_path=self.paq_path, plot=False)
        self.bad_frames = frames_discard(paq=paq[0], input_array=None, total_frames=self.n_frames, discard_all=True)

        self.save_pkl(pkl_path=self.pkl_path)

        print('\n-----DONE OnePhotonStim init of trial # %s-----' % trial)


    def paqProcessing(self, **kwargs):

        print('\n-----processing paq file for 1p photostim...')

        print('loading', self.paq_path)

        paq, _ = paq_read(self.paq_path, plot=True)
        self.paq_rate = paq['rate']

        frame_rate = self.fps / self.n_planes

        # if 'shutter_loopback' in paq['chan_names']:
        #     ans = input('shutter_loopback in this paq found, should we continue')
        #     if ans is True or 'Yes':
        #         pass
        #     else:
        #         raise Exception('need to write code for using the shutter loopback')

        # find frame_clock times
        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = pj.threshold_detect(clock_voltage, 1)
        self.frame_clock = frame_clock

        # find start and stop frame_clock times -- there might be multiple 2p imaging starts/stops in the paq trial (hence multiple frame start and end times)
        self.frame_start_times = [self.frame_clock[0]]  # initialize list
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

        plt.figure(figsize=(50, 2))
        plt.plot(clock_voltage)
        plt.plot(frame_clock, np.ones(len(frame_clock)), '.', color='orange')
        plt.plot(self.frame_clock_actual, np.ones(len(self.frame_clock_actual)), '.', color='red')
        plt.suptitle('frame clock from paq, with detected frame clock instances as scatter')
        plt.show()

        # find 1p stim times
        opto_loopback_chan = paq['chan_names'].index('opto_loopback')
        stim_volts = paq['data'][opto_loopback_chan, :]
        stim_times = pj.threshold_detect(stim_volts, 1)

        self.stim_times = stim_times
        self.stim_start_times = [self.stim_times[0]]  # initialize list
        self.stim_end_times = []
        i = len(self.stim_start_times)
        for stim in self.stim_times[1:]:
            if (stim - self.stim_start_times[i - 1]) > 1e5:
                i += 1
                self.stim_start_times.append(stim)
                self.stim_end_times.append(self.stim_times[np.where(self.stim_times == stim)[0] - 1][0])
        self.stim_end_times.append(self.stim_times[-1])

        print("\nNumber of 1photon stims found: ", len(self.stim_start_times))

        plt.figure(figsize=(50, 2))
        plt.plot(stim_volts)
        plt.plot(stim_times, np.ones(len(stim_times)), '.')
        plt.suptitle('1p stims from paq, with detected 1p stim instances as scatter')
        plt.xlim([stim_times[0] - 2e3, stim_times[-1] + 2e3])
        plt.show()

        # find all stim frames
        self.stim_frames = []
        for plane in range(self.n_planes):
            for stim in range(len(self.stim_start_times)):
                stim_frames_ = [frame for frame, t in enumerate(self.frame_clock_actual[plane::self.n_planes]) if
                                self.stim_start_times[stim] - 100 / self.paq_rate <= t <= self.stim_end_times[
                                    stim] + 100 / self.paq_rate]

                self.stim_frames.append(stim_frames_)

        # if >1 1p stims per trial, find the start of all 1p trials
        self.stim_start_frames = [stim_frames[0] for stim_frames in self.stim_frames if len(stim_frames) > 0]
        self.stim_end_frames = [stim_frames[-1] for stim_frames in self.stim_frames if len(stim_frames) > 0]
        self.stim_duration_frames = int(np.mean(
            [self.stim_end_frames[idx] - self.stim_start_frames[idx] for idx in range(len(self.stim_start_frames))]))

        print("\nStim duration of 1photon stim: %s frames (%s ms)" % (self.stim_duration_frames, round(self.stim_duration_frames / self.fps * 1000)))


        # i = len(self.stim_start_frames)
        # for stim in self.stim_frames[1:]:
        #     if (stim - self.stim_start_frames[i-1]) > 100:
        #         i += 1
        #         self.stim_start_frames.append(stim)

        # find shutter loopback frames
        if 'shutter_loopback' in paq['chan_names']:
            shutter_idx = paq['chan_names'].index('shutter_loopback')
            shutter_voltage = paq['data'][shutter_idx, :]

            shutter_times = np.where(shutter_voltage > 4)
            self.shutter_times = shutter_times[0]
            self.shutter_frames = []
            self.shutter_start_frames = []
            self.shutter_end_frames = []
            for plane in range(self.n_planes):
                shutter_frames_ = [frame for frame, t in enumerate(self.frame_clock_actual[plane::self.n_planes]) if
                                   t in self.shutter_times]
                self.shutter_frames.append(shutter_frames_)

                shutter_start_frames = [shutter_frames_[0]]
                shutter_end_frames = []
                i = len(shutter_start_frames)
                for frame in shutter_frames_[1:]:
                    if (frame - shutter_start_frames[i - 1]) > 5:
                        i += 1
                        shutter_start_frames.append(frame)
                        shutter_end_frames.append(shutter_frames_[shutter_frames_.index(frame) - 1])
                shutter_end_frames.append(shutter_frames_[-1])
                self.shutter_start_frames.append(shutter_start_frames)
                self.shutter_end_frames.append(shutter_end_frames)

        # # sanity check
        # assert max(self.stim_start_frames[0]) < self.raw[plane].shape[1] * self.n_planes

        # find voltage channel and save as lfp_signal attribute
        voltage_idx = paq['chan_names'].index('voltage')
        self.lfp_signal = paq['data'][voltage_idx]

    def collect_seizures_info(self, seizures_lfp_timing_matarray=None, discard_all=True):
        print('\ncollecting information about seizures...')
        self.seizures_lfp_timing_matarray = seizures_lfp_timing_matarray  # path to the matlab array containing paired measurements of seizures onset and offsets

        # retrieve seizure onset and offset times from the seizures info array input
        paq = paq_read(file_path=self.paq_path, plot=False)

        # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
        # NOTE: the output of all of the following function is in dimensions of the FRAME CLOCK (not official paq clock time)
        if seizures_lfp_timing_matarray is not None:
            print('-- using matlab array to collect seizures %s: ' % seizures_lfp_timing_matarray)
            bad_frames, self.seizure_frames, self.seizure_lfp_onsets, self.seizure_lfp_offsets = frames_discard(
                paq=paq[0], input_array=seizures_lfp_timing_matarray, total_frames=self.n_frames,
                discard_all=discard_all)
        else:
            print('-- no matlab array given to use for finding seizures.')
            self.seizure_frames = []
            bad_frames = frames_discard(paq=paq[0], input_array=seizures_lfp_timing_matarray,
                                        total_frames=self.n_frames,
                                        discard_all=discard_all)

        print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
        print('|- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])

        if seizures_lfp_timing_matarray is not None:
            # print('|-now creating raw movies for each sz as well (saved to the /Analysis folder) ... ')
            # self.subselect_tiffs_sz(onsets=self.seizure_lfp_onsets, offsets=self.seizure_lfp_offsets,
            #                         on_off_type='lfp_onsets_offsets')

            print('|-now classifying photostims at phases of seizures ... ')
            self.stims_in_sz = [stim for stim in self.stim_start_frames if stim in self.seizure_frames]
            self.stims_out_sz = [stim for stim in self.stim_start_frames if stim not in self.seizure_frames]
            self.stims_bf_sz = [stim for stim in self.stim_start_frames
                                  for sz_start in self.seizure_lfp_onsets
                                  if -2 * self.fps < (
                                              sz_start - stim) < 2 * self.fps]  # select stims that occur within 2 seconds before of the sz onset
            self.stims_af_sz = [stim for stim in self.stim_start_frames
                                  for sz_start in self.seizure_lfp_offsets
                                  if -2 * self.fps < -1 * (
                                              sz_start - stim) < 2 * self.fps]  # select stims that occur within 2 seconds afterof the sz offset
            print(' \n|- stims_in_sz:', self.stims_in_sz, ' \n|- stims_out_sz:', self.stims_out_sz,
                  ' \n|- stims_bf_sz:', self.stims_bf_sz, ' \n|- stims_af_sz:', self.stims_af_sz)

        else:
            print('|- No matlab measurement array given so setting all stims as outside of sz ... ')
            self.stims_in_sz = []
            self.stims_out_sz = [stim for stim in self.stim_start_frames if stim not in self.seizure_frames]
            self.stims_bf_sz = []
            self.stims_af_sz = []

        aoplot.plot_lfp_stims(self, x_axis='time')
        self.save_pkl()


# RESULTS OBJECTS
class OnePhotonResults:
    def __init__(self, save_path: str):
        # just create an empty class object that you will throw results and analyses into
        self.pkl_path = save_path

        self.save(pkl_path=self.pkl_path)

    def save(self, pkl_path: str = None):
        TwoPhotonImaging.save_pkl(self, pkl_path=pkl_path)


class AllOpticalResults:
    def __init__(self, save_path: str):
        # just create an empty class object that you will throw results and analyses into
        self.pkl_path = save_path

        self.save(pkl_path=self.pkl_path)

    def save(self, pkl_path: str = None):
        TwoPhotonImaging.save_pkl(self, pkl_path=pkl_path)
#

########
# import expobj from the pkl file

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


def s2pMaskStack(obj, pkl_list, s2p_path, parent_folder, force_redo: bool = False):
    """makes a TIFF stack with the s2p mean image, and then suite2p ROI masks for all cells detected, target cells, and also SLM targets as well?"""

    if force_redo:
        continu = True
    elif hasattr(obj, 's2p_cell_targets'):
        print('skipped re-making TIFF stack of finding s2p targets from suite2p cell list')
        continu = False
    else:
        continu = True

    if continu:

        for pkl in pkl_list:
            expobj = obj

            print('Retrieving s2p masks for:', pkl, end='\r')

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
            print('\ns2p ROI + photostim targets masks saved in TIFF to: ', save_path)


# other functions written by me

# PRE-PROCESSING FUNCTIONS - need to delete old functions (i.e. ones that have been moved under classes above)
# @njit
def moving_average(a, n=4):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# - moved this code to the twophoton code class
def _good_cells(cell_ids: list, raws: np.ndarray, photostim_frames: list, std_thresh: int, radiuses: list = None,
                min_radius_pix: int = 2.5, max_radius_pix: int = 10):
    """
    This function is used for filtering for "good" cells based on detection of Flu deflections that are above some std threshold based on std_thresh.
    Note: a moving averaging window of 4 frames is used to find Flu deflections above std threshold.

    :param cell_ids: list of cell ids to iterate over
    :param raws: raw flu values corresponding to the cell list in cell_ids
    :param photostim_frames: frames to delete from the raw flu traces across all cells (because they are photostim frames)
    :param std_thresh: std. factor above which to define reliable Flu events
    :param radiuses: radiuses of the s2p ROI mask of all cells in the same order as cell_ids
    :param min_radius_pix:
    :param max_radius_pix:
    :return:
        good_cells: list of cell_ids that had atleast 1 event above the std_thresh
        events_loc_cells: dictionary containing locations for each cell_id where the moving averaged Flu trace passes the std_thresh
        flu_events_cells: dictionary containing the dff Flu value corresponding to events_loc_cells
        stds = dictionary containing the dFF trace std value for each cell_id
    """

    good_cells = []
    events_loc_cells = {}
    flu_events_cells = {}
    stds = {}  # collect the std values for all filtered cells used later for identifying high and low std cells
    for i in range(len(cell_ids)):
        cell_id = cell_ids[i]

        if i % 100 == 0:  # print the progress once every 100 cell iterations
            print(i, " out of ", len(cell_ids), " cells done", end='\r')

        # print(i, " out of ", len(cell_ids), " cells")
        raw = raws[i]
        if np.mean(raw) > 1:  # exclude all negative raw traces and very small mean raw traces
            raw_ = np.delete(raw, photostim_frames)
            raw_dff = normalize_dff_jit(raw_)  # note that this function is defined in this file a little further down
            std_ = raw_dff.std()

            raw_dff_ = moving_average(raw_dff, n=4)

            thr = np.mean(raw_dff) + std_thresh * std_
            events = np.where(raw_dff_ > thr)
            flu_values = raw_dff[events]

            if radiuses is not None:
                radius = radiuses[i]
                if len(events[0]) > 0 and radius > min_radius_pix and radius < max_radius_pix:
                    events_loc_cells[cell_id] = events
                    flu_events_cells[cell_id] = flu_values
                    stds[cell_id] = std_
                    good_cells.append(cell_id)
            elif len(events[0]) > 0:
                events_loc_cells[cell_id] = events
                flu_events_cells[cell_id] = flu_values
                good_cells.append(cell_id)
                stds[cell_id] = std_

            # if i == 465:  # just using this here if ever need to check back with specific cells if function seems to be misbehaving
            #     print(events, len(events[0]), thr)

    print('# of good cells found: ', len(good_cells), ' (out of ', len(cell_ids), ' ROIs)')
    return good_cells, events_loc_cells, flu_events_cells, stds


def get_s2ptargets_stim_traces(expobj, normalize_to='', pre_stim=10, post_stim=200):
    """
    primary function to measure the dFF traces for photostimulated targets.
    :param expobj:
    :param normalize_to: str; either "baseline" or "pre-stim"
    :param pre_stim: number of frames to use as pre-stim
    :param post_stim: number of frames to use as post-stim
    :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
    """
    stim_timings = expobj.stim_start_frames
    # targeted_cells = [cell for cell in expobj.s2p_cell_targets if cell in expobj.good_cells]
    targeted_cells = expobj.s2p_cell_targets

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
                for i in range(len(flu)):
                    trace_dff = ((flu[i] - mean_spont_baseline) / mean_spont_baseline) * 100
                    # add nan if cell is inside sz boundary for this stim
                    if hasattr(expobj, 'slmtargets_sz_stim'):
                        if expobj.is_cell_insz(cell=cell, stim=stim_timings[i]):
                            trace_dff = [np.nan] * len(flu[i])
                    flu_dff.append(trace_dff)
            elif normalize_to == 'pre-stim':
                for i in range(len(flu)):
                    trace = flu[i]
                    mean_pre = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean_pre) / mean_pre) * 100
                    std_pre = np.std(trace[0:pre_stim])
                    dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                    # add nan if cell is inside sz boundary for this stim
                    if hasattr(expobj, 'slmtargets_sz_stim'):
                        if expobj.is_cell_insz(cell=cell, stim=stim_timings[i]):
                            trace_dff = [np.nan] * len(trace)
                            dFstdF = [np.nan] * len(trace)
                    flu_dfstdF.append(dFstdF)
                    flu_dff.append(trace_dff)
            else:
                TypeError('need to specify what to normalize to in get_targets_dFF (choose "baseline" or "pre-stim")')

            targets_dff.append(flu_dff)  # contains all individual dFF traces for all stim times
            targets_dff_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

            targets_dfstdF.append(flu_dfstdF)
            targets_dfstdF_avg.append(np.nanmean(flu_dfstdF, axis=0))

            targets_raw.append(flu)
            targets_raw_avg.append(np.nanmean(flu, axis=0))

    if normalize_to == 'baseline':
        return targets_dff, targets_dff_avg
    elif normalize_to == 'pre-stim':
        return targets_dff, targets_dff_avg, targets_dfstdF, targets_dfstdF_avg, targets_raw, targets_raw_avg


# moved to class method under alloptical
def get_nontargets_stim_traces_norm(expobj, normalize_to='', pre_stim=10, post_stim=50):
    """
    primary function to retrieve photostimulation trial timed Fluorescence traces for non-targets (ROIs taken from suite2p).
    :param expobj: alloptical experiment object
    :param normalize_to: str; either "baseline" or "pre-stim"
    :param pre_stim: number of frames to use as pre-stim
    :param post_stim: number of frames to use as post-stim
    :param stim_idx_l: list of indexes of stims to use for calculating avg stim response
    :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
    """
    stim_timings = expobj.stim_start_frames
    expobj.s2p_nontargets = [cell for cell in expobj.good_cells if cell not in expobj.s2p_cell_targets]

    # collect photostim timed average dff traces of photostim targets
    dff_traces = []
    dff_traces_avg = []

    dfstdF_traces = []
    dfstdF_traces_avg = []

    raw_traces = []
    raw_traces_avg = []
    for cell in expobj.s2p_nontargets:
        # print('considering cell # %s' % cell)
        cell_idx = expobj.cell_id.index(cell)
        flu = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
               stim not in expobj.seizure_frames]

        flu_dfstdF = []
        flu_dff = []
        if normalize_to == 'baseline':  # probably gonna ax this anyways
            mean_spont_baseline = np.mean(expobj.baseline_raw[cell_idx])
            for i in range(len(flu)):
                trace_dff = ((flu[i] - mean_spont_baseline) / mean_spont_baseline) * 100

                # add nan if cell is inside sz boundary for this stim
                if hasattr(expobj, 'slmtargets_sz_stim'):
                    if expobj.is_cell_insz(cell=cell, stim=stim_timings[i]):
                        trace_dff = [np.nan] * len(flu[i])

                flu_dff.append(trace_dff)

        elif normalize_to == 'pre-stim':
            for i in range(len(flu)):
                trace = flu[i]
                mean_pre = np.mean(trace[0:pre_stim])
                # trace_dff = ((trace - mean_pre) / mean_pre) * 100
                percentile = pj.find_percentile(trace, mean_pre)
                trace_dff = normalize_dff(trace, threshold=percentile)
                std_pre = np.std(trace[0:pre_stim])
                dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace

                # add nan if cell is inside sz boundary for this stim
                if 'post' in expobj.metainfo['exptype']:
                    if hasattr(expobj, 'slmtargets_sz_stim'):
                        if expobj.is_cell_insz(cell=cell, stim=stim_timings[i]):
                            trace_dff = [np.nan] * len(trace)
                            dFstdF = [np.nan] * len(trace)
                    else:
                        AttributeError('no slmtargets_sz_stim attr, so classify cells in sz boundary hasnot been saved for this expobj')

                flu_dfstdF.append(dFstdF)
                flu_dff.append(trace_dff)

        else:
            TypeError('need to specify what to normalize to in get_targets_dFF (choose "baseline" or "pre-stim")')

        dff_traces.append(flu_dff)  # contains all individual dFF traces for all stim times
        dff_traces_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times

        dfstdF_traces.append(flu_dfstdF)
        dfstdF_traces_avg.append(np.nanmean(flu_dfstdF, axis=0))

        raw_traces.append(flu)
        raw_traces_avg.append(np.nanmean(flu, axis=0))

    if normalize_to == 'baseline':
        print(
            '\nCompleted collecting pre to post stim traces -- normalized to spont imaging as baseline -- for %s cells' % len(
                dff_traces_avg))
        return dff_traces, dff_traces_avg
    elif normalize_to == 'pre-stim':
        print('\nCompleted collecting pre to post stim traces -- normalized to pre-stim period -- for %s cells' % len(
            dff_traces_avg))
        return dff_traces, dff_traces_avg, dfstdF_traces, dfstdF_traces_avg, raw_traces, raw_traces_avg


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
        # response = np.mean(dF_stdF[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 3*expobj.stim_duration_frames])
        response = np.mean(trace[
                           pre_stim + expobj.stim_duration_frames:pre_stim + 3 * expobj.stim_duration_frames])  # calculate the dF over pre-stim mean F response within the response window

        if to_plot is not None:
            if cell in targeted_cells[:to_plot]:
                idx = targeted_cells[:to_plot].index(cell)
                axes[idx].plot(trace)
                axes[idx].axhspan(mean_pre + 0 * std_pre, mean_pre + std_thresh * std_pre, facecolor='0.25')
                axes[idx].axvspan(pre_stim + expobj.stim_duration_frames, pre_stim + 3 * expobj.stim_duration_frames,
                                  facecolor='0.25')
                axes[idx].title.set_text('Average trace (%s) across all photostims - cell #%s' % (x_, cell))

        # post_stim_trace = trace[pre_stim_sec + expobj.stim_duration_frames:post_stim_sec]
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

    print('|- %s cells out of %s s2p target cells photostim. responses above threshold' % (
        len(good_photostim_cells), len(targeted_cells)))
    total += len(good_photostim_cells)
    total_considered += len(targeted_cells)

    expobj.good_photostim_cells_all = [y for x in expobj.good_photostim_cells for y in x]
    print('|- Total number of good photostim responsive cells found: %s (out of %s s2p photostim target cells)' % (
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


def normalize_dff(arr, threshold_pct=20, threshold_val=None):
    """normalize given array (cells x time) to the mean of the fluorescence values below given threshold. Threshold
    will refer to the that lower percentile of the given trace."""

    if arr.ndim == 1:
        if threshold_val is None:
            a = np.percentile(arr, threshold_pct)
            mean_ = arr[arr < a].mean()
        else:
            mean_ = threshold_val
        # mean_ = abs(arr[arr < a].mean())
        new_array = ((arr - mean_) / mean_) * 100
        if np.isnan(new_array).any() == True:
            Warning('Cell (unknown) contains nan, normalization factor: %s ' % mean_)

    else:
        new_array = np.empty_like(arr)
        for i in range(len(arr)):
            if threshold_val is None:
                a = np.percentile(arr[i], threshold_pct)
            else:
                a = threshold_val
            mean_ = np.mean(arr[i][arr[i] < a])
            new_array[i] = ((arr[i] - mean_) / abs(mean_)) * 100

            if np.isnan(new_array[i]).any() == True:
                print('Warning:')
                print('Cell %d: contains nan' % (i + 1))
                print('      Mean of the sub-threshold for this cell: %s' % mean_)

    return new_array


# @jit
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
            new_array[i] = (arr[i] - mean_) / abs(mean_) * 100

            if np.isnan(new_array[i]).any() == True:
                print('Warning:')
                print('Cell %d: contains nan' % (i + 1))
                print('      Mean of the sub-threshold for this cell: %s' % mean_)

    return new_array


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



# %% STATISTICS AND OTHER ANALYSES FUNCTIONS

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




# %% main functions used to initiate, run processing and analysis (including some plotting) of experiments -- these functions are run on expobj loaded from .pkl files

# for pre-processing PHOTOSTIM. experiments, creates the all-optical expobj saved in a pkl files at imaging tiff's loc - BEFORE running suite2p
def calculate_StimSuccessRate(expobj, cell_ids: list, raw_traces_stims=None, dfstdf_threshold=None,
                              post_stim_response_frames_window=10,
                              dff_threshold=None, pre_stim=10, sz_filter=False, verbose=False, plot=False):
    """calculates the percentage of successful photoresponsive trials for each targeted cell, where success is post
     stim response over the dff_threshold. the filter_for_sz argument is set to True when needing to filter out stim timings
     that occured when the cell was classified as inside the sz boundary."""

    print('Calculating success rate of stims. of all stims across the trial')
    reliability_cells = {}  # dict will be used to store the reliability results for each targeted cell
    hits_cells = {}
    responses_cells = {}
    targets_dff_all_stimtrials = {}  # dict will contain the peri-stim dFF for each cell by the cell_idx
    stim_timings = expobj.stim_start_frames

    # assert list(stim_timings) == list(expobj.slmtargets_sz_stim.keys())  # dont really need this assertion because you wont necessarily always look at the sz boundary for all stims every trial
    # stim_timings = expobj.slmtargets_sz_stim.keys()
    if dff_threshold:
        threshold = round(dff_threshold)
        # dff = True
        if raw_traces_stims is None:
            df = expobj.dff_responses_all_cells
    elif dfstdf_threshold:
        threshold = dfstdf_threshold
        # dff = False
        if raw_traces_stims is None:
            df = expobj.dfstdf_all_cells
    else:
        raise Exception("need to specify either dff_threshold or dfstdf_threshold value to use")

    # if you need to filter out cells based on their location inside the sz or not, you need a different approach
    # where you are for looping over each stim and calculating the reliability of cells like that. BUT the goal is still to collect reliability values by cell.

    if raw_traces_stims is None:
        for cell in expobj.s2p_cell_targets:
            # print('considering cell # %s' % cell)
            # if cell in expobj.cell_id:
            if sz_filter:
                if hasattr(expobj, 'slmtargets_sz_stim'):
                    stims_to_use = [str(stim) for stim in stim_timings
                                    if stim not in expobj.slmtargets_sz_stim.keys() or cell not in expobj.slmtargets_sz_stim[
                                        stim]]  # select only the stim times where the cell IS NOT inside the sz boundary
                else:
                    print(
                        'no information about cell locations in seizures by stim available, therefore not excluding any stims based on sz state')
                    stims_to_use = [str(stim) for stim in stim_timings]
            else:
                print('not excluding any stims based on sz state')
                stims_to_use = [str(stim) for stim in stim_timings]
            counter = len(stims_to_use)
            responses = df.loc[
                cell, stims_to_use]  # collect the appropriate responses for the current cell at the selected stim times
            success = sum(i >= threshold for i in responses)

            reliability_cells[cell] = success / counter * 100.
            reliability_cells[cell] = success / counter * 100.
            print(cell, reliability_cells[cell], 'calc over %s stims' % counter) if verbose else None

    # elif raw_traces_stims is not None:  # used primarily for calculating responses for SLM targets
    #     if sz_filter:
    #         warnings.warn(
    #             "the seizure filtering by *cells* functionality is only available for s2p defined cell targets as of now")
    #
    #     for idx in range(len(cell_ids)):
    #         success = 0
    #         counter = 0
    #         responses = []
    #         hits = []
    #         for trace in raw_traces_stims[idx]:
    #
    #             # calculate dFF (noramlized to pre-stim) for each trace
    #             pre_stim_mean = np.mean(trace[0:pre_stim_sec])
    #             std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #             response_trace = (trace - pre_stim_mean)
    #             # if dff_threshold:  # calculate dFF response for each stim trace
    #             #     response_trace = ((trace - pre_stim_mean)) #/ pre_stim_mean) * 100
    #             # else:  # calculate dF_stdF response for each stim trace
    #             #     pass
    #
    #             # calculate if the current trace beats the threshold for calculating reliability (note that this happens over a specific window just after the photostim)
    #             response = np.nanmean(response_trace[
    #                                   pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + expobj.stim_duration_frames + 1 + expobj.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
    #             if dfstdf_threshold:
    #                 response_result = response / std_pre  # normalize the delta F above pre-stim mean using std of the pre-stim
    #             else:
    #                 response_result = (response / pre_stim_mean) * 100  # calculate % of dFF response for each stim trace
    #             responses.append(round(response_result, 2))
    #             if response_result >= threshold:
    #                 success += 1
    #                 hits.append(counter)
    #             counter += 1
    #
    #         reliability_cells[idx] = round(success / counter * 100., 2)
    #         hits_cells[idx] = hits
    #         responses_cells[idx] = responses
    #         if verbose:
    #             print(
    #                 '|- Target # %s: %s percent hits over %s stims' % (cell_ids[idx], reliability_cells[idx], counter))
    #         if plot:
    #             random_select = np.random.randint(0, raw_traces_stims.shape[1],
    #                                               10)  # select just 10 random traces to show on the plot
    #             aoplot.plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dfstdF[idx][random_select], expobj=expobj,
    #                                           stim_duration=expobj.stim_duration_frames,
    #                                           x_label='frames', pre_stim_sec=pre_stim_sec, post_stim_sec=expobj.post_stim_sec,
    #                                           color='steelblue',
    #                                           y_lims=[-0.5, 2.5], show=False, title='Target ' + str(idx))
    #             m = expobj.stim_duration_frames + (3 * expobj.stim_duration_frames) / 2 - pre_stim_sec
    #             x = np.random.randn(len(responses)) * 1.5 + m
    #             plt.scatter(x, responses, c='chocolate', zorder=3, alpha=0.6)
    #             plt.show()

    else:
        raise Exception(
            "basically the error is that the raw traces provided weren't detected, or not provided at all")

        # old version
        # for cell in expobj.s2p_cell_targets:
        #     # print('considering cell # %s' % cell)
        #     if cell in expobj.cell_id:
        #         cell_idx = expobj.cell_id.index(cell)
        #         # collect a trace of prestim and poststim raw fluorescence for each stim time
        #         flu_all_stims = [expobj.raw[cell_idx][stim - pre_stim_sec: stim + post_stim_sec] for stim in stim_timings]
        #         success = 0
        #         counter = 0
        #         for trace in flu_all_stims:
        #             counter += 1
        #             # calculate dFF (noramlized to pre-stim) for each trace
        #             pre_stim_mean = np.mean(trace[0:pre_stim_sec])
        #             if dff:
        #                 response_trace = ((trace - pre_stim_mean) / pre_stim_mean) * 100
        #             elif not dff:
        #                 std_pre = np.std(trace[0:expobj.pre_stim_sec])
        #                 response_trace = ((trace - pre_stim_mean) / std_pre) * 100
        #
        #             # calculate if the current trace beats dff_threshold for calculating reliability (note that this happens over a specific window just after the photostim)
        #             response = np.nanmean(response_trace[
        #                                   pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 3 * expobj.stim_duration_frames])  # calculate the dF over pre-stim mean F response within the response window
        #             if response >= threshold:
        #                 success += 1
        #
        #         reliability[cell] = success / counter * 100.
        #         print(cell, reliability, 'calc over %s stims' % counter)

    print("\navg photostim. success rate is: %s pct." % (round(np.nanmean(list(reliability_cells.values())), 2)))
    return reliability_cells, hits_cells, responses_cells

def run_photostim_preprocessing(trial, exp_type, tiffs_loc_dir, tiffs_loc, naparms_loc, paqs_loc, pkl_path, metainfo,
                                new_tiffs, matlab_pairedmeasurements_path=None, processed_tiffs=True, discard_all=False, quick=False,
                                analysis_save_path=''):
    print('\n-----Processing trial # %s-----' % trial)

    paths = [[tiffs_loc_dir, tiffs_loc, naparms_loc, paqs_loc, analysis_save_path, matlab_pairedmeasurements_path]]
    for path in paths[0]:  # check that all paths required for processing run are legit and active
        try:
            assert os.path.exists(path)
        except AssertionError:
            print('we got an invalid path at: ', path)
    print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

    if 'post' in exp_type and '4ap' in exp_type:
        expobj = Post4ap(paths[0], metainfo=metainfo, stimtype='2pstim', discard_all=discard_all)
    else:
        expobj = alloptical(paths[0], metainfo=metainfo, stimtype='2pstim', quick=quick)

    # for key, values in vars(expobj).items():
    #     print(key)

    # # these functions are moved to the alloptical class init() location
    # expobj._parseNAPARMxml()
    # expobj._parseNAPARMgpl()
    # expobj._parsePVMetadata()
    # expobj.stimProcessing(stim_channel='markpoints2packio')
    # expobj._findTargets()
    # expobj.find_photostim_frames()

    # # collect information about seizures
    # if 'post' in exp_type and '4ap' in exp_type:
    #     expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path, discard_all=discard_all)

    if len(expobj.bad_frames) > 0:
        print('***  Collected a total of ', len(expobj.bad_frames),
              'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy  ***')

    # if matlab_pairedmeasurements_path is not None or discard_all is True:
    #     paq = paq_read(file_path=paqs_loc, plot=False)
    #     # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
    #     bad_frames, expobj.seizure_frames, _, _ = \
    #         frames_discard(paq=paq[0], input_array=matlab_pairedmeasurements_path,
    #                        total_frames=expobj.n_frames, discard_all=discard_all)
    #     print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
    #     print('|\n -- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])
    #     expobj.append_bad_frames(
    #         bad_frames=bad_frames)  # here only need to append the bad frames to the expobj.bad_frames property
    #
    # else:
    #     expobj.seizure_frames = []
    #     print('\nNo additional bad (seizure) frames needed for', tiffs_loc_dir)
    #
    # if len(expobj.bad_frames) > 0:
    #     print('***Saving a total of ', len(expobj.bad_frames),
    #           'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy***')
    #     np.save('%s/bad_frames.npy' % tiffs_loc_dir,
    #             expobj.bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p

    # Pickle the expobject output to save it for analysis

    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(expobj, f)
    # print("\nPkl saved to %s" % pkl_path)

    # make processed tiffs
    if processed_tiffs:
        rm_artifacts_tiffs(expobj, tiffs_loc=tiffs_loc, new_tiffs=new_tiffs)

    print('\n----- COMPLETED RUNNING run_photostim_processing() *******')
    print(metainfo)

    return expobj


###### SLM TARGETS analysis + plottings
# after running suite2p
def run_alloptical_processing_photostim(expobj, to_suite2p=None, baseline_trials=None, plots: bool = True, force_redo: bool = False):

    """
    main function for running processing photostim trace data collecting (e.g. dFF photostim trials pre- post stim).

    :param expobj: experimental object (usually from pkl file)
    :param to_suite2p: trials that were used in the suite2p run for this expobj
    :param baseline_trials: trials that were baseline (spontaneous pre-4ap) for this expobj
    :param plots: whether to plot results of processing where sub-function calls are appropriate
    :param force_redo: bool whether to redo some functions
    :param post_stim_response_window_msec: the length of the time window post stimulation that will be used for measuring the response magnitude.

    :return: n/a
    """

    if force_redo:
        expobj._findTargetsAreas()

        if not hasattr(expobj, 'meanRawFluTrace'):
            expobj.mean_raw_flu_trace(plot=True)

        if plots:
            aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
            # aoplot.plotLfpSignal(expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
            aoplot.plot_SLMtargets_Locs(expobj)
            aoplot.plot_lfp_stims(expobj)

        ####################################################################################################################
        # prep for importing data from suite2p for this whole experiment
        # determine which frames to retrieve from the overall total s2p output

        if not hasattr(expobj, 'suite2p_trials'):
            if to_suite2p is None:
                AttributeError('need to provide which trials were used in suite2p for this expobj, the attr. hasnt been set')
            if baseline_trials is None:
                AttributeError('need to provide which trials were baseline (spont imaging pre-4ap) for this expobj, the attr. hasnt been set')
            expobj.suite2p_trials = to_suite2p
            expobj.baseline_trials = baseline_trials
            expobj.save()

        # main function that imports suite2p data and starts adding attributes to the expobj
        expobj.subset_frames_current_trial(trial=expobj.metainfo['trial'], to_suite2p=expobj.suite2p_trials,
                                           baseline_trials=expobj.baseline_trials, force_redo=force_redo)
        expobj.s2pProcessing(s2p_path=expobj.s2p_path, subset_frames=expobj.curr_trial_frames, subtract_neuropil=True,
                             baseline_frames=expobj.baseline_frames, force_redo=True)
        # expobj.target_coords_all = expobj.target_coords
        expobj._findTargetedS2pROIs(force_redo=True)
        aoplot.s2pRoiImage(expobj)
        s2pMaskStack(obj=expobj, pkl_list=[expobj.pkl_path], s2p_path=expobj.s2p_path,
                     parent_folder=expobj.analysis_save_path, force_redo=force_redo)

        ####################################################################################################################
        # STA - raw SLM targets processing

        # collect raw Flu data from SLM targets
        expobj.raw_traces_from_targets(force_redo=force_redo)

    if plots:
        aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg, title='SLM targets location w/ mean Flu img')
        aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered,
                                    title='SLM targets location w/ registered mean Flu img')

    # collect SLM photostim individual targets -- individual, full traces, dff normalized
    expobj.dff_SLMTargets = normalize_dff(np.array(expobj.raw_SLMTargets))
    expobj.save()

    # collect and plot peri- photostim traces for individual SLM target, incl. individual traces for each stim
    # all stims (use for pre-4ap trials)
    if 'pre' in expobj.metainfo['exptype']:
        expobj.SLMTargets_stims_dff, expobj.SLMTargets_stims_dffAvg, expobj.SLMTargets_stims_dfstdF, \
        expobj.SLMTargets_stims_dfstdF_avg, expobj.SLMTargets_stims_raw, expobj.SLMTargets_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, stims=expobj.stim_start_frames)

        SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))

    # only out of sz stims (use for post-4ap trials)
    elif 'post' in expobj.metainfo['exptype']:

        expobj.SLMTargets_stims_dff, expobj.SLMTargets_stims_dffAvg, expobj.SLMTargets_stims_dfstdF, \
        expobj.SLMTargets_stims_dfstdF_avg, expobj.SLMTargets_stims_raw, expobj.SLMTargets_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(pre_stim=expobj.pre_stim, post_stim=expobj.post_stim,
                                                   stims=expobj.stim_start_frames)

        stims = [stim for stim in expobj.stim_start_frames if stim not in expobj.seizure_frames]
        expobj.SLMTargets_stims_dff_outsz, expobj.SLMTargets_stims_dffAvg_outsz, expobj.SLMTargets_stims_dfstdF_outsz, \
        expobj.SLMTargets_stims_dfstdF_avg_outsz, expobj.SLMTargets_stims_raw_outsz, expobj.SLMTargets_stims_rawAvg_outsz = \
            expobj.get_alltargets_stim_traces_norm(pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, stims=stims)

    # only in sz stims (use for post-4ap trials) - includes exclusion of cells inside of sz boundary
        if hasattr(expobj, 'stims_in_sz'):
            stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
            expobj.SLMTargets_stims_dff_insz, expobj.SLMTargets_stims_dffAvg_insz, expobj.SLMTargets_stims_dfstdF_insz, \
            expobj.SLMTargets_stims_dfstdF_avg_insz, expobj.SLMTargets_stims_raw_insz, expobj.SLMTargets_stims_rawAvg_insz = \
                expobj.get_alltargets_stim_traces_norm(pre_stim=expobj.pre_stim, post_stim=expobj.post_stim, stims=stims, filter_sz=True)

    else:
        raise Exception('something very weird has happened. exptype for expobj not defined as pre or post 4ap [1.]')

    # photostim. SUCCESS RATE MEASUREMENTS and PLOT - SLM PHOTOSTIM TARGETED CELLS
    # measure, for each cell, the pct of trials in which the dF_stdF > 20% post stim (normalized to pre-stim avgF for the trial and cell)
    # can plot this as a bar plot for now showing the distribution of the reliability measurement


    if 'post' in expobj.metainfo['exptype']:
        seizure_filter = True

        print('\n Calculating stim success rates and response magnitudes (of SLM targets) *********** [2.1]')
        expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
            expobj.calculate_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stim_start_frames)

        if hasattr(expobj, 'stims_out_sz'):
            stims_outsz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
            if len(stims_outsz_idx) > 0:
                print('|- calculating stim success rates (outsz) - %s stims [2.2]' % len(stims_outsz_idx))
                expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_traces_SLMtargets_successes_avg, \
                expobj.outsz_traces_SLMtargets_failures_avg = \
                    expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=stims_outsz_idx)

        if hasattr(expobj, 'stims_in_sz'):
            if hasattr(expobj, 'slmtargets_sz_stim'):
                stims_insz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                if len(stims_insz_idx) > 0:
                    print('|- calculating stim success rates (insz) - %s stims [2.3]' % len(stims_insz_idx))
                    expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_traces_SLMtargets_successes_avg,\
                        expobj.insz_traces_SLMtargets_failures_avg = \
                        expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=stims_insz_idx,
                                                                exclude_stims_targets=expobj.slmtargets_sz_stim)
            else:
                print('No slmtargets_sz_stim (sz boundary classification not done) for: %s %s' % (
                expobj.metainfo['animal prep.'], expobj.metainfo['trial']), ' [2.4] ')
                # delattr(expobj, 'insz_traces_SLMtargets_successes_avg')

        else:
            print('No stims in sz for: %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']), ' [2.5] ')


    elif 'pre' in expobj.metainfo['exptype']:
        seizure_filter = False
        print('\n Calculating stim success rates and response magnitudes (of SLM targets) *********** [3.]')
        expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
            expobj.calculate_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stim_start_frames)

        expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
        expobj.StimSuccessRate_SLMtargets, expobj.traces_SLMtargets_successes_avg, \
        expobj.traces_SLMtargets_failures_avg = \
            expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=expobj.stims_idx)


        expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]

        expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
        expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
            get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                                    post_stim=expobj.post_stim)


    expobj.save()

# creates plots for SLM targets responses
def slm_targets_responses(expobj, experiment, trial, y_spacing_factor=2, figsize=[20, 20], smooth_overlap_traces=5, linewidth_overlap_traces=0.2, dff_threshold=0.15,
                          y_lims_periphotostim_trace=[-0.5, 2.0], v_lims_periphotostim_heatmap=[-5, 5], save_results=True, force_redo=False, cmap=None):
    # plot SLM photostim individual targets -- individual, full traces, dff normalized

    # make rolling average for these plots to smooth out the traces a little more

    # force_redo = False
    if force_redo:
        # expobj._findTargets()
        # expobj.raw_traces_from_targets(force_redo=force_redo, save=True)
        # expobj.save()

        if hasattr(expobj, 'stims_in_sz'):
            seizure_filter = True

            # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
            if len(expobj.stims_out_sz) > 0:
                    expobj.StimSuccessRate_SLMtargets_outsz, expobj.hits_SLMtargets_outsz, expobj.responses_SLMtargets_outsz, expobj.traces_SLMtargets_successes_outsz = \
                    expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stims_out_sz)

                # expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
                #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=raw_traces_stims,
                #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
                #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
                #                               verbose=True, plot=False)

            # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
            if len(expobj.stims_in_sz) > 0:
                expobj.StimSuccessRate_SLMtargets_insz, expobj.hits_SLMtargets_insz, expobj.responses_SLMtargets_insz, expobj.traces_SLMtargets_successes_insz = \
                    expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stims_in_sz)

                # expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
                #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=raw_traces_stims,
                #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
                #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
                #                               verbose=True, plot=False)

        else:
            seizure_filter = False
            print('\n Calculating stim success rates and response magnitudes ***********')
            expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
                expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stim_start_frames)

            # expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
            #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=expobj.SLMTargets_stims_raw,
            #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
            #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
            #                               verbose=True, plot=False)

        expobj.save()

    else:
        pass


    ####################################################################################################################
    ## PLOTTING
    w = smooth_overlap_traces
    to_plot = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets])
    # to_plot = expobj.dff_SLMTargets
    # aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Time (secs.)',
    #                              y_label='dFF Flu', title=experiment)


    # initialize figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(4, 8)

    ax9 = fig.add_subplot(gs[0, 0])
    fig, ax9 = aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered, title=None, fig=fig, ax=ax9, show=False)


    ax0 = fig.add_subplot(gs[0, 1:])
    ax0 = aoplot.plot_lfp_stims(expobj, fig=fig, ax=ax0, show=False, x_axis='Time (secs.)')

    ax1 = fig.add_subplot(gs[1:3, :])
    aoplot.plot_photostim_traces_overlap(array=expobj.dff_SLMTargets, expobj=expobj, x_axis='Time (secs.)',
                                         y_spacing_factor=y_spacing_factor, fig=fig, ax=ax1, show=False,
                                         title='%s - dFF Flu photostims' % experiment, linewidth=linewidth_overlap_traces,
                                         figsize=(2 * 20, 2 * len(to_plot) * 0.15))

    ax2 = fig.add_subplot(gs[-1, 0:2])
    y_label = 'dF/F'
    cell_avg_stim_traces = expobj.SLMTargets_stims_dffAvg
    aoplot.plot_periphotostim_avg(arr=cell_avg_stim_traces, expobj=expobj,
                                  figsize=[5, 4], y_lims=y_lims_periphotostim_trace, fig=fig, ax=ax2, show=False,
                                  title=('responses of all photostim targets'),
                                  y_label=y_label, x_label='Time post-stimulation (seconds)')

    # fig.show()

    if hasattr(expobj, 'stims_in_sz'):

        # make response magnitude and response success rate figure
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=((5 * 4), 5), nrows=1, ncols=4)
        # stims out sz
        ax3 = fig.add_subplot(gs[-1, 2:4])
        data = [[np.mean(expobj.responses_SLMtargets_outsz.loc[i]) for i in range(expobj.n_targets_total)]]
        fig, ax3 = pj.plot_hist_density(data, x_label='response magnitude (dF/F)', title='stims_out_sz - ',
                                     fig=fig, ax=ax3, show=False)
        ax4 = fig.add_subplot(gs[-1, 4])
        fig, ax4 = pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets_outsz.values())],
                                           x_tick_labels=[trial],
                                           ylims=[0, 100], bar=False, y_label='% success stims.',
                                           title='target success rate (stims out sz)', expand_size_x=2,
                                           show=False, fig=fig, ax=ax4)
        # stims in sz
        ax5 = fig.add_subplot(gs[-1, 5:7])
        data = [[np.mean(expobj.responses_SLMtargets_insz.loc[i]) for i in range(expobj.n_targets_total)]]
        fig, ax5 = pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ',
                                        fig=fig, ax=ax5, show=False)
        ax6 = fig.add_subplot(gs[-1, 7])
        fig, ax6 = pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets_insz.values())],
                                        x_tick_labels=[trial],
                                        ylims=[0, 100], bar=False, y_label='% success stims.',
                                        title='target success rate (stims in sz)', expand_size_x=2,
                                        show=False, fig=fig, ax=ax6)
        fig.tight_layout()
        if save_results:
            save = expobj.analysis_save_path[:-17] + '/results/' + '%s_%s_slm_targets_responses_dFF' % (expobj.metainfo['animal prep.'], trial)
            os.makedirs(expobj.analysis_save_path[:-17] + '/results') if not os.path.exists(expobj.analysis_save_path[:-17] + '/results') else None
            print('saving png and svg to: %s' % save)
            fig.savefig(fname=save + '.png', transparent=True, format='png')
            fig.savefig(fname=save + '.svg', transparent=True, format='svg')

        fig.show()

    else:
        # no sz
        # fig, (ax1, ax2) = plt.subplots(figsize=((5 * 2), 5), nrows=1, ncols=2)
        data = [[np.mean(expobj.responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        ax3 = fig.add_subplot(gs[-1, 2:4])
        fig, ax3 = pj.plot_hist_density(data, x_label='response magnitude (dF/F)', title='no sz', show=False, fig=fig, ax=ax3)
        ax4 = fig.add_subplot(gs[-1, 4])
        fig, ax4 = pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                                           ylims=[0, 100], bar=False, show=False, fig=fig, ax=ax4,
                                           y_label='% success stims.', title='target success rate (pre4ap)',
                                           expand_size_x=2)

        zero_point = abs(v_lims_periphotostim_heatmap[0]/v_lims_periphotostim_heatmap[1])
        c = ColorConverter().to_rgb
        if cmap is None:
            cmap = pj.make_colormap([c('blue'), c('white'), zero_point - 0.20, c('white'), c('red')])
        ax5 = fig.add_subplot(gs[-1, 5:])
        fig, ax5 = aoplot.plot_traces_heatmap(data=cell_avg_stim_traces, expobj=expobj, vmin=v_lims_periphotostim_heatmap[0], vmax=v_lims_periphotostim_heatmap[1],
                                              stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames + 1, cbar=False,
                                              title=(expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo[
                                              'trial'] + ' - SLM targets raw Flu'), show=False, fig=fig, ax=ax5, x_label='Frames', y_label='Neurons',
                                              xlims=(0, expobj.pre_stim + expobj.stim_duration_frames + expobj.post_stim),
                                              cmap=cmap)

        fig.tight_layout()
        if save_results:
            save = expobj.analysis_save_path[:-17] + '/results/' + '%s_%s_slm_targets_responses_dFF' % (expobj.metainfo['animal prep.'], trial)
            os.makedirs(expobj.analysis_save_path[:-17] + '/results') if not os.path.exists(expobj.analysis_save_path[:-17] + '/results') else None
            print('saving png and svg to: %s' % save)
            fig.savefig(fname=save+'.png', transparent=True, format='png')
            fig.savefig(fname=save+'.svg', transparent=True,  format='svg')

        fig.show()

###### NON TARGETS analysis + plottings
def run_allopticalAnalysisNontargets(expobj, normalize_to='pre_stim', to_plot=True, save_plot_suffix='', skip_processing=False):
    if not skip_processing:

        good_cells, events_loc_cells, flu_events_cells, stds = expobj._good_cells(cell_ids=expobj.cell_id, raws=expobj.raw,
                                                                                  photostim_frames=expobj.photostim_frames, std_thresh=2.5, save=False)
        # set the correct test response windows
        if not expobj.pre_stim == int(1.0 * expobj.fps):
            print('updating expobj.pre_stim_sec to 1 sec')
            expobj.pre_stim = int(1.0 * expobj.fps)  # length of pre stim trace collected (in frames)
            expobj.post_stim = int(3.0 * expobj.fps)  # length of post stim trace collected (in frames)
            expobj.post_stim_response_window_msec = 500  # msec
            expobj.post_stim_response_frames_window = int(expobj.fps * expobj.post_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)
            expobj.pre_stim_response_window_msec = 500  # msec
            expobj.pre_stim_response_frames_window = int(expobj.fps * expobj.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)

        expobj._trialProcessing_nontargets(normalize_to, save=False)
        expobj._sigTestAvgResponse_nontargets(alpha=0.1, save=False)

        expobj.save()

    # make figure containing plots showing average responses of nontargets to photostim
    save_plot_path = expobj.analysis_save_path[:30] + 'Results_figs/' + save_plot_suffix
    fig_non_targets_responses(expobj=expobj, plot_subset=False, save_fig=save_plot_path) if to_plot else None

    print('\n** FIN. * allopticalAnalysisNontargets * %s %s **** ' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    print('-------------------------------------------------------------------------------------------------------------\n\n')


def fig_non_targets_responses(expobj, plot_subset: bool = True, save_fig=None):
    print('\n----------------------------------------------------------------')
    print('plotting nontargets responses ')
    print('----------------------------------------------------------------')

    if plot_subset:
        selection = np.random.randint(0, expobj.dff_traces_avg.shape[0], 100)
    else:
        selection = np.arange(expobj.dff_traces_avg.shape[0])

    #### SUITE2P NON-TARGETS - PLOTTING OF AVG PERI-PHOTOSTIM RESPONSES
    if sum(expobj.sig_units) > 0:
        f = plt.figure(figsize=[25, 10])
        gs = f.add_gridspec(2, 9)
    else:
        f = plt.figure(figsize=[25, 5])
        gs = f.add_gridspec(1, 9)

    # PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
    a1 = f.add_subplot(gs[0, 0:2])
    x = expobj.dff_traces_avg[selection]
    y_label = 'pct. dFF (normalized to prestim period)'
    aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                                  title='Average photostim all trials response', y_label=y_label, fig=f, ax=a1, show=False,
                                  x_label='Time (seconds)', y_lims=[-50, 200])
    # PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
    a2 = f.add_subplot(gs[0, 2:4])
    x = expobj.dfstdF_traces_avg[selection]
    y_label = 'dFstdF (normalized to prestim period)'
    aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                                  title='Average photostim all trials response', y_label=y_label, fig=f, ax=a2, show=False,
                                  x_label='Time (seconds)', y_lims=[-1, 3])
    # PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/f
    a3 = f.add_subplot(gs[0, 4:6])
    vmin = -1
    vmax = 1
    aoplot.plot_traces_heatmap(expobj.dfstdF_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax,
                               stim_on=int(1 * expobj.fps),
                               stim_off=int(1 * expobj.fps + expobj.stim_duration_frames),
                               xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                               title='dF/stdF heatmap for all nontargets', x_label='Time', cbar=True,
                               fig=f, ax=a3, show=False)
    # PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/stdf
    a4 = f.add_subplot(gs[0, -3:-1])
    vmin = -100
    vmax = 100
    aoplot.plot_traces_heatmap(expobj.dff_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=int(1 * expobj.fps),
                               stim_off=int(1 * expobj.fps + expobj.stim_duration_frames),
                               xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                               title='dF/F heatmap for all nontargets', x_label='Time', cbar=True,
                               fig=f, ax=a4, show=False)
    # bar plot of avg post stim response quantified between responders and non-responders
    a04 = f.add_subplot(gs[0, -1])
    sig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)
    nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
    data = np.asarray([sig_responders_avgresponse, nonsig_responders_avgresponse])
    pj.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'gray'],
                            y_label='avg dF/stdF', bar=False,
                            text_list=['%s pct' % (np.round(
                                (len(sig_responders_avgresponse) / expobj.post_array_responses.shape[0]), 2) * 100),
                                       '%s pct' % (np.round(
                                           (len(nonsig_responders_avgresponse) / expobj.post_array_responses.shape[0]),
                                           2) * 100)],
                            text_y_pos=1.43, text_shift=1.7, x_tick_labels=['significant', 'non-significant'], ylims=[-2, 3],
                            expand_size_y=1.5, expand_size_x=0.6,
                            fig=f, ax=a04, show=False)

    ## PLOTTING STATISTICALLY SIGNIFICANT RESPONDERS
    if sum(expobj.sig_units) > 0:
        # plot PERI-STIM AVG TRACES of sig nontargets
        a10 = f.add_subplot(gs[1, 0:2])
        x = expobj.dfstdF_traces_avg[expobj.sig_units]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a10, show=False,
                                      title='significant responders', y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of nonsig nontargets
        a11 = f.add_subplot(gs[1, 2:4])
        x = expobj.dfstdF_traces_avg[~expobj.sig_units]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a11, show=False,
                                      title='non-significant responders', y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of sig. positive responders
        a12 = f.add_subplot(gs[1, 4:6])
        x = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a12, show=False,
                                      title='positive signif. responders', y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of sig. negative responders
        a13 = f.add_subplot(gs[1, -3:-1])
        x = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a13, show=False,
                                      title='negative signif. responders', y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # bar plot of avg post stim response quantified between responders and non-responders
        a14 = f.add_subplot(gs[1, -1])
        possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
        data = np.asarray([possig_responders_avgresponse, negsig_responders_avgresponse, nonsig_responders_avgresponse])
        pj.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'blue', 'gray'],
                                y_label='avg dF/stdF', bar=False,
                                text_list=['%s pct' % (np.round(
                                    (len(possig_responders_avgresponse) / expobj.post_array_responses.shape[0]) * 100, 1)),
                                           '%s pct' % (np.round((len(negsig_responders_avgresponse) /
                                                                 expobj.post_array_responses.shape[0]) * 100, 1)),
                                           '%s pct' % (np.round((len(nonsig_responders_avgresponse) /
                                                                 expobj.post_array_responses.shape[0]) * 100, 1))],
                                text_y_pos=1.43, text_shift=1.2, ylims=[-2, 3],
                                x_tick_labels=['pos. significant', 'neg. significant', 'non-significant'],
                                expand_size_y=1.5, expand_size_x=0.5,
                                fig=f, ax=a14, show=False)

    f.suptitle(
        ('%s %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'])))
    f.tight_layout()
    f.show()

    if save_fig is not None:
        _path = save_fig[:[i for i in re.finditer('/', save_fig)][-1].end()]
        os.makedirs(_path) if not os.path.exists(_path) else None
        print('saving figure output to:', save_fig)
        plt.savefig(save_fig)


# # # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = import_resultsobj(pkl_path=results_object_path)

# %% ARCHIVE

# calculate the dFF responses of the non-targeted cells, create a pandas df of the post-stim dFF responses of all cells - THESE ARE OUTDATED FUNCS. - NEW CODE FOR COLLECTING AND QUANTIFYING RESPONSES OF NONTARGET RESPONSES .21/10/16
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
                            stim - expobj.pre_stim:stim + expobj.stim_duration_frames + expobj.post_stim]
                    trace_dff = ((trace - mean_base) / abs(mean_base)) * 100
                    response = np.mean(trace_dff[
                                       expobj.pre_stim + expobj.stim_duration_frames:expobj.pre_stim + 3 * expobj.stim_duration_frames])
                    df.at[cell, '%s' % stim] = round(response, 3)
                    df.at[cell, 'group'] = group
                if mean_base < 50:
                    risky_cells.append(cell)

            elif normalize_to == 'pre-stim':
                mean_pre_list = []
                for stim in expobj.stim_start_frames:
                    cell_idx = expobj.cell_id.index(cell)
                    trace = expobj.raw[cell_idx][
                            stim - expobj.pre_stim:stim + expobj.stim_duration_frames + expobj.post_stim]
                    mean_pre = np.mean(trace[0:expobj.pre_stim]);
                    mean_pre_list.append(mean_pre)
                    trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
                    response = np.mean(trace_dff[
                                       expobj.pre_stim + expobj.stim_duration_frames:expobj.pre_stim + 3 * expobj.stim_duration_frames])
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
    #                         stim - expobj.pre_stim_sec:stim + expobj.stim_duration_frames + expobj.post_stim_sec]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim_sec])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre))  * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #                 # response = np.mean(trace_dff[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 3*expobj.stim_duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 1 + 2 * expobj.stim_duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim_sec + expobj.stim_duration_frames:expobj.pre_stim_sec + 1 + 2 * expobj.stim_duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #     elif 'photostim target' in group:
    #         cells = expobj.s2p_cell_targets
    #         for stim in expobj.stim_start_frames:
    #             for cell in cells:
    #                 cell_idx = expobj.cell_id.index(cell)
    #                 trace = expobj.raw[cell_idx][
    #                         stim - expobj.pre_stim_sec:stim + expobj.stim_duration_frames + expobj.post_stim_sec]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim_sec])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #                 # response = np.mean(trace_dff[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 3*expobj.stim_duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 1 + 2 * expobj.stim_duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim_sec + expobj.stim_duration_frames:expobj.pre_stim_sec + 3 * expobj.stim_duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #                 df.at[cell, 'group'] = group

    print('Completed gathering dFF responses to photostim for %s cells' % len(
        np.unique([expobj.good_cells + expobj.s2p_cell_targets])))
    print('risky cells (with very low Flu values to normalize with) and very high dFF values: (%s)' % len(risky_cells), risky_cells)

    return df, risky_cells
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
                        stim - expobj.pre_stim:stim + expobj.stim_duration_frames + expobj.post_stim]
                mean_pre = np.mean(trace[0:expobj.pre_stim])
                std_pre = np.std(trace[0:expobj.pre_stim])
                dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                stim_traces_dF_stdF.append(dF_stdF)
                response = np.mean(
                    dF_stdF[
                    expobj.pre_stim + expobj.stim_duration_frames:expobj.pre_stim + 1 + 2 * expobj.stim_duration_frames])

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
    #                         stim - expobj.pre_stim_sec:stim + expobj.stim_duration_frames + expobj.post_stim_sec]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim_sec])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre))  * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #                 # response = np.mean(trace_dff[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 3*expobj.stim_duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 1 + 2 * expobj.stim_duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim_sec + expobj.stim_duration_frames:expobj.pre_stim_sec + 1 + 2 * expobj.stim_duration_frames])
    #                 df.at[cell, '%s' % stim] = round(response, 4)
    #     elif 'photostim target' in group:
    #         cells = expobj.s2p_cell_targets
    #         for stim in expobj.stim_start_frames:
    #             for cell in cells:
    #                 cell_idx = expobj.cell_id.index(cell)
    #                 trace = expobj.raw[cell_idx][
    #                         stim - expobj.pre_stim_sec:stim + expobj.stim_duration_frames + expobj.post_stim_sec]
    #                 mean_pre = np.mean(trace[0:expobj.pre_stim_sec])
    #                 trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
    #                 std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #                 # response = np.mean(trace_dff[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 3*expobj.stim_duration_frames])
    #                 dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
    #                 # response = np.mean(dF_stdF[pre_stim_sec + expobj.stim_duration_frames:pre_stim_sec + 1 + 2 * expobj.stim_duration_frames])
    #                 response = np.mean(trace_dff[
    #                                    expobj.pre_stim_sec + expobj.stim_duration_frames:expobj.pre_stim_sec + 3 * expobj.stim_duration_frames])
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

## moved to utils.funcs_pj .21/10/16
def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y
