import re

import os
import sys

sys.path.append('/home/pshah/Documents/code/')
from Vape.utils.utils_funcs import s2p_loader
from suite2p.run_s2p import run_s2p
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import xml.etree.ElementTree as ET
import tifffile as tf
from funcsforprajay import funcs as pj
from _utils_.paq_utils import paq_read
import pickle



class TwoPhotonImaging:

    def __init__(self, tiff_path, paq_path, metainfo, analysis_save_path, suite2p_path=None,
                 suite2p_run=False, save_downsampled_tiff: bool = False, quick=False):
        """
        :param suite2p_path: path to the suite2p outputs (plane0 file)
        :param suite2p_run: set to true if suite2p is already run for this trial
        """

        print('\n***** CREATING NEW TwoPhotonImaging with the following metainfo: ', metainfo)

        from _analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis import Suite2pROIsSz
        self.Suite2pROIsSz: Suite2pROIsSz = None  # analysis of suite2p processed data


        assert os.path.exists(tiff_path)
        assert os.path.exists(analysis_save_path)
        assert os.path.exists(paq_path)
        if suite2p_path: assert os.path.exists(suite2p_path)

        self.tiff_path = tiff_path
        self.paq_path = paq_path
        self.metainfo = {'animal prep.': None, 'trial': None, 'date': None, 'exptype': None, 'data_path_base': None, 'comments': None,
                         'pre4ap_trials': None, 'post4ap_trials': None}  # dict containing appropriate metainformation fields for the experimental trial
        self.metainfo = metainfo
        self.analysis_save_path = analysis_save_path
        if self.analysis_save_path[-1] != '/': self.analysis_save_path = self.analysis_save_path + '/'

        # create analysis save path location
        if not os.path.exists(analysis_save_path):
            print('\t\-making new analysis save folder at: \n  %s' % self.analysis_save_path)
            os.makedirs(self.analysis_save_path)

        # save expobj to pkl object
        self.save(pkl_path=self.pkl_path)

        if not quick and save_downsampled_tiff:
            stack = self.mean_raw_flu_trace(save_pkl=True)
            if save_downsampled_tiff:
                pj.SaveDownsampledTiff(stack=stack, save_as=analysis_save_path + '/%s_%s_downsampled.tif' % (
                    metainfo['date'], metainfo['trial']))  # specify path in Analysis folder to save pkl object')

        if suite2p_run:
            self.suite2p_path = suite2p_path
            self.s2pProcessing(s2p_path=self.suite2p_path)

        self.dFF = None  #: dFF normalized traces of Suite2p ROIs
        self.save()


    def __repr__(self):
        lastmod = time.ctime(os.path.getmtime(self.pkl_path))
        if not hasattr(self, 'metainfo'):
            information = f"uninitialized"
        else:
            prep = self.metainfo['animal prep.']
            trial = self.metainfo['trial']
            information = f"{prep} {trial}"
        return repr(f"({information}) TwoPhotonImaging experimental data object, last saved: {lastmod}")

    @property
    def date(self):
        return self.metainfo['date']

    @property
    def prep(self):
        return self.metainfo['animal prep.']

    @property
    def trial(self):
        return self.metainfo['trial']

    @property
    def exptype(self):
        return self.metainfo['exptype']


    @property
    def t_series_name(self):
        return f'{self.metainfo["animal prep."]} {self.metainfo["trial"]}'

    @property
    def tiff_path_dir(self):
        return self.tiff_path[:[(s.start(), s.end()) for s in re.finditer('/', self.tiff_path)][-1][0]]  # this is the directory where the Bruker xml files associated with the 2p imaging TIFF are located

    @property
    def pkl_path(self):
        """specify path in Analysis folder to save pkl object"""
        self._pkl_path = f"{self.analysis_save_path}{self.metainfo['date']}_{self.metainfo['trial']}.pkl"
        return self._pkl_path

    @pkl_path.setter
    def pkl_path(self, path: str):
        self._pkl_path = path

    @property
    def backup_pkl(self):
        return self.analysis_save_path + f"backups/{self.metainfo['date']}_{self.metainfo['animal prep.']}_{self.metainfo['trial']}.pkl"

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

    def getXMLFrame(self, path):
        relativeTime = []
        fr_index = []

        xml_tree = ET.parse(path)  # parse xml from a path
        root = xml_tree.getroot()  # make xml tree structure

        pv_sequence = root.find('Sequence')  # find pv state shard element in root

        for elem in pv_sequence:  # for each element in pv state shard, find the value for the specified key
            if elem.get('relativeTime') != None:
                relativeTime.append(float(elem.get('relativeTime')))
                fr_index.append(int(elem.get('index')))

        if not relativeTime:  # if no value found at all, raise exception
            raise Exception('ERROR: relative time of frames could not be found.')

        assert fr_index[-1] == self.n_frames
        self.pv_fr_time = relativeTime

    def frame_time(self, frame: int):
        return self.pv_fr_time[frame]

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

        print(
            '\n-----parsing PV Metadata')

        xml_tree = ET.parse(self.xml_path)  # parse xml from a path
        root = xml_tree.getroot()  # make xml tree structure

        sequence = root.find('Sequence')
        acq_type = sequence.get('type')

        if 'ZSeries' in acq_type:
            n_planes = len(sequence.findall('Frame'))
        else:
            n_planes = 1

        frame_period = float(self._getPVStateShard(self.xml_path, 'framePeriod')[0])
        fps = 1 / frame_period

        frame_x = int(self._getPVStateShard(self.xml_path, 'pixelsPerLine')[0])

        frame_y = int(self._getPVStateShard(self.xml_path, 'linesPerFrame')[0])

        zoom = float(self._getPVStateShard(self.xml_path, 'opticalZoom')[0])

        scanVolts, _, index = self._getPVStateShard(self.xml_path, 'currentScanCenter')
        for scanVolts, index in zip(scanVolts, index):
            if index == 'XAxis':
                scan_x = float(scanVolts)
            if index == 'YAxis':
                scan_y = float(scanVolts)

        pixelSize, _, index = self._getPVStateShard(self.xml_path, 'micronsPerPixel')
        for pixelSize, index in zip(pixelSize, index):
            if index == 'XAxis':
                pix_sz_x = float(pixelSize)
            if index == 'YAxis':
                pix_sz_y = float(pixelSize)

        env_tree = ET.parse(self.env_path)
        env_root = env_tree.getroot()

        elem_list = env_root.find('TSeries')

        n_frames = elem_list[0].get('repetitions') # Rob would get the n_frames from env file
        # change this to getting the last actual index from the xml file

        n_frames = root.findall('Sequence/Frame')[-1].get('index')

        self.__fps = fps
        self.__frame_x = frame_x
        self.__frame_y = frame_y
        self.__n_planes = n_planes
        self.__pix_sz_x = pix_sz_x
        self.__pix_sz_y = pix_sz_y
        self.__scan_x = scan_x
        self.__scan_y = scan_y
        self.__zoom = zoom
        self.__n_frames = int(n_frames)

        print('n planes:', n_planes,
              '\nn frames:', int(n_frames),
              '\nfps:', fps,
              '\nframe size (px):', frame_x, 'x', frame_y,
              '\nzoom:', zoom,
              '\npixel size (um):', pix_sz_x, pix_sz_y,
              '\nscan centre (V):', scan_x, scan_y
              )


    @property
    def xml_path(self):
        tiff_path = self.tiff_path_dir
        path = []

        try:  # look for xml file in path, or two paths up (achieved by decreasing count in while loop)
            count = 2
            while count != 0 and not path:
                count -= 1
                for file in os.listdir(tiff_path):
                    if file.endswith('.xml'):
                        path = os.path.join(tiff_path, file)
                tiff_path = os.path.dirname(tiff_path)
            return path
        except:
            raise Exception('ERROR: Could not find or load xml file for this acquisition from %s' % tiff_path)

    @property
    def env_path(self):
        tiff_path = self.tiff_path_dir
        path = []

        try:  # look for xml file in path, or two paths up (achieved by decreasing count in while loop)
            count = 2
            while count != 0 and not path:
                count -= 1
                for file in os.listdir(tiff_path):
                    if file.endswith('.env'):
                        env_path = os.path.join(tiff_path, file)
                tiff_path = os.path.dirname(tiff_path)
            return env_path
        except:
            raise Exception('ERROR: Could not find or load env file for this acquisition from %s' % tiff_path)

    @property
    def n_planes(self):
        # xml_tree = ET.parse(self.xml_path)  # parse xml from a path
        # root = xml_tree.getroot()  # make xml tree structure
        #
        # sequence = root.find('Sequence')
        # acq_type = sequence.get('type')
        #
        # if 'ZSeries' in acq_type:
        #     return len(sequence.findall('Frame'))
        # else:
        #     return 1
        return self.__n_planes

    @property
    def n_frames(self):
        # xml_tree = ET.parse(self.xml_path)  # parse xml from a path
        # root = xml_tree.getroot()  # make xml tree structure
        #
        # return int(root.findall('Sequence/Frame')[-1].get('index'))
        return self.__n_frames

    @property
    def frame_avg(self):
        frame_avg = int(self._getPVStateShard(self.xml_path, 'rastersPerFrame')[0])
        # print('Frame averaging:', frame_avg)
        return frame_avg

    @property
    def fps(self):
        # frame_period = float(self._getPVStateShard(self.xml_path, 'framePeriod')[0])
        # return (1 / frame_period)
        return int(self.__fps)

    @property
    def laser_power(self):
        laser_powers, lasers, _ = self._getPVStateShard(self.xml_path,'laserPower')
        for power,laser in zip(laser_powers,lasers):
            if laser == 'Imaging':
                imaging_power = float(power)
                return imaging_power
        # print('Imaging laser power:', imaging_power)

    @property
    def zoom(self):
        # return float(self._getPVStateShard(self.xml_path, 'opticalZoom')[0])
        return self.__zoom

    @property
    def frame_x(self):
        # return int(self._getPVStateShard(self.xml_path, 'pixelsPerLine')[0])
        return self.__frame_x

    @property
    def frame_y(self):
        # return int(self._getPVStateShard(self.xml_path, 'linesPerFrame')[0])
        return self.__frame_y


    @property
    def pix_sz_x(self):
        # pixelSize, _, index = self._getPVStateShard(self.xml_path, 'micronsPerPixel')
        # for pixelSize, index in zip(pixelSize, index):
        #     if index == 'XAxis':
        #         return float(pixelSize)
        return self.__pix_sz_x

    @property
    def pix_sz_y(self):
        # pixelSize, _, index = self._getPVStateShard(self.xml_path, 'micronsPerPixel')
        # for pixelSize, index in zip(pixelSize, index):
        #     if index == 'YAxis':
        #         return float(pixelSize)
        return self.__pix_sz_y

    @property
    def scan_x(self):
        # scanVolts, _, index = self._getPVStateShard(self.xml_path, 'currentScanCenter')
        # for scanVolts, index in zip(scanVolts, index):
        #     if index == 'XAxis':
        #         return float(scanVolts)
        return self.__scan_x

    @property
    def scan_y(self):
        # scanVolts, _, index = self._getPVStateShard(self.xml_path, 'currentScanCenter')
        # for scanVolts, index in zip(scanVolts, index):
        #     if index == 'YAxis':
        #         return float(scanVolts)
        return self.__scan_y

    @property
    def fov_size(self):
        return self.pix_sz_x * self.frame_x, self.pix_sz_y * self.frame_y

    @property
    def reg_tif_crop_path(self):
        """path to the TIFF file derived from underlying suite2p batch registered TIFFs and cropped to frames for current
        t-series of experiment analysis object"""
        return self.analysis_save_path + 'reg_tiff_%s_r.tif' % self.metainfo['trial']

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
                FminusFneu, spks, self.stat, neuropil = s2p_loader(s2p_path, subtract_neuropil)  # s2p_loader() is in Vape/utils_func
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

            print('|- Loaded %s cells, recorded for %s secs' % (
            self.raw.shape[0], round(self.raw.shape[1] / self.fps, 2)))

            self.save() if save else None

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

    def _good_cells(self, cell_ids: list, raws: np.ndarray, photostim_frames: list, std_thresh: int,
                    radiuses: list = None, min_radius_pix: int = 2.5, max_radius_pix: int = 10, save=True):
        """
        This function is used for filtering for "good" cells based on detection of Flu deflections that are above some std threshold based on std_thresh.
        Note: a moving averaging window of 4 frames is used to find Flu deflections above std threshold.

        :param cell_ids: ls of cell ids to iterate over
        :param raws: raw flu values corresponding to the cell ls in cell_ids
        :param photostim_frames: frames to delete from the raw flu traces across all cells (because they are photostim frames)
        :param std_thresh: std. factor above which to define reliable Flu events
        :param radiuses: radiuses of the s2p ROI mask of all cells in the same order as cell_ids
        :param min_radius_pix:
        :param max_radius_pix:
        :return:
            good_cells: ls of cell_ids that had atleast 1 event above the std_thresh
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
                # raw_dff = normalize_dff_jit(raw_)  # note that this function is defined in this file a little further down
                from _alloptical_utils import normalize_dff
                raw_dff = normalize_dff(raw_)  # note that this function is defined in this file a little further down
                std_ = raw_dff.std()

                from _alloptical_utils import moving_average
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

        if lfp:
            # find voltage (LFP recording signal) channel and save as lfp_signal attribute
            voltage_idx = paq['chan_names'].index('voltage')
            self.lfp_signal = paq['data'][voltage_idx]

    def convert_frames_to_paqclock(self, frame_num):
        """convert a frame number into the corresponding clock timestamp from the paq file"""
        try:
            return self.frame_clock_actual[frame_num]
        except AttributeError:
            raise AttributeError('need to run .paqProcessing to perform conversion of frames to paq clock.')


    def mean_raw_flu_trace(self, save_pkl: bool = True):
        print('\n-----collecting mean raw FOV flu trace from tiff file...')
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
            if hasattr(self, 'save_path'):
                self.save(pkl_path=self.pkl_path)
            else:
                print('pkl file not saved yet because .save_path attr not found')

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
            plt.suptitle(f"{self.metainfo['animal prep.']} {self.metainfo['trial']} frame num: {frame_num}")
        plt.show()
        return stack

    def stitch_reg_tiffs(self, force_crop: bool = False, do_stack: bool = False):

        start = self.curr_trial_frames[0] // 2000  # 2000 because that is the batch size for suite2p run
        end = self.curr_trial_frames[1] // 2000 + 1

        tif_path_save = self.analysis_save_path + 'reg_tiff_%s.tif' % self.metainfo['trial']
        tif_path_save2 = self.reg_tif_crop_path

        if force_crop:
            do_stack = True if not os.path.exists(tif_path_save) else do_stack

        reg_tif_folder = self.s2p_path + '/reg_tif/'
        reg_tif_list = os.listdir(reg_tif_folder)
        reg_tif_list.sort()
        sorted_paths = [reg_tif_folder + tif for tif in reg_tif_list][start:end + 1]

        if do_stack or not os.path.exists(tif_path_save):
            print(f"stacked registered tif path save: {tif_path_save}")
            print(f"sorted paths to indiv. tiff: \n\t{sorted_paths}")
            tif_arr = pj.make_tiff_stack(sorted_paths, save_as=tif_path_save)
            print(f"shape of stacked tiff (stitched registered TIFF): {tif_arr.shape}")

        if not os.path.exists(tif_path_save2) or force_crop:
            with tf.TiffWriter(tif_path_save2, bigtiff=True) as tif:
                with tf.TiffFile(tif_path_save, multifile=False) as input_tif:
                    data = input_tif.asarray()
                    print(f'... cropping registered tiff of shape: {data.shape}')
                reg_tif_crop = data[self.curr_trial_frames[0] - start * 2000: self.curr_trial_frames[1] - (start * 2000)]
                print('|- saving cropped reg tiff of shape: ', reg_tif_crop.shape)
                assert reg_tif_crop.shape[0] == self.n_frames, "n_frames of experiment object does not match the number of frames in cropped registered TIFF"
                tif.save(reg_tif_crop)

    def s2pMeanImage(self, s2p_path):
        os.chdir(s2p_path)

        ops = np.load('ops.npy', allow_pickle=True).item()

        mean_img = ops['meanImg']

        mean_img = np.array(mean_img, dtype='uint16')

        return mean_img

    def dfof(self):
        from _alloptical_utils import normalize_dff
        dFF = normalize_dff(self.raw)
        return dFF

    def save(self, pkl_path: str = None):
        from _utils_.io import save_pkl
        if pkl_path is None: pkl_path = self.pkl_path
        save_pkl(obj=self, save_path=pkl_path)



