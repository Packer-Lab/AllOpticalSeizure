import functools
import os
import pickle
import sys

import pandas as pd

from _exp_metainfo_.data_paths import onePresults_object_path
from _utils_.io import import_1pexobj
from _exp_metainfo_.exp_metainfo import import_resultsobj
# from archive.alloptical_utils_pj import import_resultsobj

sys.path.append('/home/pshah/Documents/code/')
import time
import matplotlib.pyplot as plt
import numpy as np
from funcsforprajay import funcs as pj

from _utils_.paq_utils import paq_read, frames_discard
from _utils_ import alloptical_plotting as aoplot, _alloptical_utils as Utils

from _main_.TwoPhotonImagingMain import TwoPhotonImaging

# %% main code

class OnePhotonStim(TwoPhotonImaging):

    compatible_responses_process = ['pre-stim dFF', 'post - pre']

    oneP_pre4ap_exp_list = [
        'PS17 t-008',
        'PS07 t-003',
        'PS07 t-010',
        'PS11 t-004',
        'PS11 t-009',
        'PS18 t-007',
        'PS09 t-008',
        'PS16 t-005',
        'PS16 t-006',
        'PS16 t-008'
    ]

    oneP_post4ap_exp_list = [
        'PS07 t-012',
        'PS07 t-015',
        'PS11 t-012',
        'PS11 t-017',
        # 'PS11 t-019',  #: not a proper 1p stim exp trial...(protocol not run fully..)
        'PS18 t-009',
        'PS09 t-011',
        'PS09 t-013',
        'PS09 t-015',
        'PS16 t-009',
        'PS16 t-010'
    ]

    # wrapper for use in AnalysisFuncs
    @staticmethod
    def runOverExperiments(run_pre4ap_trials=True, run_post4ap_trials=True, skip_trials=[], run_trials=[],
                           ignore_cache=False, supress_print=False, **kwargs):
        """decorator to use for for-looping through experiment trials across run_pre4ap_trials and run_post4ap_trials.
        the trials to for loop through are defined in allopticalResults.pre_4ap_trials and allopticalResults.post_4ap_trials"""
        # if len(run_trials) > 0 or run_pre4ap_trials is True or run_post4ap_trials is True:
        t_start = time.time()

        def main_for_loop(func):
            @functools.wraps(func)
            def inner():
                if not supress_print: print(f"\n {'..' * 5} INITIATING FOR LOOP ACROSS EXPS {'..' * 5}\n")
                if run_trials:
                    if not supress_print: print(f"\n{'-' * 5} RUNNING SPECIFIED TRIALS from `trials_run` {'-' * 5}")
                    counter1 = 0
                    res = []
                    for i, exp_prep in enumerate(run_trials):
                        # print(i, exp_prep)
                        try:  # dont continue if exp_prep already run before (as determined by location in func_cache
                            if Utils.get_from_cache(func.__name__, item=exp_prep) and ignore_cache is False:
                                run = False
                                if not supress_print: print(f"{exp_prep} found in cache for func {func.__name__} ... skipping repeat run.")
                            else:
                                run = True
                        except KeyError:
                            run = True
                        if run is True:
                            prep = exp_prep[:-6]
                            trial = exp_prep[-5:]
                            expobj, _ = import_1pexobj(prep=prep, trial=trial, verbose=False)

                            if not supress_print: Utils.working_on(expobj)
                            res_ = func(expobj=expobj, **kwargs)
                            if not supress_print: Utils.end_working_on(expobj)
                            if res_ is not None and type(res_) is not bool: res.append(res_)
                            Utils.set_to_cache(func_name=func.__name__, item=exp_prep) if res_ is True and not ignore_cache else None

                    counter1 += 1

                if run_pre4ap_trials:
                    if not supress_print: print(f"\n{'-' * 5} RUNNING PRE4AP TRIALS {'-' * 5}")
                    counter_i = 0
                    res = []
                    for i, x in enumerate(OnePhotonStim.oneP_pre4ap_exp_list):
                        counter_j = 0
                        for j, exp_prep in enumerate([x]):
                            if exp_prep in skip_trials:
                                pass
                            else:
                                # print(i, exp_prep)
                                try:  # dont continue if exp_prep already run before (as determined by location in func_cache
                                    if Utils.get_from_cache(func.__name__, item=exp_prep) and ignore_cache is False:
                                        run = False
                                        if not supress_print: print(
                                            f"{exp_prep} found in cache for func {func.__name__} ... skipping repeat run.")
                                    else:
                                        run = True
                                except KeyError:
                                    run = True
                                if run is True:
                                    prep = exp_prep[:-6]
                                    trial = exp_prep[-5:]
                                    # expobj, _ = OnePhotonStim.import_1pexobj(prep=prep, trial=trial, verbose=False)
                                    expobj, _ = import_1pexobj(prep=prep, trial=trial, verbose=False)

                                    if not supress_print: Utils.working_on(expobj)
                                    res_ = func(expobj=expobj, **kwargs)
                                    if not supress_print: Utils.end_working_on(expobj)
                                    if res_ is not None and type(res_) is not bool: res.append(res_)
                                    Utils.set_to_cache(func_name=func.__name__, item=exp_prep) if res_ is True and not ignore_cache else None
                            counter_j += 1
                        counter_i += 1
                    if res:
                        return res

                if run_post4ap_trials:
                    if not supress_print: print(f"\n{'-' * 5} RUNNING POST4AP TRIALS {'-' * 5}")
                    counter_i = 0
                    res = []
                    for i, x in enumerate(OnePhotonStim.oneP_post4ap_exp_list):
                        counter_j = 0
                        for j, exp_prep in enumerate([x]):
                            if exp_prep in skip_trials:
                                pass
                            else:
                                # print(i, exp_prep)
                                try:  # dont continue if exp_prep already run before (as determined by location in func_cache
                                    if Utils.get_from_cache(func.__name__, item=exp_prep) and ignore_cache is False:
                                        run = False
                                        if not supress_print: print(
                                            f"{exp_prep} found in cache for func {func.__name__} ... skipping repeat run.")
                                    else:
                                        run = True
                                except KeyError:
                                    run = True
                                if run is True:
                                    prep = exp_prep[:-6]
                                    trial = exp_prep[-5:]
                                    # expobj, _ = OnePhotonStim.import_1pexobj(prep=prep, trial=trial, verbose=False)
                                    expobj, _ = import_1pexobj(prep=prep, trial=trial, verbose=False)

                                    if not supress_print: Utils.working_on(expobj)
                                    res_ = func(expobj=expobj, **kwargs)
                                    if not supress_print: Utils.end_working_on(expobj)
                                    if res_ is not None and type(res_) is not bool: res.append(res_)
                                    Utils.set_to_cache(func_name=func.__name__, item=exp_prep) if res_ is True and not ignore_cache else None
                            counter_j += 1
                        counter_i += 1
                    if res:
                        return res
                t_end = time.time()
                pj.timer(t_start, t_end)
                if not supress_print: print(f" {'--' * 5} COMPLETED FOR LOOP ACROSS EXPS {'--' * 5}\n")

            return inner

        return main_for_loop

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

        print('----------------------------------------')
        print('-----Processing trial # %s-----' % trial)
        print('----------------------------------------\n')

        paths = [tiffs_loc_dir, tiffs_loc, paqs_loc]
        # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

        self.tiff_path = paths[1]
        self.paq_path = paths[2]
        TwoPhotonImaging.__init__(self, self.tiff_path, self.paq_path, metainfo=metainfo,
                                  analysis_save_path=analysis_save_path, save_downsampled_tiff=True, quick=False)
        self.paqProcessing()

        # add all frames as bad frames incase want to include this trial in suite2p run
        paq = paq_read(file_path=self.paq_path, plot=False)
        self.bad_frames = frames_discard(paq=paq[0], input_array=None, total_frames=self.n_frames, discard_all=True)

        self.save(pkl_path=self.pkl_path)

        print('\n-----DONE OnePhotonStim init of trial # %s-----' % trial)

        self.photostim_results = pd.DataFrame  # row: [stim type, photostim responses, decay constant, pre stim flu avg], col: [individual stim frames]

    def __repr__(self):
        if os.path.exists(self.pkl_path) and hasattr(self, 'metainfo'):
            lastmod = time.ctime(os.path.getmtime(self.pkl_path))
            prep = self.metainfo['animal prep.']
            trial = self.metainfo['trial']
            information = f"{prep} {trial}, {self.exptype}"
        else:
            information = f"uninitialized"
        return repr(f"({information}) [OnePhotonStimMain]TwoPhotonImaging.OnePhotonStim experimental data object, last saved: {lastmod}")

    @staticmethod
    def import_1pexobj(prep=None, trial=None, date=None, pkl_path=None, verbose=False, load_backup_path=None):
        # if need to load from backup path!
        if load_backup_path:
            pkl_path = load_backup_path
            print(f"**** loading from backup path! ****")

        if pkl_path is None:
            if date is None:
                try:
                    date = onePresults.mean_stim_responses.loc[(onePresults.mean_stim_responses.Prep == f"{prep}") & (
                                onePresults.mean_stim_responses.Trial == f'{trial}'), 'pkl_list'].values[0][30:40]
                except ValueError:
                    raise ValueError('not able to find date in onePresults')
            pkl_path = f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/{date}_{trial}.pkl"

        if not os.path.exists(pkl_path):
            raise Exception('pkl path NOT found: ' + pkl_path)
        with open(pkl_path, 'rb') as f:
            print(f'\- Loading {pkl_path}', end='\r')
            try:
                expobj = pickle.load(f)
                if expobj.analysis_save_path != f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/":
                    expobj.analysis_save_path = f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/"
                    expobj.save_pkl(pkl_path=expobj.pkl_path)
            except pickle.UnpicklingError:
                raise pickle.UnpicklingError(f"\n** FAILED IMPORT OF * {prep} {trial} * from {pkl_path}\n")
            except AttributeError:
                print(f"WARNING: needing to try using CustomUnpicklerAttributeError!")
                    # expobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()

            experiment = f"{expobj.t_series_name} {expobj.metainfo['exptype']} {expobj.metainfo['comments']}"
            print(f'|- Loaded {expobj.t_series_name} {expobj.metainfo["exptype"]}')
            print(f'|- Loaded {experiment}') if verbose else None

            return expobj, experiment

    @property
    def pre_stim(self):
        return 1  # seconds

    @property
    def post_stim(self):
        return 4  # seconds

    @property
    def response_len(self):
        return 0.5  # post-stim response period in sec

    @property
    def experiment_time(self):
        """total timepoints collected for experiment (timed from first stim to the end of last stim + stim interval)"""
        time_length = np.arange(0, ((self.stim_start_frames[-1] / self.fps) + self.stim_interval), 1 / self.fps)
        return time_length

    @property
    def experiment_frames(self):
        """total framepoints collected for experiment (timed from first stim to the end of last stim + stim interval)"""
        # time_length = np.arange(0, ((self.stim_start_frames[-1] / self.fps) + self.stim_interval), 1 / self.fps)
        experiment_frames = np.arange(self.stim_start_frames[0], self.stim_start_frames[-1] + self.stim_interval_fr)
        return experiment_frames

    @property
    def stim_interval_fr(self):
        intervals = [(self.stim_start_frames[i + 1]) - (self.stim_start_frames[i]) for i, _ in enumerate(self.stim_start_frames[:-1])]
        return round(float(np.median([interval for interval in intervals if interval < 200])))  # filtering for stim intervals below 200fr apart only, because >200 will be multiple stim trials protocols


    @property
    def stim_interval(self):
        intervals = [(self.stim_start_times[i + 1]/self.paq_rate) - (self.stim_start_times[i]/self.paq_rate) for i, _ in enumerate(self.stim_start_times[:-1])]
        return round(float(np.mean(intervals)), 1)

    @property
    def shutter_interval_fr(self):
        intervals = [(self.shutter_end_frames[0][i]) - (self.shutter_start_frames[0][i]) for i, _ in enumerate(self.shutter_start_frames)]
        return round(float(np.mean(intervals)))

    @property
    def baseline_fr(self):
        assert 'pre' in self.exptype, 'wrong exptype for this stim condition'
        return np.array([fr for fr in range(self.n_frames)])

    @property
    def interictal_fr(self):
        assert 'post' in self.exptype, 'wrong exptype for this stim condition'
        return np.array([fr for fr in range(self.n_frames) if fr in self.seizure_frames])


    @property
    def sz_liklihood_fr(self):
        """occurence of a seizure relative to imaging frame times binned by 0.5msec."""
        sz_prob = [0] * self.n_frames
        bin_width = int(0.5 * self.fps)
        frame_binned = np.arange(0, self.n_frames, bin_width)
        for idx, fr in enumerate(frame_binned[:-1]):
            slic = (fr, frame_binned[idx + 1])
            sz_s = [1 for sz_start in self.seizure_lfp_onsets if sz_start in range(slic[0], slic[1])]
            total_sz = np.sum(sz_s) if len(sz_s) > 0 else 0
            sz_prob[slic[0]: slic[1]] = [total_sz] * (slic[1] - slic[0])

        return np.asarray(sz_prob)



    @property
    def sz_occurrence_stim_intervals(self):

        bin_width = int(0.5 * self.fps)
        sz_prob = np.asarray([0] * int(self.stim_interval_fr / bin_width))

        for idx, start in enumerate(self.stim_start_frames[:-1]):
            frame_binned = np.arange(start, start + self.stim_interval_fr, bin_width)
            _sz_prob = np.asarray([0] * int(self.stim_interval_fr / bin_width))

            for jdx, fr in enumerate(frame_binned[:-1]):
                # slic = (fr, frame_binned[jdx + 1])
                sz_s = [1 for sz_start in self.seizure_lfp_onsets if sz_start in range(fr, frame_binned[jdx + 1])]
                total_sz = np.sum(sz_s) if len(sz_s) > 0 else 0
                _sz_prob[jdx] = total_sz

            sz_prob += _sz_prob


        return np.asarray(sz_prob)

    @property
    def sz_occurrence_stim_intervals2(self):
        """returns the fraction of total sz recorded across current trial at a certain bin relative to interval between one photon stims."""
        bin_width = int(1 * self.fps)  # 1 second bin is needed because the manual timing of sz onset is uncertain at best...
        sz_prob = np.asarray([0] * int(self.stim_interval_fr / bin_width))


        for i, stim in enumerate(self.stim_start_frames[:-1]):
            frame_binned = np.arange(stim - bin_width // 2, stim + self.stim_interval_fr - bin_width // 2, bin_width)
            _sz_prob = np.asarray([0] * int(self.stim_interval_fr / bin_width))

            for jdx, fr in enumerate(frame_binned[:-1]):
                low_fr = fr #- bin_width // 2
                high_fr = fr + bin_width #// 2
                # sz_s = [1 for sz_start in self.seizure_lfp_onsets if sz_start in range(low_fr, high_fr)]
                # total_sz = np.sum(sz_s) if len(sz_s) > 0 else 0
                for sz_start in self.seizure_lfp_onsets:
                    if sz_start in range(low_fr, high_fr):
                        sz_prob[jdx] += 1

            # sz_prob += _sz_prob

        # return np.asarray(sz_prob / len([i for i in self.seizure_lfp_onsets if i < self.stim_start_frames[-1]]))
        print(np.asarray(sz_prob / len([i for i in self.seizure_lfp_onsets if i < self.stim_start_frames[-1]])))
        return np.asarray(sz_prob)




    # @property -- really hard to get done.... not messing with this right now...
    # def fov_trace_shutter_blanked_dff_pre_norm(self):
    #     """mean FOV flu trace, but with the shutter frames blanked to 0."""
    #     # new_trace = [-100] * len(self.fov_trace_shutter_blanked)
    #     slic = [self.shutter_start_frames[0][0] - self.stim_interval_fr, (self.shutter_start_frames[0][-1] + self.stim_interval_fr)]
    #     if slic[0] < 0: slic[0] = self.shutter_start_frames[0][0]
    #     new_trace = self.fov_trace_shutter_blanked[slic[0]: slic[1]]
    #
    #     norm_mean = np.mean(self.meanRawFluTrace[self.interictal_fr])
    #
    #     for i, fr in enumerate(self.shutter_start_frames[0]):
    #         pre_slice = [fr - self.stim_interval_fr, fr]
    #         if pre_slice[0] < 0: pre_slice[0] = 0
    #         pre_mean = np.mean(self.fov_trace_shutter_blanked[pre_slice[0]: pre_slice[1]])
    #         new_trace[fr - self.stim_interval_fr: fr + self.stim_interval_fr] -= norm_mean
    #         # new_trace[fr - self.stim_interval_fr: fr + self.stim_interval_fr] /= pre_mean
    #
    #     plt.figure(figsize=(30,3))
    #     plt.plot(new_trace)
    #     plt.show()
    #
    #     return new_trace

    @property
    def fov_trace_shutter_blanked_dff(self):
        """mean FOV flu trace, but with the shutter frames blanked to 0."""
        # new_trace = [np.mean(self.meanRawFluTrace)] * len(self.meanRawFluTrace)
        new_trace = (self.fov_trace_shutter_blanked - np.mean(self.meanRawFluTrace)) / np.mean(self.meanRawFluTrace)

        # plt.figure(figsize=(30,3))
        # plt.plot(new_trace)
        # plt.show()

        return new_trace

    @property
    def fov_trace_shutter_blanked(self):
        """mean FOV flu trace, but with the shutter frames blanked to 0."""
        # new_trace = [np.mean(self.meanRawFluTrace)] * len(self.meanRawFluTrace)
        new_trace = self.meanRawFluTrace

        for i, fr in enumerate(self.shutter_frames[0]):
            new_trace[fr] = 0
            new_trace[fr - 1] = 0
            new_trace[fr + 1] = 0
            # new_trace[fr + 3] = np.mean(self.meanRawFluTrace)
            # new_trace[fr + 4] = np.mean(self.meanRawFluTrace)
            # new_trace[fr + 5] = np.mean(self.meanRawFluTrace)

        # plt.figure(figsize=(30,3))
        # plt.plot(self.meanRawFluTrace)
        # plt.show()

        # for i, flu in enumerate(self.meanRawFluTrace):
        #     if i in self.shutter_frames[0]:
        #         pass
        #     else:
        #         new_trace[i] = flu
        return new_trace


    def paqProcessing(self, **kwargs):

        print('\n-----processing paq file for 1p photostim...')

        print('loading', self.paq_path)

        paq, _ = paq_read(self.paq_path, plot=True)
        self.paq_rate = paq['rate']

        frame_rate = self.fps / self.n_planes

        # find frame_clock times
        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = pj.threshold_detect(clock_voltage, 1)
        self.frame_clock = frame_clock

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

        f, ax = plt.subplots(figsize=(20, 2))
        # plt.figure(figsize=(50, 2))
        ax.plot(clock_voltage)
        ax.plot(frame_clock, np.ones(len(frame_clock)), '.', color='orange')
        ax.plot(self.frame_clock_actual, np.ones(len(self.frame_clock_actual)), '.', color='red')
        ax.set_title('frame clock from paq, with detected frame clock instances as scatter')
        ax.set_xlim([1e6, 1.2e6])
        f.tight_layout(pad=2)
        f.show()

        # find 1p stim times
        opto_loopback_chan = paq['chan_names'].index('opto_loopback')
        stim_volts = paq['data'][opto_loopback_chan, :]
        stim_times = pj.threshold_detect(stim_volts, 1)

        self.stim_times = stim_times
        self.stim_start_times = [self.stim_times[0]]  # initialize ls
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

        print("\nStim duration of 1photon stim: %s frames (%s ms)" % (
        self.stim_duration_frames, round(self.stim_duration_frames / self.fps * 1000)))

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

        # find voltage channel and save as lfp_signal attribute
        voltage_idx = paq['chan_names'].index('voltage')
        self.lfp_signal = paq['data'][voltage_idx]

    def collect_seizures_info(self, seizures_lfp_timing_matarray=None, discard_all=True):
        print('\ncollecting information about seizures...')
        self.seizures_lfp_timing_matarray = seizures_lfp_timing_matarray  # path to the matlab array containing paired measurements of seizures onset and offsets

        # retrieve seizure onset and offset times from the seizures info array input
        paq = paq_read(file_path=self.paq_path, plot=False)

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
        self.save()


# %%
class OnePhotonStimPlots:
    @staticmethod
    def plotPrestimF_photostimFlu(run_pre4ap_trials=True, run_post4ap_trials=True, interictal=False, ictal=False, run_trials=[], skip_trials=[],
                                  fig=None, ax=None, **kwargs):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials,
                                          run_trials=run_trials, skip_trials=skip_trials, ignore_cache=True, supress_print=True, kwargs=kwargs)
        def _plotPrestimF_photostimFlu(fig=fig, ax=ax, **kwargs):
            expobj = kwargs['expobj']
            fig = fig if fig else None
            show = False if fig else True
            ax = ax if ax else None

            kwargs2 = kwargs['kwargs']


            if 'pre' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list], [expobj.photostim_flu_responses], supress_print=True,
                                        x_label='pre_stim_flu', y_label='photostim_responses', ax_titles=['Pre-stim Flu vs. photostim responses'],
                                        fig=fig, ax=ax, show=show, colors=['grey'], **kwargs2)
                return True

            elif 'post' in expobj.exptype:
                if ictal:
                    pj.make_general_scatter([expobj.pre_stim_flu_list_ic], [expobj.photostim_flu_responses_ic], supress_print=True,
                                            x_label='pre_stim_flu', y_label='photostim_responses', fig=fig, ax=ax, show=show,
                                            ax_titles=['Pre-stim Flu vs. photostim responses (ictal)'], colors=['purple'], **kwargs2)
                if interictal:
                    pj.make_general_scatter([expobj.pre_stim_flu_list_interic], [expobj.photostim_flu_responses_interic], supress_print=True,
                                            x_label='pre_stim_flu', y_label='photostim_responses', fig=fig, ax=ax, show=show,
                                            ax_titles=['Pre-stim Flu vs. photostim responses (inter-ictal)'], colors=['green'], **kwargs2)

        return _plotPrestimF_photostimFlu()


    @staticmethod
    def plotPrestimF_decayconstant(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False, run_trials=[], skip_trials=[],
                                  fig=None, ax=None, x_lim=[0,2000]):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials, ignore_cache=True,
                                          run_trials=run_trials, skip_trials=skip_trials, supress_print=True)
        def _plotPrestimF_decayconstant(fig=fig, ax=ax, **kwargs):
            expobj = kwargs['expobj']
            fig = fig if fig else None
            show = False if fig else True
            ax = ax if ax else None

            if 'pre' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list], [expobj.decay_constants], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax, show=show,
                                        ax_titles=['Pre-stim Flu vs. decay constants'], x_lim=x_lim, colors=['grey'])

            elif 'post' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list_ic], [expobj.decay_constants_ic], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax, show=show,
                                        x_lim=x_lim, colors=['purple'])

                pj.make_general_scatter([expobj.pre_stim_flu_list_interic], [expobj.decay_constants_interic], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax, show=show,
                                        ax_titles=['Pre-stim Flu vs. decay constants (inter-ictal green, ictal purple)'], x_lim=x_lim, colors=['green'])


        return _plotPrestimF_decayconstant()

    @staticmethod
    def plotTimeToOnset_preStimFlu(run_pre4ap_trials=True, run_post4ap_trials=True, run_trials=[], skip_trials=[], fig=None, ax=None,
                                   **kwargs):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials, ignore_cache=True,
                                          run_trials=run_trials, skip_trials=skip_trials, supress_print=False, kwargs=kwargs)
        def _plotTimeToOnset_preStimFlu(fig=fig, ax=ax, **kwargs):
            print('start')
            expobj = kwargs['expobj']
            fig = fig if fig else None
            show = False if fig else True
            ax = ax if ax else None

            kwargs2 = kwargs['kwargs']

            if 'pre' in expobj.exptype:
                stims = expobj.stim_start_frames
                pj.make_general_scatter(x_list=[np.random.random(len(stims)) * 1.5],
                                        y_data=[expobj.photostim_results.loc['pre stim flu avg', stims]],
                                        supress_print=True, x_label='time to onset (secs)', y_label='pre stim F (a.u.)', fig=fig,
                                        ax=ax, show=show, colors=['grey'], **kwargs2)
                ax.set_title(f'time to onset vs. pre stim F', wrap=True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.spines['left'].set_visible(False)


            elif 'post' in expobj.exptype:
                stims = expobj.stims_out_sz
                pj.make_general_scatter(x_list=[expobj.photostim_results.loc['time to seizure onset (secs)', stims]],
                                        y_data=[expobj.photostim_results.loc['pre stim flu avg', stims]],
                                        supress_print=True, x_label='time to onset (secs)', y_label='pre stim F (a.u.)', fig=fig,
                                        ax=ax, show=show, colors=['green'], **kwargs2)

                stims = expobj.stims_in_sz
                pj.make_general_scatter(x_list=[expobj.photostim_results.loc['time to seizure onset (secs)', stims]],
                                        y_data=[expobj.photostim_results.loc['pre stim flu avg', stims]],
                                        supress_print=True, x_label='time to onset (secs)', y_label='pre stim F (a.u.)', fig=fig,
                                        ax=ax, show=show, ax_titles=['time to onset vs. pre stim F (inter-ictal green, ictal purple)'],
                                        colors=['purple'], **kwargs2)
                ax.set_title(f'time to onset vs. pre stim F (inter-ictal green, ictal purple)',wrap=True)
        return _plotTimeToOnset_preStimFlu()


    @staticmethod
    def plotTimeToOnset_photostimResponse(run_pre4ap_trials=True, run_post4ap_trials=True,
                                          run_trials=[], skip_trials=[], fig=None, ax=None, **kwargs):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials,
                                               run_post4ap_trials=run_post4ap_trials, ignore_cache=True,
                                               run_trials=run_trials, skip_trials=skip_trials, supress_print=False, kwargs=kwargs)
        def _plotTimeToOnset_photostimResponse(fig=fig, ax=ax, **kwargs):
            print('start')
            expobj = kwargs['expobj']
            fig = fig if fig else None
            show = False if fig else True
            ax = ax if ax else None

            kwargs2 = kwargs['kwargs']

            if 'pre' in expobj.exptype:
                stims = expobj.stim_start_frames
                pj.make_general_scatter(x_list=[np.random.random(len(stims)) * 1.5],
                                        y_data=[expobj.photostim_results.loc['photostim responses', stims]],
                                        supress_print=True, x_label='time to onset (secs)', y_label='photostim responses (dFF)', fig=fig,
                                        ax=ax, show=show, colors=['grey'], **kwargs2)
                ax.set_title(f'time to onset vs. photostim responses', wrap=True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            elif 'post' in expobj.exptype:
                stims = expobj.stims_out_sz
                pj.make_general_scatter(x_list=[expobj.photostim_results.loc['time to seizure onset (secs)', stims]],
                                        y_data=[expobj.photostim_results.loc['photostim responses', stims]],
                                        supress_print=True, x_label='time to onset (secs)', y_label='photostim responses (dFF)', fig=fig,
                                        ax=ax, show=show, colors=['green'], **kwargs2)

                stims = expobj.stims_in_sz
                pj.make_general_scatter(x_list=[expobj.photostim_results.loc['time to seizure onset (secs)', stims]],
                                        y_data=[expobj.photostim_results.loc['photostim responses', stims]],
                                        supress_print=True, x_label='time to onset (secs)', y_label='photostim responses (dFF)', fig=fig,
                                        ax=ax, show=show, colors=['purple'], **kwargs2)
                ax.set_title(f'time to onset vs. photostim responses (inter-ictal green, ictal purple)', wrap=True)

        return _plotTimeToOnset_photostimResponse()


# %%


def checkAttr(attr: str):
    @OnePhotonStim.runOverExperiments(ignore_cache=True)
    def _checkAttr(attr=attr, **kwargs):
        expobj = kwargs['expobj']
        try:
            assert hasattr(expobj, attr)
            return 'Passed'
        except AssertionError:
            print(f"\n\n\t***** {expobj.t_series_name} no attr: {attr}\n\n")
            return 'Failed'

    if 'Failed' not in _checkAttr():
        print('ALL CHECKS PASSED')

# checkAttr(attr='pre_stim_flu_list')


# %%
onePresults = import_resultsobj(pkl_path=onePresults_object_path)

