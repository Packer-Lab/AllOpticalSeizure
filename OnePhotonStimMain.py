import functools
import os
import pickle
import sys


from alloptical_utils_pj import import_expobj, working_on, end_working_on, import_resultsobj
import _alloptical_utils as Utils

sys.path.append('/home/pshah/Documents/code/')
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from funcsforprajay import funcs as pj

from utils.paq_utils import paq_read, frames_discard
import alloptical_plotting_utils as aoplot

from TwoPhotonImagingMain import TwoPhotonImaging

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
        # 'PS11 t-019',
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
                           ignore_cache=False, supress_print=False):
        """decorator to use for for-looping through experiment trials across run_pre4ap_trials and run_post4ap_trials.
        the trials to for loop through are defined in allopticalResults.pre_4ap_trials and allopticalResults.post_4ap_trials"""
        # if len(run_trials) > 0 or run_pre4ap_trials is True or run_post4ap_trials is True:
        t_start = time.time()

        def main_for_loop(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
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
                            expobj, _ = OnePhotonStim.import_1pexobj(prep=prep, trial=trial, verbose=False)

                            if not supress_print: Utils.working_on(expobj)
                            res_ = func(expobj=expobj, **kwargs)
                            if not supress_print: Utils.end_working_on(expobj)
                            if res_ is not None and type(res_) is not bool: res.append(res_)
                            Utils.set_to_cache(func_name=func.__name__, item=exp_prep) if res_ is True else None

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
                                    expobj, _ = OnePhotonStim.import_1pexobj(prep=prep, trial=trial, verbose=False)

                                    if not supress_print: Utils.working_on(expobj)
                                    res_ = func(expobj=expobj, **kwargs)
                                    if not supress_print: Utils.end_working_on(expobj)
                                    if res_ is not None and type(res_) is not bool: res.append(res_)
                                    Utils.set_to_cache(func_name=func.__name__, item=exp_prep) if res_ is True else None
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
                                    expobj, _ = OnePhotonStim.import_1pexobj(prep=prep, trial=trial, verbose=False)

                                    if not supress_print: Utils.working_on(expobj)
                                    res_ = func(expobj=expobj, **kwargs)
                                    if not supress_print: Utils.end_working_on(expobj)
                                    if res_ is not None and type(res_) is not bool: res.append(res_)
                                    Utils.set_to_cache(func_name=func.__name__, item=exp_prep) if res_ is True else None
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

    def __repr__(self):
        if os.path.exists(self.pkl_path) and hasattr(self, 'metainfo'):
            lastmod = time.ctime(os.path.getmtime(self.pkl_path))
            prep = self.metainfo['animal prep.']
            trial = self.metainfo['trial']
            information = f"{prep} {trial}, {self.exptype}"
        else:
            information = f"uninitialized"
        return repr(f"({information}) TwoPhotonImaging.OnePhotonStim experimental data object, last saved: {lastmod}")

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
                    raise ValueError('not able to find date in allopticalResults.metainfo')
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


results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'

onePresults = import_resultsobj(pkl_path=results_object_path)




# %%
class OnePhotonStimPlots:
    @staticmethod
    def plotPrestimF_photostimFlu(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False, run_trials=[], skip_trials=[],
                                  fig=None, ax=None, x_lim=[0,2000]):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials, ignore_cache=ignore_cache,
                                          run_trials=run_trials, skip_trials=skip_trials, supress_print=True)
        def _plotPrestimF_photostimFlu(fig=fig, ax=ax, **kwargs):
            expobj = kwargs['expobj']
            fig = fig if fig else None
            show = False if fig else True
            ax = ax if ax else None

            if 'pre' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list], [expobj.photostim_flu_responses], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='photostim_responses', fig=fig, ax=ax, show=show,
                                        ax_titles=['Pre-stim Flu vs. photostim responses'], x_lim=x_lim, colors=['grey'])
                return True

            elif 'post' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list_ic], [expobj.photostim_flu_responses_ic], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='photostim_responses', fig=fig, ax=ax, show=show,
                                        ax_titles=['Pre-stim Flu vs. photostim responses (ictal)'], x_lim=x_lim, colors=['purple'])

                pj.make_general_scatter([expobj.pre_stim_flu_list_interic], [expobj.photostim_flu_responses_interic], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='photostim_responses', fig=fig, ax=ax, show=show,
                                        ax_titles=['Pre-stim Flu vs. photostim responses (inter-ictal)'], x_lim=x_lim, colors=['green'])
                return True
            else:
                return False

        return _plotPrestimF_photostimFlu()


    @staticmethod
    def plotPrestimF_decayconstant(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False, run_trials=[], skip_trials=[],
                                  fig=None, ax=None, x_lim=[0,2000]):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials, ignore_cache=ignore_cache,
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
                return True

            elif 'post' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list_ic], [expobj.decay_constants_ic], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax, show=show,
                                        x_lim=x_lim, colors=['purple'])

                pj.make_general_scatter([expobj.pre_stim_flu_list_interic], [expobj.decay_constants_interic], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax, show=show,
                                        ax_titles=['Pre-stim Flu vs. decay constants (inter-ictal green, ictal purple)'], x_lim=x_lim, colors=['green'])
                return True
            else:
                return False


        return _plotPrestimF_decayconstant()

    @staticmethod
    def plotTimeToOnset_photostimResponse(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False, run_trials=[],
                                   skip_trials=[], fig=None, ax=None, x_lim=[0, 2000]):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials,
                                          ignore_cache=ignore_cache,
                                          run_trials=run_trials, skip_trials=skip_trials, supress_print=True)
        def _plotTimeToOnset_photostimResponse(fig=fig, ax=ax, **kwargs):
            expobj = kwargs['expobj']
            fig = fig if fig else None
            show = False if fig else True
            ax = ax if ax else None

            if 'pre' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list], [expobj.decay_constants], s=50, supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax,
                                        show=show,
                                        ax_titles=['Pre-stim Flu vs. decay constants'], x_lim=x_lim, colors=['grey'])
                return True

            elif 'post' in expobj.exptype:
                pj.make_general_scatter([expobj.pre_stim_flu_list_ic], [expobj.decay_constants_ic], s=50,
                                        supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax,
                                        show=show,
                                        x_lim=x_lim, colors=['purple'])

                pj.make_general_scatter([expobj.pre_stim_flu_list_interic], [expobj.decay_constants_interic], s=50,
                                        supress_print=True,
                                        alpha=0.5, x_label='pre_stim_flu', y_label='decay constants', fig=fig, ax=ax,
                                        show=show,
                                        ax_titles=[
                                            'Pre-stim Flu vs. decay constants (inter-ictal green, ictal purple)'],
                                        x_lim=x_lim, colors=['green'])
                return True
            else:
                return False

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



