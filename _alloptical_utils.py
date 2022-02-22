#### NOTE: THIS IS NOT CURRENTLY SETUP TO BE ABLE TO HANDLE MULTIPLE GROUPS/STIMS (IT'S REALLY ONLY FOR A SINGLE STIM TRIGGER PHOTOSTIM RESPONSES)

import functools
from pathlib import Path
from datetime import datetime


import os
import sys

from _utils_.io import import_expobj, allopticalResults

sys.path.append('/home/pshah/Documents/code/')

import time
import numpy as np
import pandas as pd
from funcsforprajay import funcs as pj
import pickle

# %% SET OPTIONS
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("expand_frame_repr", True)


# %% DECORATORS
## DECORATORS

# ALL OPTICAL EXPERIMENTS RUN
def run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, skip_trials=[], run_trials=[],
                             set_cache=True, allow_rerun=False, supress_print=False):
    """decorator to use for for-looping through experiment trials across run_pre4ap_trials and run_post4ap_trials.
    the trials to for loop through are defined in allopticalResults.pre_4ap_trials and allopticalResults.post_4ap_trials.

    NOTE: WHEN RETURNING ITEMS IN FUNCTIONS THAT ARE DECORATED USING THIS DECORATOR, THE ITEMS FROM ALL ITERATIONS ARE
    RETURNED AS A LIST. EACH FOR LOOP'S RETURN ITEM IS APPENDED INTO A LIST BY THIS DECORATOR.

    """
    # if len(run_trials) > 0 or run_pre4ap_trials is True or run_post4ap_trials is True:
    print(f"\n {'..'*5} [1] INITIATING FOR LOOP DECORATOR {'..'*5}\n")
    t_start = time.time()
    def main_for_loop(func):
        print(f"\n {'..' * 5} [2] RETURNING FOR LOOP DECORATOR {'..' * 5}\n")
        @functools.wraps(func)
        def inner(*args, **kwargs):
            print(f"\n {'..' * 5} [3] INITIATING FOR LOOP ACROSS EXPS FOR func: {func} {'..' * 5}\n")

            if run_trials:
                print(f"\n{'-' * 5} RUNNING SPECIFIED TRIALS from `trials_run` {'-' * 5}")
                counter1 = 0
                res = []
                for i, exp_prep in enumerate(run_trials):
                    # print(i, exp_prep)
                    try:  # dont continue if exp_prep already run before (as determined by location in func_cache
                        if get_from_cache(func.__name__, item=exp_prep) and not allow_rerun:
                            run = False
                            if not supress_print: print(
                                f"{exp_prep} found in previously completed record for func {func.__name__} ... skipping repeat run.")
                        else:
                            run = True
                    except KeyError:
                        run = True
                    if run:
                        prep = exp_prep[:-6]
                        trial = exp_prep[-5:]
                        try:
                            expobj = import_expobj(prep=prep, trial=trial, verbose=False)
                        except:
                            raise ImportError(f"IMPORT ERROR IN {prep} {trial}")
                        working_on(expobj) if not supress_print else None
                        res_ = func(expobj=expobj, **kwargs)
                        # try:
                        #     func(expobj=expobj, **kwargs)
                        # except:
                        #     print('Exception on the wrapped function call')
                        end_working_on(expobj) if not supress_print else None
                        res.append(res_) if res_ is not None else None
                        set_to_cache(func_name=func.__name__, item=exp_prep) if set_cache else None
                counter1 += 1


            if run_pre4ap_trials:
                print(f"\n{'-' * 5} RUNNING PRE4AP TRIALS {'-' * 5}")
                counter_i = 0
                res = []
                for i, x in enumerate(allopticalResults.pre_4ap_trials):
                    counter_j = 0
                    for j, exp_prep in enumerate(x):
                        if exp_prep in skip_trials:
                            pass
                        else:
                            # print(i, exp_prep)
                            try:  # dont continue if exp_prep already run before (as determined by location in func_cache
                                if get_from_cache(func.__name__, item=exp_prep) and not allow_rerun:
                                    run = False
                                    if not supress_print: print(
                                        f"{exp_prep} found in cache for func {func.__name__} ... skipping repeat run.")
                                else:
                                    run = True
                            except KeyError:
                                run = True
                            if run is True:
                                prep = exp_prep[:-6]
                                pre4aptrial = exp_prep[-5:]
                                try:
                                    expobj = import_expobj(prep=prep, trial=pre4aptrial, verbose=False)
                                except:
                                    raise ImportError(f"IMPORT ERROR IN {prep} {pre4aptrial}")

                                working_on(expobj) if not supress_print else None
                                res_ = func(expobj=expobj, **kwargs)
                                # try:
                                #     func(expobj=expobj, **kwargs)
                                # except:
                                #     print('Exception on the wrapped function call')
                                end_working_on(expobj) if not supress_print else None
                                res.append(res_) if res_ is not None else None
                                set_to_cache(func_name=func.__name__, item=exp_prep) if set_cache else None

                        counter_j += 1
                    counter_i += 1
                if res:
                    return res

            if run_post4ap_trials:
                print(f"\n{'-' * 5} RUNNING POST4AP TRIALS {'-' * 5}")
                counter_i = 0
                res = []
                for i, x in enumerate(allopticalResults.post_4ap_trials):
                    counter_j = 0
                    for j, exp_prep in enumerate(x):
                        if exp_prep in skip_trials:
                            pass
                        else:
                            # print(i, exp_prep)
                            try:  # dont continue if exp_prep already run before (as determined by location in func_cache
                                if get_from_cache(func.__name__, item=exp_prep) and not allow_rerun:
                                    run = False
                                    if not supress_print: print(
                                        f"{exp_prep} found in cache for func {func.__name__} ... skipping repeat run.")
                                else:
                                    run = True
                            except KeyError:
                                run = True
                            if run:
                                prep = exp_prep[:-6]
                                post4aptrial = exp_prep[-5:]
                                try:
                                    expobj = import_expobj(prep=prep, trial=post4aptrial, verbose=False)
                                except:
                                    raise ImportError(f"IMPORT ERROR IN {prep} {post4aptrial}")

                                working_on(expobj) if not supress_print else None
                                res_ = func(expobj=expobj, **kwargs)
                                # try:
                                #     func(expobj=expobj, **kwargs)
                                # except:
                                #     print('Exception on the wrapped function call')
                                end_working_on(expobj) if not supress_print else None
                                res.append(res_) if res_ is not None else None
                                set_to_cache(func_name=func.__name__, item=exp_prep) if set_cache else None

                        counter_j += 1
                    counter_i += 1
                if res:
                    return res
            t_end = time.time()
            pj.timer(t_start, t_end)
            print(f" {'--' * 5} COMPLETED FOR LOOP ACROSS EXPS {'--' * 5}\n")
        return inner
    return main_for_loop

# %% UTILITY CLASSES
class ExpobjStrippedTimeDelaySzInvasion:
    def __init__(self, expobj):
        self.t_series_name = expobj.t_series_name
        self.seizure_lfp_onsets = expobj.seizure_lfp_onsets
        self.seziure_lfp_offsets = expobj.seizure_lfp_offsets
        self.raw_SLMTargets = expobj.raw_SLMTargets
        self.mean_raw_flu_trace = expobj.meanRawFluTrace
        from _utils_._lfp import LFP
        self.lfp_downsampled = LFP.downsampled_LFP(expobj=expobj)
        self.save(expobj=expobj)

    def __repr__(self):
        return f"ExpobjStrippedTimeDelaySzInvasion - {self.t_series_name}"

    def save(self, expobj):
        path = expobj.analysis_save_path + '/export/'
        os.makedirs(path, exist_ok=True)
        path += f"{expobj.prep}_{expobj.trial}_stripped.pkl"
        with open(f'{path}', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            print(f"|- Successfully saved stripped expobj: {expobj.prep}_{expobj.trial} to {path}")




# %% UTILITY FUNCTIONS

# caching func
cache_folder = '/home/pshah/mnt/qnap/Analysis/temp/cache/'
cache_path = f"{cache_folder}func_run_cache.p"

def __load_cache():
    if os.path.exists(cache_path):
        func_dict = pickle.load(open(f"{cache_path}", 'rb'))
        return func_dict
    else:
        raise FileNotFoundError(f"{cache_path}")


def set_to_cache(func_name=None, item=None, reset_cache=False):
    if reset_cache:
        func_dict = {}
        print(f"resetting cache pkl located at: {cache_path}")
    else:
        if not os.path.exists(f"{cache_path}"):
            func_dict = {}
        else:
            func_dict = __load_cache()
            if func_name not in [*func_dict]:
                func_dict[func_name] = []
            func_dict[func_name].append(item)

    pickle.dump(func_dict, open(f"{cache_path}", "wb"))

def get_from_cache(func_name, item):
    func_dict = __load_cache()
    return True if item in func_dict[func_name] else False

def delete_from_cache(func_name: str):
    func_dict = __load_cache()
    if func_name in [*func_dict]:
        func_dict.pop(func_name, None)
        if func_name not in [*func_dict]:
            pickle.dump(func_dict, open(f"{cache_path}", "wb"))
            print(f"Deleted {func_name} from cache.")
        else:
            print(f"Delete failed: {func_name} found in cache, but was not able to delete. unexpected error.")
    else:
        print(f'{func_name} was not found in cache. nothing deleted.')


# random plot just to initialize plotting for PyCharm
def random_plot():
    pj.make_general_scatter(x_list=[np.random.rand(5)], y_data=[np.random.rand(5)], s=60, alpha=1, figsize=(3,3))

def working_on(expobj):
    print(f"STARTING on: {expobj.metainfo['exptype']} {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} {'.'*15}")

def end_working_on(expobj):
    print(f"{'.'*19} {expobj.metainfo['exptype']} {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} FINISHED \n")

def save_figure(fig, save_path_suffix: str = None, save_path_full: str = None):
    if not save_path_full and save_path_suffix:
        ## SET DEFAULT FIGURE SAVE DIRECTORY
        today_date = datetime.today().strftime('%Y-%m-%d')
        save_path_prefix = f"/home/pshah/mnt/qnap/Analysis/Results_figs/{today_date}/"
        os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None
        save_path_full = save_path_prefix + save_path_suffix
    else:
        ValueError('not able to determine where to save figure to!')
    print(f'\nsaving figure to: {save_path_full}')
    fig.savefig(save_path_full)


def save_to_csv(df: pd.DataFrame, exp_name: str = None, savepath: Path = None):
    if not savepath:
        if exp_name: savepath = Path('/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/' + exp_name + '.csv')
        else: KeyError('trying to make savepath, need to provide `exp_name`')
    savepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(savepath)
    print(f"saved dataframe to {savepath}")

# %% terms dictionary
# dictionary of terms, phrases, etc. that are particular to this analysis
phrase_dictionary = {
    'delta(trace_dFF)': "photostim measurement; measuring photostim responses with post-stim minus pre-stim, where the post-stim "
                        "and pre-stim values are dFF values obtained from normalization of the whole trace within the present t-series",
    'dfprestimf': "photostim meausurement; measuring photostim responses as (post-stim minus pre-stim)/(mean[pre-stim period], "
                  "where the original trace is the raw (neuropil subtracted) trace",
    'dfstdf': "photostim measurement; measuring photostim responses as (post-stim minus pre-stim)/(std[pre-stim period], "
              "where the original trace is the raw (neuropil subtracted) trace",
    'hits': "successful photostim responses, as defined based on a certain threshold level for dfstdf (>0.3 above prestim) and delta(trace_dFF) (>10 above prestim)",
    'SLM Targets': "ROI placed on the motion registered TIFF based on the primary target coordinate and expanding a circle of 10um? diameter centered on the coordinate",
    's2p ROIs': "all ROIs derived directly from the suite2p output",
    's2p nontargets': "suite2p ROIs excluding those which are filtered out for being in the target exclusion zone",
    'good cells': "good cells are suite2p ROIs which have some sort of a Flu transient based on a sliding window and std filtering process",
    'stim_id': "the imaging frame number on which the photostimulation is calculated to have first started",
    'photostim response': 'synonymous with `delta(trace_dFF)`',
    'im_time_sec': 'anndata storage var key that shows the time (in secs) of the photostim frame from the start of the imaging collection'
}

def define(x):
    try:
        print(f"{x}:    {phrase_dictionary[x]}") if type(x) is str else print('ERROR: please provide a string object as the key')
    except KeyError:
        print('input not found in phrase_dictionary, you should CONSIDER ADDING IT RIGHT NOW!')

def get_phrases():
    print(f"entries in phrase_dictionary: \n {[*phrase_dictionary]}")




# %% data processing utils
# paq2py by Llyod Russel
def paq_read(file_path=None, plot=False):
    """
    Read PAQ file (from PackIO) into python
    Lloyd Russell 2015
    Parameters
    ==========
    file_path : str, optional
        full path to file to read in. if none is supplied a load file dialog
        is opened, buggy on mac osx - Tk/matplotlib. Default: None.
    plot : bool, optional
        plot the data after reading? Default: False.
    Returns
    =======
    data : ndarray
        the data as a m-by-n array where m is the number of channels and n is
        the number of datapoints
    chan_names : list of str
        the names of the channels provided in PackIO
    hw_chans : list of str
        the hardware lines corresponding to each channel
    units : list of str
        the units of measurement for each channel
    rate : int
        the acquisition sample rate, in Hz
    """

    # file load gui
    if file_path is None:
        import Tkinter
        import tkFileDialog
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename()
        root.destroy()

    # open file
    fid = open(file_path, 'rb')

    # get sample rate
    rate = int(np.fromfile(fid, dtype='>f', count=1))

    # get number of channels
    num_chans = int(np.fromfile(fid, dtype='>f', count=1))

    # get channel names
    chan_names = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        chan_name = ''
        for j in range(num_chars):
            chan_name = chan_name + chr(np.fromfile(fid, dtype='>f', count=1))
        chan_names.append(chan_name)

    # get channel hardware lines
    hw_chans = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        hw_chan = ''
        for j in range(num_chars):
            hw_chan = hw_chan + chr(np.fromfile(fid, dtype='>f', count=1))
        hw_chans.append(hw_chan)

    # get acquisition units
    units = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        unit = ''
        for j in range(num_chars):
            unit = unit + chr(np.fromfile(fid, dtype='>f', count=1))
        units.append(unit)

    # get data
    temp_data = np.fromfile(fid, dtype='>f', count=-1)
    num_datapoints = int(len(temp_data) / num_chans)
    data = np.reshape(temp_data, [num_datapoints, num_chans]).transpose()

    # close file
    fid.close()

    # plot
    if plot:
        # import matplotlib
        # matplotlib.use('QT4Agg')
        import matplotlib.pylab as plt
        f, axes = plt.subplots(num_chans, 1, sharex=True, figsize=(10, num_chans), frameon=False)
        for idx, ax in enumerate(axes):
            ax.plot(data[idx])
            ax.set_xlim([0, num_datapoints - 1])
            ax.set_ylim([data[idx].min() - 1, data[idx].max() + 1])
            # ax.set_ylabel(units[idx])
            ax.set_title(chan_names[idx])
        plt.tight_layout()
        plt.show()

    return {"data": data,
            "chan_names": chan_names,
            "hw_chans": hw_chans,
            "units": units,
            "rate": rate,
            "num_datapoints": num_datapoints}


# useful for returning indexes when a
def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    frames = np.where(thresh_signal)
    return frames[0]

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

##


