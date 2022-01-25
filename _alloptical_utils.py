#### NOTE: THIS IS NOT CURRENTLY SETUP TO BE ABLE TO HANDLE MULTIPLE GROUPS/STIMS (IT'S REALLY ONLY FOR A SINGLE STIM TRIGGER PHOTOSTIM RESPONSES)

import functools
import re
from datetime import datetime


import os
import sys

sys.path.append('/home/pshah/Documents/code/')

import time
import numpy as np
import pandas as pd
from funcsforprajay import funcs as pj
import pickle

## SET OPTIONS
pd.set_option('max_columns', None)
pd.set_option('max_rows', 10)




# %% UTILITY FUNCTIONS and DECORATORS

def save_pkl(obj, save_path: str = None):
    if save_path is None:
        if not hasattr(obj, 'save_path'):
            raise ValueError(
                'pkl path for saving was not found in object attributes, please provide path to save to')
    else:
        obj.pkl_path = save_path

    with open(obj.pkl_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"\- Saving expobj saved to {obj.pkl_path} -- ")

    backup_dir = pj.return_parent_dir(obj.backup_pkl)
    os.makedirs(backup_dir, exist_ok=True) if not os.path.exists(backup_dir) else None
    with open(obj.backup_pkl, 'wb') as f:
        pickle.dump(obj, f)


def import_expobj(aoresults_map_id: str = None, trial: str = None, prep: str = None, date: str = None, pkl_path: str = None,
                  exp_prep: str = None, verbose: bool = False, do_processing: bool = False, load_backup_path: str = None):
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

    if exp_prep is not None:
        prep = exp_prep[:-6]
        trial = exp_prep[-5:]

    # if need to load from backup path!
    if load_backup_path:
        pkl_path = load_backup_path
        print(f"**** loading from backup path! ****")

    if pkl_path is None:
        if date is None:
            try:
                date = allopticalResults.metainfo.loc[
                    allopticalResults.metainfo['prep_trial'] == f"{prep} {trial}", 'date'].values[0]
            except ValueError:
                raise ValueError('not able to find date in allopticalResults.metainfo')
        pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    if not os.path.exists(pkl_path):
        raise Exception('pkl path NOT found: ' + pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f'\- Loading {pkl_path}', end='\r')
        try:
            expobj = pickle.load(f)
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"\n** FAILED IMPORT OF * {prep} {trial} * from {pkl_path}\n")
        experiment = f"{expobj.t_series_name} {expobj.metainfo['exptype']} {expobj.metainfo['comments']}"
        print(f'|- Loaded {expobj.t_series_name} ({pkl_path}) .. DONE') if not verbose else None
        print(f'|- Loaded {experiment}') if verbose else None

    ### roping in some extraneous processing steps if there's expobj's that haven't completed for them

    # check for existence of backup (if not then make one through the saving func).
    expobj.save() if not os.path.exists(expobj.backup_pkl) else None

    # save the pkl if loaded from backup path
    expobj.save() if load_backup_path else None

    if expobj.analysis_save_path[-1] != '/':
        expobj.analysis_save_path = expobj.analysis_save_path + '/'
        print(f"updated expobj.analysis_save_path to: {expobj.analysis_save_path}")
        expobj.save()

    # move expobj to the official save_path from the provided save_path that expobj was loaded from (if different)
    if pkl_path is not None:
        if expobj.pkl_path != pkl_path:
            expobj.save_pkl(save_path=expobj.pkl_path)
            print('saved new copy of expobj to save_path: ', expobj.pkl_path)

    # other misc. things you want to do when importing expobj -- should be temp code basically - not essential for actual importing of expobj

    return expobj, experiment

def import_resultsobj(pkl_path: str):
    assert os.path.exists(pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f"\nimporting resultsobj from: {pkl_path} ... ")
        resultsobj = pickle.load(f)
        print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
    return resultsobj

# caching func
def set_to_cache(func_name=None, item=None, reset_cache=False):
    cache_folder = '/home/pshah/mnt/Analysis/temp/cache/'
    cache_path = f"{cache_folder}func_run_cache.p"

    if reset_cache:
        func_dict = {}
        print(f"resetting cache pkl located at: {cache_path}")
    else:
        if not os.path.exists(f"{cache_path}"):
            func_dict = {}
        else:
            func_dict = pickle.load(open(f"{cache_path}", 'rb'))
            if func_name not in [*func_dict]:
                func_dict[func_name] = []
            func_dict[func_name].append(item)

    pickle.dump(func_dict, open(f"{cache_path}", "wb"))

def get_from_cache(func_name, item):
    cache_folder = '/home/pshah/mnt/Analysis/temp/cache/'
    cache_path = f"{cache_folder}func_run_cache.p"
    if os.path.exists(cache_path):
        func_dict = pickle.load(open(f"{cache_path}", 'rb'))
        return True if item in func_dict[func_name] else False
    else:
        return False




# random plot just to initialize plotting for PyCharm
def random_plot():
    pj.make_general_scatter(x_list=[np.random.rand(5)], y_data=[np.random.rand(5)], s=60, alpha=1, figsize=(3,3))

def define(x):
    try:
        print(f"{x}:    {phrase_dictionary[x]}") if type(x) is str else print('ERROR: please provide a string object as the key')
    except KeyError:
        print('input not found in phrase_dictionary, you should CONSIDER ADDING IT RIGHT NOW!')

def show_phrases():
    print(f"entries in phrase_dictionary: \n {list(phrase_dictionary.keys())}")

def working_on(expobj):
    print(f"STARTING on: {expobj.metainfo['exptype']} {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ... ")

def end_working_on(expobj):
    print(
        f"FINISHED on: {expobj.metainfo['exptype']} {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} \** \n")

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


##

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
try:
    allopticalResults = import_resultsobj(
        pkl_path=results_object_path)  # this needs to be run AFTER defining the AllOpticalResults class
except FileNotFoundError:
    print(f'not able to get allopticalResults object from {results_object_path}')

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
    'photostim response': 'synonymous with `delta(trace_dFF)`'
}
