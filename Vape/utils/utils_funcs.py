import numpy as np
import json
import tifffile as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import copy

# # global plotting params
# params = {'legend.fontsize': 'x-large',
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
# plt.rcParams.update(params)
# sns.set()
# sns.set_style('white')


def intersect(lst1, lst2):
    return list(set(lst1) & set(lst2)) 

def dfof(arr):
    '''takes 1d ls or array or 2d array and returns dfof array of same
       dim (JR 2019) This is extraordinarily slow, use dfof2'''

    if type(arr) is list or type(arr) == np.ndarray and len(arr.shape) == 1:
        F = np.mean(arr)
        dfof_arr = [((f - F) / F) * 100 for f in arr]

    elif type(arr) == np.ndarray and len(arr.shape) == 2:
        dfof_arr = []
        for trace in arr:
            F = np.mean(trace)
            dfof_arr.append([((f - F) / F) * 100 for f in trace])

    else:
        raise NotImplementedError('input type not recognised')

    return np.array(dfof_arr)


def dfof2(flu):
    '''
    delta f over f, this function is orders of magnitude faster 
    than the dumb one above takes input matrix flu 
    (num_cells x num_frames)
    (JR 2019)

    '''

    flu_mean = np.mean(flu, 1)
    flu_mean = np.reshape(flu_mean, (len(flu_mean), 1))
    return (flu - flu_mean) / flu_mean


def get_tiffs(path):

    tiff_files = []
    for file in os.listdir(path):
        if file.endswith('.tif') or file.endswith('.tiff'):
            tiff_files.append(os.path.join(path, file))

    return tiff_files


def s2p_loader(s2p_path, subtract_neuropil=True, neuropil_coeff=0.7):

    found_stat = False

    for root, dirs, files in os.walk(s2p_path):

        for file in files:

            if file == 'F.npy':
                all_cells = np.load(os.path.join(root, file), allow_pickle=True)
            elif file == 'Fneu.npy':
                neuropil = np.load(os.path.join(root, file), allow_pickle=True)
            elif file == 'iscell.npy':
                is_cells = np.load(os.path.join(root, file), 
                                   allow_pickle=True)[:, 0]
                is_cells = np.ndarray.astype(is_cells, 'bool')
                print('Loading {} traces labelled as cells'
                      .format(sum(is_cells)))
            elif file == 'spks.npy':
                spks = np.load(os.path.join(root, file), allow_pickle=True)
            elif file == 'stat.npy':
                stat = np.load(os.path.join(root, file), allow_pickle=True)
                found_stat = True

    if not found_stat:
        raise FileNotFoundError('Could not find stat, '
                                'this is likely not a suit2p folder, s2p_path: ', s2p_path)
    for i, s in enumerate(stat):
        s['original_index'] = i

    all_cells = all_cells[is_cells, :]
    neuropil = neuropil[is_cells, :]
    spks = spks[is_cells, :]
    stat = stat[is_cells]


    if not subtract_neuropil:
        return all_cells, spks, stat, neuropil

    else:
        print('Subtracting neuropil with a coefficient of {}'
              .format(neuropil_coeff))
        neuropil_corrected = all_cells - neuropil * neuropil_coeff
        return neuropil_corrected, spks, stat, neuropil


def correct_s2p_combined(s2p_path, n_planes):

    len_count = 0
    for i in range(n_planes):

        iscell = np.load(os.path.join(s2p_path, 'plane{}'.format(i), 
                                     'iscell.npy'), allow_pickle=True)

        if i == 0:
            allcells = iscell
        else:
            allcells = np.vstack((allcells, iscell))

        len_count += len(iscell)

    combined_iscell = os.path.join(s2p_path, 'combined', 'iscell.npy')

    ic = np.load(combined_iscell, allow_pickle=True)
    assert ic.shape == allcells.shape
    assert len_count == len(ic)

    np.save(combined_iscell, allcells)


def read_fiji(csv_path):
    '''reads the csv file saved through plot z axis profile in fiji'''

    data = []

    with open(csv_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            data.append(float(row[0].split(',')[1]))

    return np.array(data)


def save_fiji(arr):
    '''saves numpy array in current folder as fiji friendly tiff'''
    tf.imsave('Vape_array.tiff', arr.astype('int16'))


def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]


def pade_approx_norminv(p):
    q = math.sqrt(2*math.pi) * (p - 1/2) - (157/231) * math.sqrt(2) * \
        math.pi**(3/2) * (p - 1/2)**3
    r = 1 - (78/77) * math.pi * (p - 1/2)**2 + (241 * math.pi**2 / 2310) * \
        (p - 1/2)**4
    return q/r


def d_prime(hit_rate, false_alarm_rate):
    return pade_approx_norminv(hit_rate) - \
        pade_approx_norminv(false_alarm_rate)


def paq_data(paq, chan_name, threshold_ttl=False, plot=False):
    '''
    returns the data in paq (from paq_read) from channel: chan_names
    if threshold_tll: returns sample that trigger occured on
    '''

    chan_idx = paq['chan_names'].index(chan_name)
    data = paq['data'][chan_idx, :]
    if threshold_ttl:
        data = threshold_detect(data, 1)

    if plot:
        if threshold_ttl:
            plt.plot(data, np.ones(len(data)), '.')
        else:
            plt.plot(data)

    return data


def stim_start_frame_mat(stim_times, frames_ms, fs=5, debug_print=False):
    ''' function to replace stim_start_frames
        Inputs:
        stim_times -- times that stim occured (same reference frame
                      as frames_ms
        frames_ms -- matrix of cell frame times [num_cells x num_frames]
        fs -- frame rate of the imaging for an inidividual plane (frames/second)
        debug_print -- whether to print useful debugging statment about each
                       stim and associated frame time
        Returns:
        stim_idxs -- matrix of frame indexes that stim occured on for each cell
                     [num_trials x num_cells]

        '''

    # The substituion with -1 causes an inplace mutation of the start_times 
    # variable in the run objects, copy to avoid this
    stim_times_copy = copy.deepcopy(stim_times)

    # get rid of stims that occur outside the frame clock
    max_frame = np.nanmax(frames_ms)
    min_frame = np.nanmin(frames_ms)
    keep_idx = np.repeat(False, len(stim_times_copy))
    keep_idx[np.where((stim_times_copy < max_frame) & (
        stim_times_copy > min_frame))[0]] = True
    stim_times_copy[~keep_idx] = -1

    ifi_ms = 1/fs * 1000  # inter-frame-interval in ms
    arg_sorted = np.argsort(frames_ms, axis=1)
    n_cells = frames_ms.shape[0]

    for i, stim_time in enumerate(stim_times_copy):

        def closest_finder(arr, sorter): return np.searchsorted(
            arr, stim_time, sorter=sorter)
        arg_idx = [closest_finder(frames_ms[i, :], arg_sorted[i, :])
                   for i in range(n_cells)]
        stim_idx = arg_sorted[np.arange(len(arg_idx)), arg_idx]

        # times of the indexes
        vals = frames_ms[np.arange(len(stim_idx)), stim_idx]

        # if the closest frame is after the stim, move one back
        ahead_idx = np.where(vals > stim_time)[0]
        stim_idx[ahead_idx] = stim_idx[ahead_idx] - 1
        # times of the indexes
        vals = frames_ms[np.arange(len(stim_idx)), stim_idx]

        # if there are nans in the frames_ms row then this stim was likely
        # not imaged. If the stim is > inter-frame_interval from a
        # frame then discount this stim (currently only checking this
        # on the first cell
        if np.isnan(vals).any() or abs(stim_time - vals[0]) > ifi_ms or \
        stim_time == -1:
            stim_idx = np.full(stim_idx.shape, np.nan)
        else:
            if debug_print:
                print('\nval is {}'.format(vals[0]))
                print('stim_time is {}'.format(stim_time))
                print('stim_idx is {}'.format(max(stim_idx)))

            vals2 = frames_ms[range(len(stim_idx)), stim_idx]

            if debug_print:
                print('val2 is {}'.format(vals2[0]))

        if i == 0:
            stim_idxs = stim_idx
        else:
            stim_idxs = np.vstack((stim_idxs, stim_idx))

    return np.transpose(stim_idxs)


def stim_start_frame(paq=None, stim_chan_name=None, frame_clock=None,
                     stim_times=None):
    '''Returns the frames from a frame_clock that a stim occured on.
       Either give paq and stim_chan_name as arugments if using 
       unprocessed paq. 
       Or predigitised frame_clock and stim_times in reference frame
       of that clock

    '''

    if frame_clock is None:
        frame_clock = paq_data(paq, 'frame_clock', threshold_ttl=True)
        stim_times = paq_data(paq, stim_chan_name, threshold_ttl=True)

    stim_times = [stim for stim in stim_times if stim < np.nanmax(frame_clock)]

    frames = []

    for stim in stim_times:
        # the sample time of the frame immediately preceeding stim
        frame = next(frame_clock[i-1] for i, sample in enumerate(frame_clock)
                     if sample - stim > 0)
        frames.append(frame)

    return np.array(frames)


def myround(x, base=5):
    '''allow rounding to nearest base number for
       use with multiplane stack slicing'''

    return base * round(x/base)


def tseries_finder(tseries_lens, frame_clock, paq_rate=20000):

    ''' Finds chunks of frame clock that correspond to the tseries in 
        tseries lens
        tseries_lens -- ls of the number of frames each tseries you want
                        to find contains
        frame_clock  -- thresholded times each frame recorded in paqio occured
        paq_rate     -- input sampling rate of paqio

        '''

    # frame clock recorded in paqio, includes TTLs from cliking 'live' 
    # and foxy extras
    clock = frame_clock / paq_rate

    # break the recorded frame clock up into individual aquisitions
    # where TTLs are seperated by more than 1s
    gap_idx = np.where(np.diff(clock) > 1)
    gap_idx = np.insert(gap_idx, 0, 0)
    gap_idx = np.append(gap_idx, len(clock))
    chunked_paqio = np.diff(gap_idx)

    # are each of the frames recorded by the frame clock actually 
    # in processed tseries?
    real_frames = np.zeros(len(clock))
    # the max number of extra frames foxy could spit out
    foxy_limit = 20
    # the number of tseries blocks that have already been found
    series_found = 0
    # count how many frames have been labelled as real or not
    counter = 0

    for chunk in chunked_paqio:
        is_tseries = False

        # iterate through the actual length of each analysed tseries
        for idx, ts in enumerate(tseries_lens):
            # ignore previously found tseries
            if idx < series_found:
                continue

            # the frame clock block matches the number of frames in a tseries
            if chunk >= ts and chunk <= ts + foxy_limit:
                # this chunk of paqio clock is a recorded tseries
                is_tseries = True
                # advance the number of tseries found so they are not 
                # detected twice
                series_found += 1
                break

        if is_tseries:
            # foxy bonus frames
            extra_frames = chunk - ts
            # mark tseries frames as real
            real_frames[counter:counter+ts] = 1
            # move the counter on by the length of the real tseries
            counter += ts
            # set foxy bonus frames to not real
            real_frames[counter:counter+extra_frames] = 0
            # move the counter along by the number of foxy bonus frames
            counter += extra_frames

        else:
            # not a tseries so just move the counter along by the chunk 
            # of paqio clock
            # this could be wrong, not sure if i've fixed the ob1 error,
            # go careful
            counter += chunk + 1

    real_idx = np.where(real_frames == 1)

    return frame_clock[real_idx]


def flu_splitter(flu, clock, t_starts, pre_frames, post_frames):
    '''Split a fluoresence matrix into trial by trial array

       flu -- fluoresence matrix [num_cells x num_frames]
       clock -- the time that each frame occured
       t_starts -- the time each frame started
       pre_frames -- the number of frames before t_start
                     to include in the trial
       post_frames --  the number of frames after t_start
                       to include in the trial

       returns 
       trial_flu -- trial by trial array 
                    [num_cells x trial frames x num_trials]
       imaging_trial -- ls of booleans of len num_trials,
                        was the trial imaged? 

       n.b. clock and t_start must have same units and
            reference frame (see rsync_aligner.py

       '''

    assert flu.shape[1] == len(clock), '{} frames in fluorescent array '\
                                       '{} frames in clock'\
                                       .format(flu.shape[1], len(clock))

    num_trials = len(t_starts)
    imaging_trial = [False] * num_trials  # was the trial imaged?

    first = True  # dumb way of building array in for loop
    for trial, t_start in enumerate(t_starts):
        # the trial occured before imaging started
        if t_start < min(clock):
            continue

        # find the first frame occuring after each trial start
        for idx, frame in enumerate(clock):
            # the idx the frame immediiately proceeds the t_start
            if frame - t_start >= 0:

                imaging_trial[trial] = True
                flu_chunk = flu[:, idx-pre_frames:idx+post_frames]

                if first:
                    trial_flu = flu_chunk
                    first = False
                else:
                    trial_flu = np.dstack((trial_flu, flu_chunk))
                break

    assert trial_flu.shape[2] == sum(imaging_trial)

    return trial_flu, imaging_trial



def flu_splitter2(flu, stim_times, frames_ms, pre_frames=10, post_frames=30):

    stim_idxs = stim_start_frame_mat(stim_times, frames_ms, debug_print=False)

    stim_idxs = stim_idxs[:, np.where((stim_idxs[0, :]-pre_frames > 0) &
                                      (stim_idxs[0, :] + post_frames 
                                      < flu.shape[1]))[0]]

    n_trials = stim_idxs.shape[1]
    n_cells = frames_ms.shape[0]

    for i, shift in enumerate(np.arange(-pre_frames, post_frames)):
        if i == 0:
            trial_idx = stim_idxs + shift
        else:
            trial_idx = np.dstack((trial_idx, stim_idxs + shift))

    tot_frames = pre_frames + post_frames
    trial_idx = trial_idx.reshape((n_cells, n_trials*tot_frames))

    flu_trials = []
    for i, idxs in enumerate(trial_idx):
        idxs = idxs[~np.isnan(idxs)].astype('int')
        flu_trials.append(flu[i, idxs])

    n_trials_valid = len(idxs)
    flu_trials = np.array(flu_trials).reshape(
        (n_cells, int(n_trials_valid/tot_frames), tot_frames))

    return flu_trials


def flu_splitter3(flu, stim_times, frames_ms, pre_frames=10, post_frames=30):

    stim_idxs = stim_start_frame_mat(stim_times, frames_ms, debug_print=False)

    # not 100% sure about this line, keep an eye
    stim_idxs[:, np.where((stim_idxs[0, :]-pre_frames <= 0) |
                          (stim_idxs[0, :] + post_frames 
                           >= flu.shape[1]))[0]] = np.nan

    n_trials = stim_idxs.shape[1]
    n_cells = frames_ms.shape[0]

    for i, shift in enumerate(np.arange(-pre_frames, post_frames)):
        if i == 0:
            trial_idx = stim_idxs + shift
        else:
            trial_idx = np.dstack((trial_idx, stim_idxs + shift))

    tot_frames = pre_frames + post_frames
    trial_idx = trial_idx.reshape((n_cells, n_trials*tot_frames))

    # flu_trials = np.repeat(np.nan, n_cells*n_trials*tot_frames)
    # flu_trials = np.reshape(flu_trials, (n_cells, n_trials, tot_frames))
    flu_trials = np.full_like(trial_idx, np.nan)
    # iterate through each cell and add trial frames
    for i, idxs in enumerate(trial_idx):

        non_nan = ~np.isnan(idxs)
        idxs = idxs[~np.isnan(idxs)].astype('int')
        flu_trials[i, non_nan] = flu[i, idxs]

    flu_trials = np.reshape(flu_trials, (n_cells, n_trials, tot_frames))
    return flu_trials


def closest_frame_before(clock, t):
    ''' returns the idx of the frame immediately preceeding 
        the time t. Frame clock must be digitised and expressed
        in the same reference frame as t
        '''
    subbed = np.array(clock) - t
    return np.where(subbed < 0, subbed, -np.inf).argmax()


def test_responsive(flu, frame_clock, stim_times, pre_frames=10, 
                    post_frames=10, offset=0):
    ''' Tests if cells in a fluoresence array are significantly responsive 
        to a stimulus

        Inputs:
        flu -- fluoresence matrix [n_cells x n_frames] likely dfof from suite2p
        frame_clock -- timing of the frames, must be digitised and in same 
                       reference frame as stim_times
        stim_times -- times that stims to test responsiveness on occured, 
                      must be digitised and in same reference frame 
                      as frame_clock
        pre_frames -- the number of frames before the stimulus occured to 
                      baseline with
        post_frames -- the number of frames after stimulus to test differnece 
                       compared
                       to baseline
        offset -- the number of frames to offset post_frames from the 
                  stimulus, so don't take into account e.g. stimulus artifact

        Returns:
        pre -- matrix of fluorescence values in the pre_frames period 
               [n_cells x n_frames]
        post -- matrix of fluorescence values in the post_frames period 
                [n_cells x n_frames]
        pvals -- vector of pvalues from the significance test [n_cells]

        '''

    n_frames = flu.shape[1]

    pre_idx = np.repeat(False, n_frames)
    post_idx = np.repeat(False, n_frames)

    # keep track of the previous stim frame to warn against overlap
    prev_frame = 0

    for i, stim_time in enumerate(stim_times):

        stim_frame = closest_frame_before(frame_clock, stim_time)

        if stim_frame-pre_frames <= 0 or stim_frame+post_frames+offset \
           >= n_frames:
            continue
        elif stim_frame - pre_frames <= prev_frame:
            print('WARNING: STA for stim number {} overlaps with the '
                  'previous stim pre and post arrays can not be '
                  'reshaped to trial by trial'.format(i))

        prev_frame = stim_frame

        pre_idx[stim_frame-pre_frames: stim_frame] = True
        post_idx[stim_frame+offset: stim_frame+post_frames+offset] = True

    pre = flu[:, pre_idx]
    post = flu[:, post_idx]

    _, pvals = stats.ttest_ind(pre, post, axis=1)

    return pre, post, pvals


def build_flu_array(run, stim_times, pre_frames=10, post_frames=50,
                    use_spks=False, is_prereward=False):

    ''' converts [n_cells x n_frames] matrix to trial by trial array
        [n_cells x n_trials x pre_frames+post_frames]

        Inputs:
        run -- BlimpImport object with attributes flu and frames_ms
        stim_times -- times of trial start stims, should be same
                      reference frame as frames_ms
        pre_frames -- number of frames before stim to include in
                      trial
        post_frames -- number of frames after stim to include 
                       in trial

        Returns:
        flu_array -- array [n_cells x n_trials x pre_frames+post_frames]

    '''

    if use_spks:
        flu = run.spks
    else:
        flu = run.flu

    if is_prereward:
        frames_ms = run.frames_ms_pre
    else:
        frames_ms = run.frames_ms

    # split flu matrix into trials based on stim time
    flu_array = flu_splitter3(flu, stim_times, frames_ms,
                              pre_frames=pre_frames, post_frames=post_frames)

    return flu_array


def averager(array_list, pre_frames=10, post_frames=50, offset=0, 
             trial_filter=None, plot=False, fs=5):

    ''' Averages ls of trial by trial fluoresence arrays and can
        visualise results

        Inputs:
        array_list -- ls of tbt fluoresence arrays
        pre_frames -- number of frames before stim to include in
                      trial
        post_frames -- number of frames after stim to include 
                       in trial
        offset -- number of frames to offset post_frames to avoid artifacts
        trial_filter -- ls of trial indexs to include
        plot -- whether to plot result
        fs -- frame rate / plane

        Returns:
        session_average -- mean array [n_sessions x pre_frames+post_frames]
        scaled_average -- same as session average but all traces start 
                          at dfof = 0
        grand_average -- average across all sessions [pre_frames + post_frames]
        cell_average -- ls with length n_sessions contains arrays
                        [n_cells x pre_frames+post_frames]

        '''

    if trial_filter:
        assert len(trial_filter) == len(array_list)
        array_list = [arr[:, filt, :]
                      for arr, filt in zip(array_list, trial_filter)]

    n_sessions = len(array_list)

    cell_average = [np.nanmean(k, 1) for k in array_list]

    session_average = np.array([np.nanmean(np.nanmean(k, 0), 0)
                               for k in array_list])

    scaled_average = np.array([session_average[i, :] - session_average[i, 0]
                               for i in range(n_sessions)])

    grand_average = np.nanmean(scaled_average, 0)

    if plot:
        x_axis = range(len(grand_average))
        plt.plot(x_axis, grand_average)
        plt.plot(x_axis[0:pre_frames],
                 grand_average[0:pre_frames], color='red')
        plt.plot(x_axis[pre_frames+offset:pre_frames+offset
                 +(post_frames-offset)], grand_average[pre_frames+offset:
                 pre_frames+offset+(post_frames-offset)], color='red')
        for s in scaled_average:
            plt.plot(x_axis, s, alpha=0.2, color='grey')

        plt.ylabel(r'$\Delta $F/F')
        plt.xlabel('Time (Seconds)')
        plt.axvline(x=pre_frames-1, ls='--', color='red')

    return session_average, scaled_average, grand_average, cell_average


def lick_binner(run):
    ''' makes new easytest binned lick variable in run object '''

    licks = run.session.times.get('lick_1')

    binned_licks = []

    for i, t_start in enumerate(run.trial_start):
        if i == len(run.trial_start) - 1:
            t_end = np.inf
        else:
            t_end = run.trial_start[i+1]

        trial_idx = np.where((licks >= t_start) & (licks <= t_end))[0]

        trial_licks = licks[trial_idx] - t_start

        binned_licks.append(trial_licks)

    run.licks = licks
    # attribute already exists called 'binned_licks' and cannot overwrite it
    run.binned_licks_easytest = binned_licks

    return run


def prepost_diff(array_list, pre_frames=10,
                 post_frames=50, offset=0, filter_list=None):

    n_sessions = len(array_list)

    if filter_list:
        array_list = [array_list[i][:, filter_list[i], :]
                      for i in range(n_sessions)]

    session_average, _, _, cell_average = averager(
        array_list, pre_frames, post_frames)

    post = np.nanmean(
                      session_average[:, pre_frames+offset:pre_frames+offset
                      +(post_frames-offset)], 1
                     )
    pre = np.nanmean(session_average[:, 0:pre_frames], 1)

    return post - pre


def raster_plot(arr, y_pos=1, color=np.random.rand(3,), alpha=1,
                marker='.', markersize=12, label=None):

    plt.plot(arr, np.ones(len(arr)) * y_pos, marker,
             color=color, alpha=alpha, markersize=markersize,
             label=label)


def adamiser(string):
    words = string.split(' ')
    n_pleases = int(len(words) / 10)
    
    for please in range(n_pleases):
        idx = randrange(len(words))
        words.insert(idx, 'please')
        
    n_caps = int(len(words) / 3)
    for cap in range(n_caps):
        idx = randrange(len(words))
        words[idx] = words[idx].upper()
        
    return ' '.join(words)


