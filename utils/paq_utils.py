# this package contains a bunch of functions useful for reading, plotting and working with data recorded in .paq files

import sys;

sys.path.append('/home/pshah/Documents/code/Vape/')
from utils import sta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.signal as signal
from scipy import io

# from scipy.stats import pearsonr
# from scipy.signal import savgol_filter
#
# import tifffile as tf
# import pickle as pkl

import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
import plotly.graph_objects as go
import plotly.express as px


# %%

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
            chan_name = chan_name + chr(np.fromfile(fid, dtype='>f', count=1)[0])
        chan_names.append(chan_name)

    # get channel hardware lines
    hw_chans = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        hw_chan = ''
        for j in range(num_chars):
            hw_chan = hw_chan + chr(np.fromfile(fid, dtype='>f', count=1)[0])
        hw_chans.append(hw_chan)

    # get acquisition units
    units = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        unit = ''
        for j in range(num_chars):
            unit = unit + chr(np.fromfile(fid, dtype='>f', count=1)[0])
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

            # -- Prajay edit
            # change x axis ticks to seconds
            label_format = '{:,.0f}'
            labels = [item for item in ax.get_xticks()]
            for item in labels:
                labels[labels.index(item)] = int(round(item / rate))
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels([label_format.format(x) for x in labels])
            ax.set_xlabel('Time (secs)')
            # --

        plt.tight_layout()
        plt.show()

    # make pandas data frame using data in channels
    df = pd.DataFrame(data.T, columns=chan_names)

    return {"data": data,
            "chan_names": chan_names,
            "hw_chans": hw_chans,
            "units": units,
            "rate": rate,
            "num_datapoints": num_datapoints}, df


def plot_paq_interactive(paq, input_path, channels_to_plot=None):
    name = input_path[input_path.find('/20', 30) + 1:len(
        input_path)]  # note that the arguments for find here are very arbitrary, might break in some circumstances
    # Create figure

    # set layout
    layout = go.Layout(
        title="LFP - Voltage series - %s" % name,  # set title as the full name of the .paq file
        plot_bgcolor="#FFF",  # Sets background color to white
        hovermode='x',
        hoverdistance=10,
        spikedistance=1000,
        xaxis=dict(
            title="time",
            linecolor="#BCCCDC",  # Sets color of X-axis line
            showgrid=False,  # Removes X-axis grid lines
            # rangeslider=list(),

            # format spikes
            showspikes=True,
            spikethickness=2,
            spikedash='dot',
            spikecolor="#999999",
            spikemode='across'
        ),
        yaxis=dict(
            title="price",
            linecolor="#BCCCDC",  # Sets color of Y-axis line
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange=False,
            rangemode='normal'
        )
    )

    fig = go.Figure(data=go.Scatter(x=range(paq['num_datapoints']), y=paq['data'][3], line=dict(width=0.95)),
                    # downsampling data by 10,
                    layout=layout)

    # fig.update_traces(hovertemplate=None)

    # fig.add_trace(
    #     go.Scatter(x=list(t[::10]), y=list(V[0][::10]), line=dict(width=0.75)))  # downsampling data by 10

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="linear"
        )
    )

    fig.show()


def plot_paq_interactive_line(paq_df, input_path, channels_to_plot=None):
    name = input_path[input_path.find('/20', 30) + 1:len(
        input_path)]  # note that the arguments for find here are very arbitrary, might break in some circumstances
    # Create figure

    # set layout
    layout = go.Layout(
        title="LFP - Voltage series - %s" % name,  # set title as the full name of the .paq file
        plot_bgcolor="#FFF",  # Sets background color to white
        hovermode='x',
        hoverdistance=10,
        spikedistance=1000,
        xaxis=dict(
            title="time",
            linecolor="#BCCCDC",  # Sets color of X-axis line
            showgrid=False,  # Removes X-axis grid lines
            # rangeslider=list(),

            # format spikes
            showspikes=True,
            spikethickness=2,
            spikedash='dot',
            spikecolor="#999999",
            spikemode='across'
        ),
        yaxis=dict(
            title="price",
            linecolor="#BCCCDC",  # Sets color of Y-axis line
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange=False,
            rangemode='normal'
        )
    )

    fig = px.line(paq_df, x=paq_df.index[:1000], y="voltage", title='Life expectancy in Canada')
    fig.show()

    # fig.update_traces(hovertemplate=None)

    # fig.add_trace(
    #     go.Scatter(x=list(t[::10]), y=list(V[0][::10]), line=dict(width=0.75)))  # downsampling data by 10

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="linear"
        )
    )

    fig.show()


def frames_discard(paq, input_array, total_frames, discard_all=False):
    '''
    calculate which 2P imaging frames to discard (or use as bad frames input into suite2p) based on the bad frames
    identified by manually inspecting the paq files in EphysViewer.m
    :param paq: paq file
    :param input_array: .m file path to read that contains the timevalues for signal to remove
    :param total_frames: the number of frames in the TIFF file of the actual 2p imaging recording
    :param discard_all: bool; if True, then add all 2p imaging frames from this paq file as bad frames to discard
    :return: array that contains the indices of bad frames (in format ready to input into suite2p processing)
    '''

    frame_times = sta.threshold_detect(paq['data'][0], 1)
    frame_times = frame_times[
                  0:total_frames]  # this is necessary as there are more TTL triggers in the paq file than actual frames (which are all at the end)

    all_btwn_paired_frames = []
    paired_frames_first = []
    paired_frames_last = []
    if input_array is not None:
        print('\nadding seizure frames loaded up from: ', input_array)
        measurements = io.loadmat(input_array)
        for set_ in range(len(measurements['PairedMeasures'])):
            # calculate the sample value for begin and end of the set
            begin = int(measurements['PairedMeasures'][set_][3][0][0] * paq['rate'])
            end = int(measurements['PairedMeasures'][set_][5][0][0] * paq['rate'])
            frames_ = list(np.where(np.logical_and(frame_times >= begin, frame_times <= end))[0])
            all_btwn_paired_frames.append(frames_)
            paired_frames_first.append(frames_[0])
            paired_frames_last.append(frames_[-1])

        all_btwn_paired_frames = [item for x in all_btwn_paired_frames for item in x]

    if discard_all and input_array is None:
        frames_to_discard = list(range(len(frame_times)))
        return frames_to_discard
    elif not discard_all and input_array is not None:
        frames_to_discard = all_btwn_paired_frames
        return frames_to_discard, all_btwn_paired_frames, paired_frames_first, paired_frames_last
    elif discard_all and input_array is not None:
        frames_to_discard = list(range(len(frame_times)))
        return frames_to_discard, all_btwn_paired_frames, paired_frames_first, paired_frames_last
    else:
        raise ReferenceError('something wrong....No frames selected for discarding')



# #%%
# # path to paq file
# input_path = "/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_RL108_001.paq"  # server path
#
# # input_path = '/Users/prajayshah/Documents/data-to-process/2020-12-18/2020-12-18_RL108_001.paq'  # local path
# paq, paq_df = paq_read(input_path, plot=False)
#
#
# voltage = paq['data'][3]

# %%

def clean_lfp_signal(paq, input_array: str, chan_name: str = 'voltage', plot=False):
    '''
    the idea is to use EphysViewer.m in matlab to view .paq files and then from there export an excel file that
    will contain paired sets of timevalues that are the start and end of the signal to clean up.

    note: clean up here doesn't mean to remove the signal values but instead to set them to a constant value that
    will connect the end of the pre-clean signal and post-clean signal, so that it returns a continuous signal of the
    same length as the original signal.

    :param plot: to make plot of the fixed up LFP signal or not
    :param chan_name: channel name in paq file that contains the LFP series
    :param paq: paq file containing the LFP series
    :param input_array: path to .mat file to read that contains the timevalues for signal to remove
    :return: cleaned up LFP signal
    '''

    measurements = io.loadmat(input_array)

    lfp_series = paq['data'][paq['chan_names'].index(chan_name)]
    for set in range(len(measurements['PairedMeasures'])):
        # calculate the sample value for begin and end of the set
        begin = int(measurements['PairedMeasures'][set][3][0][0] * paq['rate'])
        end = int(measurements['PairedMeasures'][set][5][0][0] * paq['rate'])

        lfp_series[begin:end] = lfp_series[begin - 1]  # set the signal value to equal the voltage value just before

    # detrend the LFP signal
    signal.detrend(lfp_series)

    if plot:
        plt.figure(figsize=[40, 4])
        plt.plot(lfp_series, linewidth=0.2)
        plt.suptitle('LFP voltage')
        plt.show()

    return lfp_series
    # replot the LFP signal

# input_array = "/Users/prajayshah/Documents/data-to-process/2020-12-18/paired_measurements/2020-12-18_RL108_001.mat"
# volt_clean = clean_lfp_signal(paq, input_array=input_array, plot=True)
#
#
#
# #%%
#
# volt_ = voltage[int(0.5e7):int(1.5e7)]
#
# plt.figure(figsize=[40,2])
# plt.plot(volt_, linewidth=0.2); plt.suptitle('LFP voltage'); plt.show()
#
# #%%
# volt_ = voltage[int(0.5e7):int(1.5e7)]
# v_detrended = signal.detrend(volt_)
# plt.figure(figsize=[40,4])
# plt.plot(volt_, linewidth=0.2, color='red')
# plt.plot(v_detrended, linewidth=0.2); plt.suptitle('LFP voltage'); plt.ylim([-2, 2]), plt.show()
#
# #%%
# name = input("enter a number:")
# print('herelo', name)
