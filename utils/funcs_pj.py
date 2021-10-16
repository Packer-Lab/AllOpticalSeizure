import os
import sys

import numpy
import numpy as np
import pandas as pd
import tifffile
from scipy import stats, ndimage, io
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.decomposition import PCA
import tifffile as tf
import math


# plotting settings
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

############### STATS/DATA ANALYSIS FUNCTIONS ##########################################################################
# calculate correlation across all cells


def corrcoef_array(array):
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

def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y


# calculate distance between 2 points on a cartesian plane
def calc_distance_2points(p1: tuple, p2: tuple):
    """
    uses the hypothenus method to calculate the straight line distance between two given points on a 2d cartesian plane.
    :param p1: point 1
    :param p2: point 2
    :return:
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# find percentile of a value within an array
def find_percentile(d, threshold):
    return sum(np.abs(d) < threshold) / float(len(d)) * 100

# random func for rotating images and calculating the image intensity along one axis of the image
def rotate_img_avg(input_img, angle):
    """this function will be used to rotate the input_img (ideally will be the avg seizure image) at the given angle.
    The function also will return the 1 x n length average across non-zero values along the x axis.

    :param input_img: ndarray comprising the image
    :param angle: the angle to rotate the image with (in degrees), +ve = counter-clockwise
    """
    full_img_rot = ndimage.rotate(input_img, angle, reshape=True)

    return full_img_rot


# PCA decomposition(/compression) of an image
def pca_decomp_image(input_img, components: int = 3, plot_quant: bool = False):
    """
    the method for PCA based decomposition/compression of an image, and also (optional) quantification of the resulting
    image across the x axis

    :param input_img: ndarray; input image
    :param components: int; # of principle components to use for the PCA decomposition (compression) of the input_img
    :param plot_quant: bool; plot quantification of the average along x-axis of the image
    :return: ndarray; compressed image, imshow plots of the original and PCA compressed images, as well as plots of average across the x-axis
    """

    print("Extracting the top %d eigendimensions from image" % components)
    pca = PCA(components)
    img_transformed = pca.fit_transform(input_img)
    img_compressed = pca.inverse_transform(img_transformed)

    if plot_quant:
        # quantify the input image
        fig = plt.figure(figsize=(15, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.imshow(input_img, cmap='gray')
        ax2.imshow(img_compressed, cmap='gray')

        img_t = input_img.T
        avg = np.zeros([img_t.shape[0], 1])
        for i in range(len(img_t)):
            x = img_t[i][img_t[i] > 0]
            if len(x) > 0:
                avg[i] = x.mean()
            else:
                avg[i] = 0

        ax3.plot(avg)
        ax3.set_xlim(20, len(img_t) - 20)
        ax3.set_title('average plot quantification of the input img', wrap=True)
        plt.show()

        # quantify the PC reconstructed image
        fig = plt.figure(figsize=(15, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.imshow(input_img, cmap='gray')
        ax2.imshow(img_compressed, cmap='gray')

        img_compressed = img_compressed.T
        avg = np.zeros([img_compressed.shape[0], 1])
        for i in range(len(img_compressed)):
            x = img_compressed[i][img_compressed[i] > 0]
            if len(x) > 0:
                avg[i] = x.mean()
            else:
                avg[i] = 0

        ax3.plot(avg)
        ax3.set_xlim(20, len(img_compressed.T) - 20)
        ax3.title.set_text('average plot quantification of the PCA compressed img - %s dimensions' % components)

        plt.show()

    return img_compressed


# grouped average / smoothing of a 1dim array (basically the same as grouped average on imageJ)
def smooth_signal(signal, w):
    return np.convolve(signal, np.ones(w), 'valid') / w


############### CALCIUM IMAGING RELATED STUFF ##########################################################################
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


# normalize dFF for 1dim array
def dff(flu, baseline=None):
    """delta F over F ratio (not % dFF )"""
    if baseline is not None:
        flu_dff = (flu - baseline) / baseline
    else:
        flu_mean = np.mean(flu, 1)
        flu_dff = (flu - flu_mean) / flu_mean

    return flu_dff


# simple ZProfile function for any sized square in the frame (equivalent to ZProfile function in Fiji)
def ZProfile(movie, area_center_coords: tuple = None, area_size: int = -1, plot_trace: bool = True,
             plot_image: bool = True, plot_frame: int = 1, vasc_image: np.array = None, **kwargs):
    """
    from Sarah Armstrong

    Plot a z-profile of a movie, averaged over space inside a square area

    movie = can be np.array of the TIFF stack or a tiff path from which it is read in
    area_center_coords = coordinates of pixel at center of box (x,y)
    area_size = int, length and width of the square in pixels
    plot_frame = which movie frame to take as a reference to plot the area boundaries on
    vasc_image = optionally include a vasculature image tif of the correct dimensions to plot the coordinates on.
    """

    if type(movie) is str:
        movie = tf.imread(movie)
    print('plotting zprofile for TIFF of shape: ', movie.shape)

    # assume 15fps for 1024x1024 movies and 30fps imaging for 512x512 movies
    if movie.shape[1] == 1024:
        img_fps = 15
    elif movie.shape[1] == 512:
        img_fps = 30
    else:
        img_fps = None

    assert area_size <= movie.shape[1] and area_size <= movie.shape[2], "area_size must be smaller than the image"
    if area_size == -1:  # this parameter used to plot whole FOV area
        area_size = movie.shape[1]
        area_center_coords = (movie.shape[1]/2, movie.shape[2]/2)
    assert area_size % 2 == 0, "pls give an even area size"

    x = area_center_coords[0]
    y = area_center_coords[1]
    x1 = int(x - 1 / 2 * area_size)
    x2 = int(x + 1 / 2 * area_size)
    y1 = int(y - 1 / 2 * area_size)
    y2 = int(y + 1 / 2 * area_size)
    smol_movie = movie[:, y1:y2, x1:x2]
    smol_mean = np.nanmean(smol_movie, axis=(1, 2))
    print('|- Output shape for z profile: ', smol_mean.shape)

    if plot_image:
        f, ax1 = plt.subplots()
        ref_frame = movie[plot_frame, :, :]
        if vasc_image is not None:
            assert vasc_image.shape == movie.shape[1:], 'vasculature image has incompatible dimensions'
            ax1.imshow(vasc_image, cmap="binary_r")
        else:
            ax1.imshow(ref_frame, cmap="binary_r")

        rect1 = patches.Rectangle(
            (x1, y1), area_size, area_size, linewidth=1.5, edgecolor='r', facecolor="none")

        ax1.add_patch(rect1)
        ax1.set_title("Z-profile area")
        plt.show()

    if plot_trace:
        if 'figsize' in kwargs:
            figsize = kwargs['figsize']
        else:
            figsize = [10, 4]
        fig, ax2 = plt.subplots(figsize=figsize)
        if img_fps is not None:
            ax2.plot(np.arange(smol_mean.shape[0])/img_fps, smol_mean, linewidth=0.5, color='black')
            ax2.set_xlabel('Time (sec)')
        else:
            ax2.plot(smol_mean, linewidth=0.5, color='black')
            ax2.set_xlabel('frames')
        ax2.set_ylabel('Flu (a.u.)')
        if 'title' in kwargs:
            ax2.set_title(kwargs['title'])
        plt.show()

    return smol_mean


def SaveDownsampledTiff(tiff_path: str = None, stack: np.array = None, group_by: int = 4, save_as: str = None, plot_zprofile: bool = True):
    """
    Create and save a downsampled version of the original tiff file. Original tiff file can be given as a numpy array stack
    or a str path to the tiff.

    :param tiff_path: path to the tiff to downsample
    :param stack: numpy array stack of the tiff file already read in
    :param group_by: specified interval for grouped averaging of the TIFF
    :param save_as: path to save the downsampled tiff to, if none provided it will save to the same directory as the provided tiff_path
    :param plot_zprofile: if True, plot the zaxis profile using the full TIFF stack provided.
    :return: numpy array containing the downsampled TIFF stack
    """
    print('downsampling of tiff stack...')

    if save_as is None:
        assert tiff_path is not None, "please provide a save path to save_as"
        save_as = tiff_path[:-4] + '_downsampled.tif'

    if stack is None:
        # open tiff file
        print('|- working on... %s' % tiff_path)
        stack = tf.imread(tiff_path)

    resolution = stack.shape[1]

    # plot zprofile of full TIFF stack
    if plot_zprofile:
        ZProfile(movie=stack, plot_image=True, title=tiff_path)

    # downsample to 8-bit
    stack8 = np.full_like(stack, fill_value=0)
    for frame in np.arange(stack.shape[0]):
        stack8[frame] = convert_to_8bit(stack[frame], 0, 255)

    # stack8 = stack

    # grouped average by specified interval
    num_frames = stack8.shape[0] // group_by
    # avgd_stack = np.empty((num_frames, resolution, resolution), dtype='uint16')
    avgd_stack = np.empty((num_frames, resolution, resolution), dtype='uint8')
    frame_count = np.arange(0, stack8.shape[0], group_by)
    for i in np.arange(num_frames):
        frame = frame_count[i]
        avgd_stack[i] = np.mean(stack8[frame:frame + group_by], axis=0)

    avgd_stack = avgd_stack.astype(np.uint8)

    # bin down to 512 x 512 resolution if higher resolution
    shape = np.shape(avgd_stack)
    if shape[1] != 512:
        # input_size = avgd_stack.shape[1]
        # output_size = 512
        # bin_size = input_size // output_size
        # final_stack = avgd_stack.reshape((shape[0], output_size, bin_size,
        #                                   output_size, bin_size)).mean(4).mean(2)
        final_stack = avgd_stack
    else:
        final_stack = avgd_stack

    # write output
    print("\nsaving %s to... %s" % (final_stack.shape, save_as))
    tf.imwrite(save_as, final_stack, photometric='minisblack')

    return final_stack


def subselect_tiff(tiff_path: str = None, tiff_stack: np.array = None, select_frames: tuple = (0, 0), save_as: str = None):
    if tiff_stack is None:
        # open tiff file
        print('running subselecting tiffs')
        print('|- working on... %s' % tiff_path)
        tiff_stack = tf.imread(tiff_path)

    stack_cropped = tiff_stack[select_frames[0]:select_frames[1]]

    # stack8 = convert_to_8bit(stack_cropped)

    if save_as is not None:
        tf.imwrite(save_as, stack_cropped, photometric='minisblack')

    return stack_cropped


def make_tiff_stack(sorted_paths: list, save_as: str):
    """
    read in a bunch of tiffs and stack them together, and save the output as the save_as

    :param sorted_paths: list of string paths for tiffs to stack
    :param save_as: .tif file path to where the tif should be saved
    """

    num_tiffs = len(sorted_paths)
    print('working on tifs to stack: ', num_tiffs)

    with tf.TiffWriter(save_as, bigtiff=True) as tif:
        for i, tif_ in enumerate(sorted_paths):
            with tf.TiffFile(tif_, multifile=True) as input_tif:
                data = input_tif.asarray()
            msg = ' -- Writing tiff: ' + str(i + 1) + ' out of ' + str(num_tiffs)
            print(msg, end='\r')
            tif.save(data)


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

############### GENERALLY USEFUL FUNCTIONS #############################################################################

# reporting sizes of variables
def _sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def print_size_of(var):
    print(_sizeof_fmt(sys.getsizeof(var)))


def print_size_vars():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, _sizeof_fmt(size)))


# finding paths to files with a certain extension
def path_finder(umbrella, *args, is_folder=False):
    '''
    returns the path to the single item in the umbrella folder
    containing the string names in each arg
    is_folder = False if args is list of files
    is_folder = True if  args is list of folders
    '''
    # list of bools, has the function found each argument?
    # ensures two folders / files are not found
    found = [False] * len(args)
    # the paths to the args
    paths = [None] * len(args)

    if is_folder:
        for root, dirs, files in os.walk(umbrella):
            for folder in dirs:
                for i, arg in enumerate(args):
                    if arg in folder:
                        assert not found[i], 'found at least two paths for {},' \
                                             'search {} to find conflicts' \
                            .format(arg, umbrella)
                        paths[i] = os.path.join(root, folder)
                        found[i] = True

    elif not is_folder:
        for root, dirs, files in os.walk(umbrella):
            for file in files:
                for i, arg in enumerate(args):
                    if arg in file:
                        assert not found[i], 'found at least two paths for {},' \
                                             'search {} to find conflicts' \
                            .format(arg, umbrella)
                        paths[i] = os.path.join(root, file)
                        found[i] = True

    print(paths)
    for i, arg in enumerate(args):
        if not found[i]:
            raise ValueError('could not find path to {}'.format(arg))

    return paths


# progress bar
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# find the closest value in a list to the given input
def findClosest(list, input):
    subtract = list - input
    positive_values = abs(subtract)
    # closest_value = min(positive_values) + input
    index = np.where(positive_values == min(positive_values))[0][0]
    closest_value = list[index]

    return closest_value, index


############### PLOTTING FUNCTIONS #####################################################################################
# custom colorbar for heatmaps
from matplotlib.colors import LinearSegmentedColormap


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)


# generate an array of random colors
def _get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]

def _color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])

def _generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = _get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([_color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def make_random_color_array(n_colors):
    """
    Generates a list of random colors for an input number of colors required.

    :param n_colors: # of colors required
    :return: list of colors in RGB
    """
    colors = []
    for i in range(0, n_colors):
        colors.append(_generate_new_color(colors, pastel_factor=0.2))
    return colors


# plotting function for plotting a bar graph with the individual data points shown as well
def plot_bar_with_points(data, title='', x_tick_labels=[], legend_labels: list = [], points: bool = True, bar: bool = True, colors: list = ['black'], ylims=None, xlims=True, text_list=None,
                         x_label=None, y_label=None, alpha=0.2, savepath=None, expand_size_x=1, expand_size_y=1, shrink_text: float = 1, show_legend=False,
                         paired=False, **kwargs):
    """
    all purpose function for plotting a bar graph of multiple categories with the option of individual datapoints shown
    as well. The individual datapoints are drawn by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph. The individual points can also be paired in which case they will be centered. The bar can also be turned off.

    :param data: list; provide data from each category as a list and then group all into one list
    :param title: str; title of the graph
    :param x_tick_labels: labels to use for categories on x axis
    :param legend_labels:
    :param points: bool; if True plot individual data points for each category in data using scatter function
    :param bar: bool, if True plot the bar, if False plot only the mean line
    :param colors: colors (by category) to use for each x group
    :param ylims: tuple; y axis limits
    :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
    :param x_label: x axis label
    :param y_label: y axis label
    :param text_list: list of text to add to each category of data on the plot
    :param text_shift: float; number between 0.5 to 1 used to adjust precise positioning of the text in text_list
    :param alpha: transparency of the individual points when plotted in the scatter
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :param expand_size_x: factor to use for expanding figure size
    :param expand_size_y: factor to use for expanding figure size
    :param paired: bool, if True then draw lines between data points of the same index location in each respective list in the data
    :return: matplotlib plot
    """

    # collect some info about data to plot
    w = 0.3  # mean bar width
    x = list(range(len(data)))
    y = data
    if len(colors) != len(x):
        colors = colors * len(x)


    # initialize plot
    if 'fig' in kwargs.keys():
        f = kwargs['fig']
        ax = kwargs['ax']
    else:
        f, ax = plt.subplots(figsize=((5 * len(x) / 2) * expand_size_x, 3 * expand_size_y))

    if paired:
        assert len(x) > 1

    # start making plot
    if not bar:
        for i in x:
            ax.plot(np.linspace(x[i] * w * 2.5 - w / 2, x[i] * w * 2.5 + w / 2, 3), [np.mean(y[i])] * 3, color='black')
        lw = 0,
        edgecolor = None
    else:
        if not kwargs['edgecolor']:
            edgecolor = 'black',
            lw = 1
        else:
            edgecolor = kwargs['edgecolor'],
            lw = 1

    # plot bar graph, or if no bar (when lw = 0 from above) then use it to plot the error bars
    ax.bar([x * w * 2.5 for x in x],
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi, ddof=1) for yi in y],  # error bars
           capsize=4.5,  # error bar cap width in points
           width=w,  # bar width
           linewidth=lw,  # width of the bar edges
           edgecolor=edgecolor,
           color=(0, 0, 0, 0),  # face edgecolor transparent
           zorder=2
           )
    ax.set_xticks([x * w * 2.5 for x in x])
    ax.set_xticklabels(x_tick_labels)

    if xlims:
        ax.set_xlim([(x[0] * w * 2) - w * 1.20, (x[-1] * w * 2.5) + w * 1.20])
    elif len(x) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims_ = [-1, 1]
        ax.set_xlim(xlims_)

    if len(legend_labels) == 0:
        if len(x_tick_labels) == 0:
            x_tick_labels = [None] * len(x)
        legend_labels = x_tick_labels

    if points:
        if not paired:
            for i in x:
                # distribute scatter randomly across whole width of bar
                ax.scatter(x[i] * w * 2.5 + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i], alpha=alpha, label=legend_labels[i])

        else:  # connect lines to the paired scatter points in the list
            if len(x) > 0:
                for i in x:
                    # plot points  # dont scatter location of points if plotting paired lines
                    ax.scatter([x[i] * w * 2.5] * len(y[i]), y[i], color=colors[i], alpha=0.5,
                               label=legend_labels[i], zorder=3)
                for i in x[:-1]:
                    for point_idx in range(len(y[i])):  # draw the lines connecting pairs of data
                        ax.plot([x[i] * w * 2.5 + 0.058, x[i+1] * w * 2.5 - 0.048], [y[i][point_idx], y[i+1][point_idx]], color='black', zorder=2, alpha=alpha)

                # for point_idx in range(len(y[i])):  # slight design difference, with straight line going straight through the scatter points
                #     ax.plot([x * w * 2.5 for x in x],
                #             [y[i][point_idx] for i in x], color='black', zorder=0, alpha=alpha)

            else:
                ReferenceError('cannot do paired scatter plotting with only one data category')

    if ylims:
        ax.set_ylim(ylims)
    elif len(x) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=10)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel(x_label, fontsize=8*shrink_text)
    ax.set_ylabel(y_label, fontsize=12*shrink_text)
    if savepath:
        plt.savefig(savepath)
    if len(x) > 1:
        plt.xticks(rotation=45)
        # plt.setp(ax.get_xticklabels(), rotation=45)

    # add text to the figure if given:
    if text_list:
        assert len(x) == len(text_list), 'please provide text_list of same len() as data'
        if 'text_shift' in kwargs.keys():
            text_shift = kwargs['text_shift']
        else:
            text_shift = 0.8
        for i in x:
            ax.text(x[i] * w * 2.5 - text_shift*w / 2, np.max(y[i]) * 0.75, text_list[i]),

    if len(legend_labels) > 1:
        if show_legend:
            ax.legend(bbox_to_anchor=(1.01, 0.90), fontsize=8*shrink_text)

    # add title
    if 'fig' not in kwargs.keys():
        ax.set_title((title), horizontalalignment='center', verticalalignment='top', pad=25,
                     fontsize=12*shrink_text, wrap=True)
    else:
        ax.title.set_text((title))

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            # Tweak spacing to prevent clipping of ylabel
            # f.tight_layout()
            f.show()
        else:
            return f, ax
    else:
        # Tweak spacing to prevent clipping of ylabel
        # f.tight_layout()
        f.show()

# histogram density plot with gaussian best fit line
def plot_hist_density(data, colors: list = None, fill_color: list = None, legend_labels: list = [None], num_bins=10, **kwargs):

    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=[20, 3])

    if legend_labels is None:
        legend_labels = [None] * len(data)
    if colors is None:
        colors = ['black'] * len(data)
    if len(data) == 1:
        fill_color = ['steelblue']
    else:
        assert len(data) == len(colors)
        assert len(data) == len(fill_color), print('please provide a fill color for each dataset')

    for i in range(len(data)):
        # the histogram of the data
        n, bins, patches = ax.hist(data[i], num_bins, density=1, alpha=0.4, color=fill_color[i], label=legend_labels[i])  # histogram hidden currently

        # add a 'best fit' line
        mu = np.mean(data[i])  # mean of distribution
        sigma = np.std(data[i])  # standard deviation of distribution

        x = np.linspace(bins[0], bins[-1], num_bins*5)
        y1 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
              np.exp(-0.5 * (1 / sigma * (x - mu))**2))
        ax.plot(x, y1, linewidth=2, c=colors[i], zorder=2)
        ax.fill_between(x, y1, color=fill_color[i], zorder=2, alpha=0.3)
        if 'x_label' in kwargs:
            ax.set_xlabel(kwargs['x_label'])
        if 'y_label' in kwargs:
            ax.set_ylabel(kwargs['y_label'])
        else:
            ax.set_ylabel('Probability density')

    # set x limits
    if 'x_lim' in kwargs:
        ax.set_xlim(kwargs['x_lim'])

    ax.legend()

    # setting shrinking factor for font size for title
    if 'shrink_text' in kwargs.keys():
        shrink_text = kwargs['shrink_text']
    else:
        shrink_text = 1

    # add title
    if 'fig' not in kwargs.keys():
        if 'title' in kwargs:
            if len(data) == 1:
                ax.set_title(kwargs['title'] + r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)), wrap=True, fontsize=12*shrink_text)
            else:
                ax.set_title(kwargs['title'], wrap=True, fontsize=12*shrink_text)
        else:
            if len(data) == 1:
                ax.set_title(r'Histogram: $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)))
            else:
                ax.set_title('Histogram density plot')


    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            # Tweak spacing to prevent clipping of ylabel
            fig.tight_layout()
            fig.show()
        else:
            pass
    else:
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        fig.show()

    if 'fig' in kwargs.keys():
        # adding text because adding title doesn't seem to want to work when piping subplots
        ax.set_title(kwargs['title'] + r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)), wrap=True, fontsize=12*shrink_text)
        # ax.text(0.98, 0.97, kwargs['title'] + r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)),
        #         verticalalignment='top', horizontalalignment='right',
        #         transform=ax.transAxes, fontweight='bold',
        #         color='black', fontsize=10 * shrink_text)
        return fig, ax



# imshow gray plot for a single frame tiff
def plot_single_tiff(tiff_path: str, title: str = None, frame_num: int = 0):
    """
    plots an image of a single tiff frame after reading using tifffile.
    :param tiff_path: path to the tiff file
    :param title: give a string to use as title (optional)
    :return: imshow plot
    """
    stack = tf.imread(tiff_path, key=frame_num)
    plt.imshow(stack, cmap='gray')
    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle('frame num: %s' % frame_num)
    plt.show()
    return stack

# read matlab array
def load_matlab_array(path):
    """
    Returns a matlab array read in from the path given in path.
    :param path: path to the matlab output file ending in .mat
    :return: array
    """
    return io.loadmat(path)
#######

