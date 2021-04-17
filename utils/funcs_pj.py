import os
import sys

import numpy as np
import pandas as pd
from scipy import stats, ndimage, io
import itertools
import matplotlib.pyplot as plt
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

def calc_distance_2points(p1: tuple, p2: tuple):
    """
    uses the hypothenus method to calculate the straight line distance between two given points on a 2d cartesian plane.
    :param p1: point 1
    :param p2: point 2
    :return:
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# making a new class inherited from alloptical for post4ap functions and elements and variables and attributes
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

# read matlab array
def load_matlab_array(path):
    """
    Returns a matlab array read in from the path given in path.
    :param path: path to the matlab output file ending in .mat
    :return: array
    """
    return io.loadmat(path)

# useful for returning indexes when a
def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    frames = np.where(thresh_signal)
    return frames[0]


# normalize Ca values
def dff(flu, baseline=None):
    if baseline is not None:
        flu_dff = (flu - baseline) / baseline
    else:
        flu_mean = np.mean(flu, 1)
        flu_dff = (flu - flu_mean) / flu_mean

    return flu_dff


# simple plot of the location of the given cell(s) against a black FOV
def plot_cell_loc(expobj, cells: list, color: str = '#EDEDED', title=None, show: bool = True, background: np.array = None):
    """
    plots an image of the FOV to show the locations of cells given in cells list.
    :param background: either 2dim numpy array to use as the backsplash or None (where black backsplash will be created)
    :param expobj: alloptical or 2p imaging object
    :param color: str to specify color of the scatter plot for cells
    :param cells: list of cells to plot
    :param title: str title for plot
    :param show: if True, show the plot at the end of the function
    """

    if background is None:
        black = np.zeros((expobj.frame_x, expobj.frame_y), dtype='uint16')
        plt.imshow(black)
    else:
        plt.imshow(background)


    for cell in cells:
        y, x = expobj.stat[expobj.cell_id.index(cell)]['med']
        if hasattr(expobj, 's2p_cell_targets'):
            if cell in expobj.s2p_cell_targets:
                color_ = '#F02A71'
            else:
                color_ = 'none'
        else:
            color_ = 'none'
        plt.scatter(x=x, y=y, edgecolors=color, facecolors=color_, linewidths=0.8)

    if background is None:
        plt.xlim(0, expobj.frame_x)
        plt.ylim(0, expobj.frame_y)

    if title is not None:
        plt.suptitle(title)

    if show:
        plt.show()


############### GENERALLY USEFUL FUNCTIONS #############################################################################

# reporting sizes of variables
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_size_of(var):
    print(sizeof_fmt(sys.getsizeof(var)))

def print_size_vars():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


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
    closest_value = min(positive_values) + input
    index = np.where(positive_values == min(positive_values))[0][0]

    return closest_value, index


############### PLOTTING FUNCTIONS #####################################################################################
# custom colorbar for heatmaps
from matplotlib.colors import LinearSegmentedColormap, ColorConverter


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


# generate an array of random colors, based on previous
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
def make_random_color_array(array_of_ids):
    "array_of_ids: an array containing neuron IDs (the length of this array will be the number of colors returned)"
    colors = []
    for i in range(0, len(array_of_ids)):
        colors.append(_generate_new_color(colors, pastel_factor=0.2))
    return colors


# plotting function for plotting a bar graph with the individual data points shown as well
def bar_with_points(data, title='', x_tick_labels=[], points=True, bar=True, colors=['black'], ylims=None, xlims=None,
                    x_label=None, y_label=None, alpha=0.2, savepath=None, expand_size_x=1, expand_size_y=1):
    """
    general purpose function for plotting a bar graph of multiple categories with the individual datapoints shown
    as well. The latter is achieved by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph.

    :param expand_size: factor to use for expanding figure size
    :param data: list; provide data from each category as a list and then group all into one list
    :param title: str; title of the graph
    :param x_tick_labels: labels to use for categories on x axis
    :param points: bool; if True plot individual data points for each category in data using scatter function
    :param bar: bool, if True plot the bar, if False plot only the mean line
    :param colors: colors (by category) to use for each x group
    :param ylims: tuple; y axis limits
    :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
    :param x_label: x axis label
    :param y_label: y axis label
    :param alpha: transparency of the individual points when plotted in the scatter
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :return: matplotlib plot
    """

    w = 0.3  # bar width
    x = list(range(len(data)))
    y = data
    if len(colors) != len(x):
        colors = colors * len(x)

    # plt.figure(figsize=(2, 10))
    fig, ax = plt.subplots(figsize=((2 * len(x) / 2) * expand_size_x, 3 * expand_size_y))
    if not bar:
        for i in x:
            # ax.plot(np.linspace(x[i] - w / 2, x[i] + w / 2, 3), [np.mean(yi) for yi in y] * 3, color=colors[i])
            ax.plot(np.linspace(x[i] * w * 2 - w / 2, x[i] * w * 2 + w / 2, 3), [np.mean(y[i])] * 3, color='black')
        lw = 0,
        edgecolor = None
    else:
        edgecolor = 'black',
        lw = 1

    # plot bar graph, or if no bar (when lw = 0 from above) then use it to plot the error bars
    ax.bar([x * w * 2 for x in x],
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi) for yi in y],  # error bars
           capsize=4.5,  # error bar cap width in points
           width=w,  # bar width
           linewidth=lw,  # width of the bar edges
           # tick_label=x_tick_labels,
           edgecolor=edgecolor,
           color=(0, 0, 0, 0),  # face color transparent
           )
    ax.set_xticks([x * w * 2 for x in x])
    ax.set_xticklabels(x_tick_labels)

    if xlims:
        ax.set_xlim([xlims[0] - 2 * w, xlims[1] + 2 * w])
    elif len(x) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims = [-1, 1]
        ax.set_xlim(xlims)

    if points:
        for i in x:
            # distribute scatter randomly across whole width of bar
            ax.scatter(x[i] * w * 2 + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i], alpha=alpha)

    if ylims:
        ax.set_ylim(ylims)
    elif len(x) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    ax.set_title((title), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savepath:
        plt.savefig(savepath)
    if len(x) > 1:
        plt.xticks(rotation=45)
        # plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()


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

#######
