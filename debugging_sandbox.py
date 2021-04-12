# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')



# %run ./setup_notebook.ipynb
# print(sys.path)
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
from utils import funcs_pj as pj

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
import utils.funcs_pj as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw
import tifffile as tf

# original = '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/AllOptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file021_chan0.tif'
# recreated = '/home/pshah/mnt/qnap/Analysis/2021-01-10/2021-01-10_t-008/reg_tiff_t-008.tif'
#
# with tf.TiffFile(original, multifile=False) as input_tif:
#     data_original = input_tif.asarray()
#     print('shape of tiff: ', data_original.shape)
#
# with tf.TiffFile(recreated, multifile=False) as input_tif:
#     data_recreated = input_tif.asarray()
#     print('shape of tiff: ', data_recreated.shape)
#     data_recreated1 = data_recreated[0]
#

sorted_paths = ['/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/AllOptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file021_chan0.tif',
                '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/AllOptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file022_chan0.tif',
                '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/AllOptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file023_chan0.tif',
                '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/AllOptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file024_chan0.tif',
                '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/AllOptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file025_chan0.tif']

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
                for frame in data:
                    tif.write(frame, contiguous=True)

                # tif.save(data[0])
            msg = ' -- Writing tiff: ' + str(i + 1) + ' out of ' + str(num_tiffs)
            print(msg, end='\r')
            # tif.save(data)

make_tiff_stack(sorted_paths=sorted_paths, save_as='/home/pshah/mnt/qnap/Analysis/2021-01-10/2021-01-10_t-008/reg_tiff_t-008.tif')

# series0 = np.random.randint(0, 255, (32, 32, 3), 'uint8')
# series1 = np.random.randint(0, 1023, (4, 256, 256), 'uint16')
series0 = np.random.randint(0, 1023, (4, 256, 256), 'uint16')
series1 = np.random.randint(0, 1023, (4, 256, 256), 'uint16')
tf.imwrite('temp.tif', series0, photometric='minisblack')
tf.imwrite('temp.tif', series1, append=True)

img = tf.imread('temp.tif')