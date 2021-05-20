### code for analysis of one photon photostim experiments

import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot

animal_prep = 'PS09'
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-24/PS09/'
date = '2021-01-24'
# date = data_path_base[-11:-1]

# need to update these 3 things for every trial
# trial = 't-012'  # note that %s magic command in the code below will be using these trials listed here
trials = ['t-013', 't-015']  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim, post 4ap'
comments = ['2 seizures, start of trial mid sz; 2nd sz seems to be very clearly induced by the 1p stim; 20x 1p stims',
            '1 seziure, seems to be spurred on by 1p stim; 20x 1p stim; sequential 1p stims are able to induce spreading activity']


for trial in trials:
    metainfo = {
        'animal prep.': animal_prep,
        'trial': trial,
        'date': date,
        'exptype': exp_type,
        'data_path_base': data_path_base,
        'comments': comments[trials.index(trial)]
    }
    expobj = aoutils.OnePhotonStim(data_path_base, date, animal_prep, trial, metainfo)


