### code for analysis of one photon photostim experiments

import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot

animal_prep = 'PS07'
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-19/'
date = '2021-01-19'
# date = data_path_base[-11:-1]

# need to update these 3 things for every trial
# trial = 't-012'  # note that %s magic command in the code below will be using these trials listed here
trials = ['t-012']  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim, post 4ap'
comments = ['10x trials of 1p stim; lots of responses throughout the FOV, and also can clearly see the 4ap onset zone with lots of standing discharges maybe 2 seizures?? one during 1p stim and 1 after end of stim trials. seizure onset difficult to pinpoint on LFP.']


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


