### code for analysis of one photon photostim experiments

import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot

animal_prep = 'PS16'
data_path_base = '/home/pshah/mnt/qnap/Data/2021-02-01/'
# date = '2021-02-02'
date = data_path_base[-11:-1]

# need to update these 3 things for every trial
# trial = 't-012'  # note that %s magic command in the code below will be using these trials listed here
trials = ['t-009', 't-010']  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim, post 4ap'
comments = '20x 1p stims'


for trial in trials:
    metainfo = {
        'animal prep.': animal_prep,
        'trial': trial,
        'date': date,
        'exptype': exp_type,
        'data_path_base': data_path_base,
        'comments': comments
    }
    expobj = aoutils.OnePhotonStim(data_path_base, date, animal_prep, trial, metainfo)


