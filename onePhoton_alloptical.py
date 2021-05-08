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
# date = data_path_base[-10:]

# need to update these 3 things for every trial
trial = 't-008'  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim, pre 4ap'
comments = '20x trials of 1p stim'

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}

expobj = aoutils.OnePhotonStim(data_path_base, date, animal_prep, trial, metainfo)


#%%
import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot


###### IMPORT pkl file containing data in form of expobj
trial = 't-003'
date = '2021-01-19'
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date)


# %% # look at the average Ca Flu trace pre and post stim, just calculate the average of the whole frame and plot as continuous timeseries
# - this approach should also allow to look at the stims that give rise to extended seizure events where the Ca Flu stays up

# # EXCLUDE CERTAIN STIM START FRAMES
# expobj.stim_start_frames = [frame for frame in expobj.stim_start_frames if 4000 > frame or frame > 5000]
# expobj.stim_end_frames = [frame for frame in expobj.stim_end_frames if 4000 > frame or frame > 5000]
# expobj.stim_start_times = [time for time in expobj.stim_start_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_end_times = [time for time in expobj.stim_end_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_duration_frames = int(np.mean(
#     [expobj.stim_end_frames[idx] - expobj.stim_start_frames[idx] for idx in range(len(expobj.stim_start_frames))]))

aoplot.plotMeanRawFluTrace(expobj, stim_span_color='white', x_axis='frames', xlims=[0, 3000])
aoplot.plotLfpSignal(expobj, x_axis='paq')

aoplot.plot_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None, y_axis='dff')
aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75)

# TODO need to classify seizures in paq files for 1p photostim trials


# %% make downsampled tiff for viewing raw data

trial = 't-012'
date = '2021-01-19'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

