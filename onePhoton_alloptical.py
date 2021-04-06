### code for analysis of one photon photostim experiments

import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot

# %%
animal_prep = 'PS07'
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-19'
date = data_path_base[-10:]

# need to update these 3 things for every trial
trial = 't-012'  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim, post 4ap'
comments = '10x trials of 1p stim'

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}


# paqs_loc = '%s/%s_%s_%s.paq' % (data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
# tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
# tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# # paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
# new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# # make the necessary Analysis saving subfolder as well
# analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]


expobj = aoutils.run_1p_processing(data_path_base, date, animal_prep, trial, metainfo)
# expobj = aoutils.run_1p_processing(tiffs_loc_dir, tiffs_loc, paqs_loc, pkl_path, metainfo, trial, analysis_save_path)


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
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)
# expobj.metainfo = metainfo
# expobj.save()

# %% # look at the average Ca Flu trace pre and post stim, just calculate the average of the whole frame and plot as continuous timeseries
# - this approach should also allow to look at the stims that give rise to extended seizure events where the Ca Flu stays up

# # EXCLUDE CERTAIN STIM START FRAMES
# expobj.stim_start_frames = [frame for frame in expobj.stim_start_frames if 4000 > frame or frame > 5000]
# expobj.stim_end_frames = [frame for frame in expobj.stim_end_frames if 4000 > frame or frame > 5000]
# expobj.stim_start_times = [time for time in expobj.stim_start_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_end_times = [time for time in expobj.stim_end_times if 5.5e6 > time or time > 6.5e6]
# expobj.stim_duration_frames = int(np.mean(
#     [expobj.stim_end_frames[idx] - expobj.stim_start_frames[idx] for idx in range(len(expobj.stim_start_frames))]))

aoplot.plot_flu_trace_1pstim(expobj, stim_span_color='white', x_axis='frames', xlims=[0, 3000])
aoplot.plot_lfp_1pstim(expobj, x_axis='paq')

aoplot.plot_1pstim_avg_trace(expobj, x_axis='time', individual_traces=True, stim_span_color=None)
aoplot.plot_lfp_1pstim_avg_trace(expobj, x_axis='time', individual_traces=False)

