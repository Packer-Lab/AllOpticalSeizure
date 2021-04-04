### code for analysis of one photon photostim experiments

import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot

# %%
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-09'
animal_prep = 'PS04'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
date = data_path_base[-10:]
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 5 things for every trial
trial = 't-021'  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim'
comments = '20x 1p opto stim; tiff images are built properly'
paqs_loc = '%s/%s_PS04_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials


tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}

def run_1p_processing(tiffs_loc_dir, tiffs_loc, paqs_loc, pkl_path, metainfo):
    print('\n-----Processing trial # %s-----' % trial)

    paths = [tiffs_loc_dir, tiffs_loc, paqs_loc]
    # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

    expobj = aoutils.onePstim(paths, metainfo)

    # set analysis save path for expobj
    # make the necessary Analysis saving subfolder as well
    expobj.analysis_save_path = analysis_save_path
    if os.path.exists(expobj.analysis_save_path):
        pass
    elif os.path.exists(expobj.analysis_save_path[:-17]):
        os.mkdir(expobj.analysis_save_path)
    elif os.path.exists(expobj.analysis_save_path[:-27]):
        os.mkdir(expobj.analysis_save_path[:-17])
        os.mkdir(expobj.analysis_save_path)

    expobj.save_pkl(pkl_path=pkl_path)

    return expobj

expobj = run_1p_processing(tiffs_loc_dir, tiffs_loc, paqs_loc, pkl_path, metainfo)


# %%

import utils.funcs_pj as pjf
import matplotlib.pyplot as plt
import numpy as np
import os
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot

###### IMPORT pkl file containing data in form of expobj
trial = 't-021'
date = '2021-01-09'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)


# %% # look at the average Ca Flu trace pre and post stim, just calculate the average of the whole frame and plot as continuous timeseries
# - this approach should also allow to look at the stims that give rise to extended seizure events where the Ca Flu stays up


# exclude certain stim start frames
expobj.stim_start_frames = [frame for frame in expobj.stim_start_frames if 4000 > frame or frame > 5000]

aoplot.plot_flu_trace_1pstim(expobj, stim_span_color='black', x_axis='frames')
aoplot.plot_1pstim_avg_trace(expobj, x_axis='frames', individual_traces=True)

# TODO make plot of voltage signal pre and post 1p stim