import alloptical_utils_pj as aoutils


# Step #1) in all optical experiment analysis - preprocessing the data to prep for suite2p analysis and creating some starter experiment objects


# 1) ### prepare trial and photostim experiment information below before running run_photostim_processing()
data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-19'
animal_prep = 'RL109'
# date = '2021-02-02'
date = data_path_base[-10:]
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 4 things for every trial
trial = 't-013'  # note that %s magic command in the code below will be using these trials listed here
comments = 'photostim: 9 cells x 4 groups; 5mW per cell; preset: 2020-11-25_PS_250ms-stim-50hz (approach #1); same targets and protocol as t011 but random vs ekmeans group assignment'
naparms_loc = '/photostim/2020-12-19_RL109_ps_014/'  # make sure to include '/' at the end to indicate the child directory
exp_type = 'pre 4ap 2p all optical'  # use 'post' and '4ap' in the description to create the appropriate run_post4ap_trials exp object
analysis_save_path = f'/home/pshah/mnt/qnap/Analysis/{date}/{animal_prep}/{animal_prep}_{trial}'

pre4ap_trials = ['']  # add all optical t-series from pre4ap_trials
post4ap_trials = ['']  # add all optical t-series from post4ap_trials


# paqs_loc = '%s/%s_RL111_%s.paq' % (data_path_base, date, '008')  # path to the .paq files for the selected trials
######

## everything below should autopopulate and run automatically
paqs_loc = '%s/%s_%s_%s.paq' % (data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
# tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
# tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
tiffs_loc = f'{data_path_base}/{date}_{trial}/{date}_{trial}_Cycle00001_Ch3.tif'
# pkl_path = "%s/%s_%s.pkl" % (analysis_save_path, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
# analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]

if 'post' in exp_type and 'no seizure' not in comments:
    matlab_pairedmeasurements_path = '%s/%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-22], animal_prep, date, animal_prep, trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
else:
    matlab_pairedmeasurements_path = None


metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments,
    'pre4ap_trials': pre4ap_trials,
    'post4ap_trials': post4ap_trials
}


expobj = aoutils.run_photostim_preprocessing(trial, exp_type=exp_type, tiffs_loc=tiffs_loc,
                                             naparms_loc=(data_path_base + naparms_loc), paqs_loc=paqs_loc,
                                             metainfo=metainfo, new_tiffs=new_tiffs,
                                             matlab_pairedmeasurements_path=matlab_pairedmeasurements_path,
                                             processed_tiffs=False, discard_all=True,
                                             analysis_save_path=analysis_save_path)

# %% 2) ### the below is usually run from jupyter notebooks dedicated to each experiment prep.
to_suite2p = ['t-002', 't-006', 't-007', 't-008', 't-009', 't-011', 't-016', 't-017'] # specify all trials that were used in the suite2p runtotal_frames_stitched = 0
baseline_trials = ['t-002', 't-006'] # specify which trials to use as spont baseline
# note ^^^ this only works currently when the spont baseline trials all come first, and also back to back


trials = ['t-011']

for trial in trials:
    ###### IMPORT pkl file containing expobj
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, animal_prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, do_processing=False)
    expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/%s/suite2p/alloptical-2p-1_25x-alltrials/plane0' % date
    aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=to_suite2p, baseline_trials=baseline_trials,
                                                force_redo=True)
