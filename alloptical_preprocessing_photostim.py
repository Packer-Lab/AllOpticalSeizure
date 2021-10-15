# Step #1) in all optical experiment analysis - preprocessing the data to prep for suite2p analysis and creating some starter experiment objects

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import alloptical_utils_pj as aoutils

# 1) ### prepare trial and photostim experiment information below before running run_photostim_processing()
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-19'
animal_prep = 'PS07'
# date = '2021-02-02'
date = data_path_base[-10:]
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 4 things for every trial
trial = 't-011'  # note that %s magic command in the code below will be using these trials listed here
comments = '18 cells x 3 groups; 7mW per cell; 250ms stim multi_interleaved (prot. #3);  photostim responses seem to be more like the prot 3b pre 4ap 100ms stim. definitely there are dF responders short seizures'
naparms_loc = '/photostim/2021-01-19_PS07_photostim_013/'  # make sure to include '/' at the end to indicate the child directory
exp_type = 'post 4ap 2p all optical'  # use 'post' and '4ap' in the description to create the appropriate post4ap exp object
analysis_save_path = '/home/pshah/mnt/qnap/Analysis/%s/%s/' % (date, animal_prep)
# paqs_loc = '%s/%s_RL111_%s.paq' % (data_path_base, date, '008')  # path to the .paq files for the selected trials
######

## everything below should autopopulate and run automatically
paqs_loc = '%s/%s_%s_%s.paq' % (data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
analysis_save_path = analysis_save_path + tiffs_loc_dir[-16:]
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "%s/%s_%s.pkl" % (analysis_save_path, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
# analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]

matlab_pairedmeasurements_path = '%s/%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-22], animal_prep, date, animal_prep, trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
# matlab_pairedmeasurements_path = None

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}

expobj = aoutils.run_photostim_preprocessing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs, metainfo=metainfo,
                                             tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc, naparms_loc=(data_path_base+naparms_loc),
                                             paqs_loc=paqs_loc, matlab_pairedmeasurements_path=matlab_pairedmeasurements_path,
                                             processed_tiffs=False, discard_all=True, analysis_save_path=analysis_save_path)

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
