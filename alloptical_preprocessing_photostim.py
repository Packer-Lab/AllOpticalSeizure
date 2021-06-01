# Step #1) in all optical experiment analysis - preprocessing the data to prep for suite2p analysis and creating some starter experiment objects

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import alloptical_utils_pj as aoutils

# %% prepare trial and photostim experiment information below before running run_photostim_processing()
data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-18'
animal_prep = 'RL108'
# date = '2021-01-10'
date = data_path_base[-10:]
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 4 things for every trial
trial = 't-016'  # note that %s magic command in the code below will be using these trials listed here
comments = '10 cells x 5 groups; 7mW per cell; 250ms-stim-40hz-multi (prot. #2b) - no seizures'
naparms_loc = '/photostim/ 2021-01-10_PS06_photostim_009/'  # make sure to include '/' at the end to indicate the child directory
exp_type = 'post 4ap 2p all optical'  # use 'post' and '4ap' in the description to create the appropriate post4ap exp object
analysis_save_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/RL108/'
# paqs_loc = '%s/%s_RL111_%s.paq' % (data_path_base, date, '008')  # path to the .paq files for the selected trials
######


paqs_loc = '%s/%s_%s_%s.paq' % (data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
# analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]
analysis_save_path = analysis_save_path + tiffs_loc_dir[-16:]

# matlab_badframes_path = '%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-17], date, animal_prep, trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
matlab_badframes_path = None

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
                                             paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
                                             processed_tiffs=False, discard_all=True, analysis_save_path=analysis_save_path)