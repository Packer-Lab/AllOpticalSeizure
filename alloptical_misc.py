import pickle

import pandas as pd

import alloptical_utils_pj as aoutils
from _utils_ import alloptical_plotting_utils as aoplot
import matplotlib.pyplot as plt
import numpy as np
from funcsforprajay import funcs as pj

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

import tifffile as tf

import _alloptical_utils as Utils
expobj = Utils.import_expobj(prep='PS04', trial='t-018')

# %%
sz_nums = np.unique([i for i in list(expobj.slmtargets_data.var.seizure_num) if type(i) is int and i > 0])
ncols = 3
nrows = int(np.ceil(len(sz_nums) / ncols)) if int(np.ceil(len(sz_nums) / ncols)) > 1 else 2
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
counter = 0
for sz in sz_nums:
    idx = np.where(expobj.slmtargets_data.var.seizure_num == sz)[0][0]  # first seizure invasion frame
    stim_frm = expobj.slmtargets_data.var.stim_start_frame[idx]
    time_del = expobj.slmtargets_data.var.delay_from_sz_onset_sec[idx]

    # plotting
    avg_stim_img_path = f'{expobj.analysis_save_path[:-1]}avg_stim_images/{expobj.metainfo["date"]}_{expobj.metainfo["trial"]}_stim-{stim_frm}.tif'
    bg_img = tf.imread(avg_stim_img_path)
    # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, cells=in_sz, edgecolors='yellowgreen', background=bg_img)
    # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, cells=out_sz, edgecolors='white', background=bg_img)
    ax = axs[counter // ncols, counter % ncols]
    fig, ax = aoplot.plot_SLMtargets_Locs(expobj, fig=fig, ax=ax,
                                          title=f"sz #: {sz}, stim_fr: {stim_frm}, time inv.: {time_del}s", show=False,
                                          background=bg_img)

    try:
        inframe_coord1_x = expobj.slmtargets_data.var["seizure location"][idx][0][0]
        inframe_coord1_y = expobj.slmtargets_data.var["seizure location"][idx][0][1]
        inframe_coord2_x = expobj.slmtargets_data.var["seizure location"][idx][1][0]
        inframe_coord2_y = expobj.slmtargets_data.var["seizure location"][idx][1][1]
    except TypeError:
        print('hitting nonetype error')
    ax.plot([inframe_coord1_x, inframe_coord2_x], [inframe_coord1_y, inframe_coord2_y], c='darkorange',
            linestyle='dashed', alpha=1, lw=2)

    counter += 1

fig.suptitle(f"{expobj.t_series_name} {expobj.date}")
fig.show()




# %%
import functools
import concurrent.futures


def run_for_loop_across_exps(func):
    t_start = time.time()
    @functools.wraps(func)
    def inner(*args, **kwargs):
        def somefunc(exp_prep):
            pass

        for exp_prep in pj.flattenOnce(allopticalResults.post_4ap_trials):
            prep = exp_prep[:-6]
            pre4aptrial = exp_prep[-5:]
            expobj, _ = aoutils.import_expobj(prep=prep, trial=pre4aptrial, verbose=False)
            aoutils.working_on(expobj)
            res_ = func(expobj=expobj, **kwargs)
            aoutils.end_working_on(expobj)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     trials = pj.flattenOnce(allopticalResults.post_4ap_trials)
        #     executor.map(somefunc, trials)
        #     # _ = [executor.submit(somefunc, exp_prep) for exp_prep in trials]

        t_end = time.time()
        pj.timer(t_start, t_end)
        print(f" {'--' * 5} COMPLETED FOR LOOP ACROSS EXPS {'--' * 5}\n")
    return inner

# @run_for_loop_across_exps
# def another_func(**kwargs):
#     expobj = kwargs['expobj']
#     print(f"experiment loaded: {expobj.t_series_name}")
#
# another_func()

@run_for_loop_across_exps
def collect_responses_vs_distance_to_seizure_SLMTargets(response_type: str, **kwargs):
    """

    :param response_type: either 'dFF (z scored)' or 'dFF (z scored) (interictal)'
    :param kwargs: must contain expobj as arg key
    """
    print(f"\t\- collecting responses vs. distance to seizure [5.0-2]")
    expobj = kwargs['expobj']

    # uncomment if need to rerun for a particular expobj....but shouldn't really need to be doing so
    if not hasattr(expobj, 'responses_SLMtargets_tracedFF'):
        expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.hits_SLMtargets_tracedFF, expobj.responses_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes = \
            expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10, stims_to_use=expobj.stim_start_frames)
        print(f'WARNING: {expobj.t_series_name} had to rerun .get_SLMTarget_responses_dff')

    # (re-)make pandas dataframe
    df = pd.DataFrame(columns=['target_id', 'stim_id', 'inorout_sz', 'distance_to_sz', response_type])

    for target in expobj.responses_SLMtargets_tracedFF.index:
        # idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        stim_ids = [(idx, stim) for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]

        ## z-scoring of SLM targets responses:
        z_scored = expobj.responses_SLMtargets_tracedFF  # initializing z_scored df
        if response_type == 'dFF (z scored)' or response_type == 'dFF (z scored) (interictal)':
            # set a different slice of stims for different variation of z scoring
            if response_type == 'dFF (z scored)': slice = expobj.responses_SLMtargets_tracedFF.columns  # (z scoring all stims all together from t-series)
            elif response_type == 'dFF (z scored) (interictal)': slice = expobj.stim_idx_outsz  # (z scoring all stims relative TO the interictal stims from t-series)
            __mean = expobj.responses_SLMtargets_tracedFF.loc[target, slice].mean()
            __std = expobj.responses_SLMtargets_tracedFF.loc[target, slice].std(ddof=1)
            # __mean = expobj.responses_SLMtargets_tracedFF.loc[target, :].mean()
            # __std = expobj.responses_SLMtargets_tracedFF.loc[target, :].std(ddof=1)

            __responses = expobj.responses_SLMtargets_tracedFF.loc[target, :]
            z_scored.loc[target, :] = (__responses - __mean) / __std

        for idx, stim in stim_ids:
            if target in expobj.slmtargets_szboundary_stim[stim]: inorout_sz = 'in'
            else: inorout_sz = 'out'

            distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, stim]

            if response_type == 'dFF': response = expobj.responses_SLMtargets_tracedFF.loc[target, idx]
            elif response_type == 'dFF (z scored)' or response_type == 'dFF (z scored) (interictal)': response = z_scored.loc[target, idx]  # z - scoring of SLM targets responses:
            else: raise ValueError('response_type arg must be `dFF` or `dFF (z scored)` or `dFF (z scored) (interictal)`')

            df = df.append({'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                            response_type: response}, ignore_index=True)

    expobj.responses_vs_distance_to_seizure_SLMTargets = df

    # convert distances to microns
    expobj.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz_um'] = round(expobj.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz'] / expobj.pix_sz_x, 2)
    expobj.save()


# run_calculating_min_distance_to_seizure(no_slmtargets_szboundary_stim)
response_type = 'dFF (z scored)'
collect_responses_vs_distance_to_seizure_SLMTargets(response_type=response_type)

key = 'f'; exp = 'post'; expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")


# %%

import concurrent.futures
import time

def somefunc(exp_prep):
    # print(f"\n{'-' * 5} RUNNING PRE4AP TRIALS {'-' * 5}")
    prep = exp_prep[:-6]
    pre4aptrial = exp_prep[-5:]
    expobj, _ = aoutils.import_expobj(prep=prep, trial=pre4aptrial, verbose=False)
    return expobj

### parallel processing
t_start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    trials = pj.flattenOnce(allopticalResults.pre_4ap_trials)
    results = [executor.submit(somefunc, exp_prep) for exp_prep in trials]
    # results = executor.map(somefunc, trials)

print('\n')

t_end = time.time()
pj.timer(t_start, t_end)


# ### for loop
# t_start = time.time()
# trials = pj.flattenOnce(allopticalResults.pre_4ap_trials)
# for exp_prep in trials:
#     a = somefunc(exp_prep)
#
# t_end = time.time()
# pj.timer(t_start, t_end)

# %%
to_suite2p = ['t-005', 't-006', 't-007', 't-008', 't-011', 't-012', 't-013', 't-014', 't-016',
              't-017', 't-018', 't-019', 't-020', 't-021']
baseline_trials = ['t-005', 't-006'] # specify which trials to use as spont baseline
# note ^^^ this only works currently when the spont baseline trials all come first, and also back to back


# trials = ['t-016', 't-017', 't-018', 't-019', 't-020', 't-021']
# trials = ['t-016', 't-017', 't-018', 't-020']
# trials = ['t-017', 't-018', 't-020']
# trials = ['t-018', 't-020']
trials = ['t-017']

for trial in trials:
    ###### IMPORT pkl file containing expobj
    date = '2020-12-19'
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, do_processing=False)
    expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-19/suite2p/alloptical-2p-1x-alltrials/plane0'

    expobj.pre_stim = int(0.5 * expobj.fps)  # length of pre stim trace collected
    expobj.post_stim = int(3 * expobj.fps)  # length of post stim trace collected
    expobj.post_stim_response_window_msec = 500  # msec
    expobj.post_stim_response_frames_window = int(expobj.fps * expobj.post_stim_response_window_msec / 1000)

    aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=to_suite2p, baseline_trials=baseline_trials,
                                                force_redo=True)

# %% testing which pkl objects are corrupted (if any)

pre_4ap_trials = [
    ['RL108 t-009'],
    ['RL108 t-010'],
    ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],
    ['RL109 t-014'],
    ['PS04 t-012', 'PS04 t-014',
     'PS04 t-017'],
    ['PS05 t-010'],
    ['PS07 t-007'],
    ['PS07 t-009'],
    ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
    ['PS06 t-011'],
    ['PS06 t-012'],
    ['PS11 t-007'],
    ['PS11 t-010'],
    ['PS17 t-005'],
    ['PS17 t-006', 'PS17 t-007'],
    ['PS18 t-006']
]

post_4ap_trials = [
    ['RL108 t-013'],
    ['RL108 t-011'],
    ['RL109 t-020'],
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016',  'RL109 t-017'],
    ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    ['PS07 t-017'],
    ['PS06 t-014', 'PS06 t-015'],
    ['PS06 t-013'],
    ['PS06 t-016'],
    ['PS11 t-016'],
    ['PS11 t-011'],
    ['PS17 t-011'],
    ['PS17 t-009'],
    ['PS18 t-008']
]


for key in pj.flattenOnce(post_4ap_trials)[-3:]:
    prep = key[:-6]
    trial = key[-5:]
    try:
        expobj, _ = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
    except pickle.UnpicklingError:
        print(f"\n** FAILED IMPORT OF * {prep} {trial}")
    aoplot.plot_lfp_stims(expobj)

# for key in ['RL109 t-020']:
#     prep = key[:-6]
#     trial = key[-5:]
#     expobj, _ = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
#     expobj.metainfo['pre4ap_trials'] = ['t-007', 't-008', 't-011', 't-012', 't-013', 't-014']
#     expobj.metainfo['post4ap_trials'] = ['t-016', 't-017', 't-018', 't-019', 't-020', 't-021']
#     expobj.save()


# %% updating the non-targets exclusion region to 15um from the center of the spiral coordinate

ls = ['pre', 'post']
for i in ls:
    ncols = 5
    nrows = 5
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 30))
    fig.tight_layout()
    counter = 0

    for key in list(allopticalResults.trial_maps[i].keys()):
        for j in range(len(allopticalResults.trial_maps[i][key])):
            # import expobj
            expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')

            expobj._parseNAPARMgpl()
            expobj._findTargetsAreas()
            expobj._findTargetedS2pROIs(force_redo=False, plot=False)

            ax = axs[counter // ncols, counter % ncols]

            fig, ax = aoplot.plot_cells_loc(expobj, cells=expobj.s2p_cell_targets, show=False, fig=fig, ax=ax,
                                            show_s2p_targets=True, title=None, invert_y=True)

            targ_img = np.zeros([expobj.frame_x, expobj.frame_y], dtype='uint16')
            target_areas = np.array(expobj.target_areas)
            targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1
            ax.imshow(targ_img, cmap='Greys_r', zorder=0)
            ax.set_title(f"{expobj.metainfo['animal prep.']}, {expobj.metainfo['trial']}")

            counter += 1

    title = f"s2p cell targets (red-filled) and target areas (white) - {i}4ap trials"
    save_path = f'/home/pshah/mnt/qnap/Analysis/Results_figs/{title}.png'
    plt.suptitle(title, y=0.90)
    plt.savefig(save_path)
    # fig.show()

#
# self._findTargetedCells()
#
#
# listdir = os.listdir(self.naparm_path)
# scale_factor = self.frame_x / 512
# ## All SLM targets
# for path in listdir:
#     if 'AllFOVTargets' in path:
#         target_file = path
# target_image = tf.imread(os.path.join(self.naparm_path, target_file))
#
# plt.imshow(target_image, cmap='Greys')
# plt.show()
#
#
#
#
# targetsareasFOV = np.zeros((self.frame_x, self.frame_y))
# for (y,x) in pj.flattenOnce(self.target_areas):
#     targetsareasFOV[x, y] = 1
# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(targetsareasFOV, cmap='Greys_r', zorder=0)
# ax.set_title('self.target_areas')
# fig.show()
#
#
#
# targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
# target_areas = np.array(self.target_areas)
# targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1
# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(targ_img, cmap='Greys_r', zorder=0)
# ax.set_title('12')
# fig.show()
#
#
#
# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(cell_img, cmap='Greys_r', zorder=0)
# ax.set_title('s2p cell targets (red-filled) and all target coords (green) %s/%s' % (
#     self.metainfo['trial'], self.metainfo['animal prep.']))
# fig.show()




print('yes')


# %%
prep = 'PS18'
date = '2021-02-02'
trials = ['t-002', 't-004', 't-005', 't-006', 't-007', 't-008', 't-009']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)

    # pj.plot_single_tiff(expobj.tiff_path, frame_num=201, title='%s - frame# 201' % trial)
    # cropped_tiff = aoutils.subselect_tiff(expobj.tiff_path, select_frames=(2668, 4471))#, save_as='/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_2668-4471fr.tif' % (date, prep, date, trial, date, trial))
    # aoutils.SaveDownsampledTiff(tiff_path=expobj.tiff_path, #stack=cropped_tiff,
    #                             save_as='/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_2668-4471fr_downsampled.tif' % (date, prep, date, trial, date, trial))
    # expobj.collect_seizures_info(seizures_lfp_timing_matarray='/home/pshah/mnt/qnap/Analysis/%s/%s/paired_measurements/%s_%s_%s.mat' % (date, prep, date, prep, trial[-3:]))
    # expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=False, save_img=True, force_redo=True)

    # expobj.MeanSeizureImages(
    #     baseline_tiff="/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_t-005/2020-12-18_t-005_Cycle00001_Ch3.tif",
    #     frames_last=1000)


# %% add 100ms to the stim dur for expobj which need it (i.e. trials where the stim end is just after the stim_dur and traces are still coming down)
ls = ['PS05 t-010', 'PS06 t-011', 'PS11 t-010', 'PS17 t-005', 'PS17 t-006', 'PS17 t-007', 'PS18 t-006']
for i in ls:
    prep, trial = re.split(' ', i)
    expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
    expobj.stim_dur = expobj.stim_dur + 100
    expobj.save()

# %% plot signals of suite2p outputs of a cell with F-neuropil and neuropil -- trying to see if neuropil signal contains anything of predictive value for the cell's spiking activity?

i = allopticalResults.post_4ap_trials[0]
j = 0
prep = i[j][:-6]
trial = i[j][-5:]
print('\nLoading up... ', prep, trial)
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

cell = 10
# window = [0, expobj.n_frames]
fig, ax = plt.subplots(figsize=(10, 3))
ax2 = ax.twinx()
ax.plot(expobj.frame_clock_actual[:expobj.n_frames], expobj.raw[cell], color='black', lw=0.2)
# ax.plot(expobj.spks[cell], color='blue')
ax.plot(expobj.frame_clock_actual[:expobj.n_frames], expobj.neuropil[cell], color='red', lw=0.2)
ax2.plot(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual], color='steelblue', lw=0.2)
# ax.set_xlim(window[0], window[1])
fig.show()




# %% running processing of SLM targets responses outsz

prep = 'RL108'
trial = 't-013'
date = ls(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == (prep + ' ' + trial), 'date'])[0]

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep, verbose=False)
hasattr(expobj, 'outsz_responses_SLMtargets')


# %% fixing squashing of images issue in PS18 trials

import cv2


prep = 'PS18'
date = '2021-02-02'
trial = 't-006'

###### IMPORT pkl file containing data in form of expobj
save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)
fr = pj.plot_single_tiff(expobj.tiff_path, frame_num=238, title='%s - frame 238' % trial)

# unsquash the bottom 2/3rd of the frame
lw = fr.shape[0]
fr_ = fr[int(1/3*lw):, ]
res = cv2.resize(fr_, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

# plot image
plt.imshow(res, cmap='gray')
plt.suptitle('%s' % trial)
plt.show()



# compare with spont trial that was correct
trial = 't-002'

###### IMPORT pkl file containing data in form of expobj
save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)
fr = pj.plot_single_tiff(expobj.tiff_path, frame_num=238, title='%s - frame 238' % trial)

# plot image
plt.imshow(fr, cmap='gray')
plt.suptitle('%s' % trial)
plt.show()




# open tiff file
print('\nWorking on... %s' % expobj.tiff_path)
tiffstack = tf.imread(expobj.tiff_path)



# %% checking number of frames in paq file and the 2p tiff path

prep = 'PS18'
date = '2021-02-02'
trials = ['t-008', 't-009']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)
    pj.plot_single_tiff(expobj.tiff_path, frame_num=238, title='%s - frame 238' % trial)
    # open tiff file
    print('\nWorking on... %s' % expobj.tiff_path)
    tiffstack = tf.imread(expobj.tiff_path)
    n_frames_tiff = len(tiffstack)
    if not hasattr(expobj, 'frame_clock_actual'):
        expobj.paqProcessing()
    n_frames_paq = len(expobj.frame_clock_actual)

    print('|- n_frames_tiff: %s      n_frames_paq: %s' % (n_frames_tiff, n_frames_paq))



# %%

prep = 'PS06'
date = '2021-01-10'
trials = ['t-008', 't-009', 't-010', 't-011']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)

    print('\n%s' % trial)
    print('frame clock count: ', len(expobj.frame_clock_actual))
    print('raw Flu trace count: ', len(expobj.meanRawFluTrace))
    print('xml nframes', expobj.n_frames)

# paq_read(expobj.paq_path, plot=True)


# %%
data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-19'
animal_prep = 'RL109'
date = '2020-12-19'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

trials = ['t-007']
for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    # trial = 't-009'  # note that %s magic command in the code below will be using these trials listed here
    save_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)

    # expobj._findTargets()
    # expobj.save()

    # aoplot.plotSLMtargetsLocs(expobj, background=expobj.meanFluImg_registered, title=None)

    aoutils.slm_targets_responses(expobj, experiment, trial, y_spacing_factor=4, smooth_overlap_traces=5, figsize=[30, 20],
                                  linewidth_overlap_traces=0.2, y_lims_periphotostim_trace=[-0.5, 3.0], v_lims_periphotostim_heatmap=[-0.5, 1.0],
                                  save_results=False)




# %% re-running pre-processing on an expobj that already exists for a given trial

data_path_base = '/home/pshah/mnt/qnap/Data/2021-02-02'
animal_prep = 'PS17'
date = data_path_base[-10:]
for trial in ['t-009', 't-011']:
    save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
    expobj, experiment = aoutils.import_expobj(save_path=save_path)
    comments = expobj.metainfo['comments']
    naparms_loc = expobj.naparm_path[-41:]
    exp_type = expobj.metainfo['exptype']
    analysis_save_path = '/home/pshah/mnt/qnap/Analysis/%s/%s/' % (date, animal_prep)

    paqs_loc = '%s/%s_%s_%s.paq' % (
    data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
    tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
    analysis_save_path = analysis_save_path + tiffs_loc_dir[-16:]
    tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
    save_path = "%s/%s_%s.pkl" % (analysis_save_path, date, trial)  # specify path in Analysis folder to save pkl object
    # paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
    new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
    matlab_badframes_path = '%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-17], date, animal_prep, trial[
                                                                                                                  2:])  # choose matlab path if need to use or use None for no additional bad frames
    metainfo = expobj.metainfo

    expobj = aoutils.run_photostim_preprocessing(trial, exp_type=exp_type, save_path=save_path, new_tiffs=new_tiffs,
                                                 metainfo=metainfo,
                                                 tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc,
                                                 naparms_loc=(data_path_base + naparms_loc),
                                                 paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
                                                 processed_tiffs=False, discard_all=True,
                                                 analysis_save_path=analysis_save_path)


# %%
save_path = '/home/pshah/mnt/qnap/Analysis/2021-01-08/PS05/2021-01-08_t-011/2021-01-08_t-011.pkl'
expobj, experiment = aoutils.import_expobj(save_path=save_path)


# x = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])

from matplotlib.colors import ColorConverter
c = ColorConverter().to_rgb
bwr_custom = pj.make_colormap([c('blue'), c('white'), 0.28, c('white'), c('red')])
aoplot.plot_traces_heatmap(expobj.SLMTargets_stims_dfstdF_avg, expobj, vmin=-2, vmax=5, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames + 1, x_axis='time', cbar=True,
                           title=(expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo['trial'] + ' - targets only'), xlims=(0, expobj.pre_stim +expobj.stim_duration_frames+ expobj.post_stim), cmap=bwr_custom)






# expobj.stim_start_frames = expobj.stim_start_frames + 3
# aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
#
#
# fig, ax = plt.subplots(figsize=[20, 3])


#
#
# #%%
# # paq_path = '/home/pshah/mnt/qnap/Data/2021-01-19/2021-01-19_PS07_015.paq'
# paq, _ = paq_read(expobj.paq_path, plot=True)
#
#%%

# CREATE AND SAVE DOWNSAMPLED TIFF
prep = 'PS06'
date = '2021-01-10'
trials = ['t-008']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, save_path=save_path, verbose=False)
    stack = pj.subselect_tiff(
        tiff_path=expobj.tiff_path, select_frames=(-500, -1))
    # stack = pj.subselect_tiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
    #                           select_frames=(-2000, -1))
    pj.SaveDownsampledTiff(stack=stack, save_as="/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_Cycle00001_Ch3_cropped_downsampled1.tif" % (date, prep, date, trial, date, trial))



#%% PLOT THE ZPROFILE OF A TIFF STACK
trial = 't-015'
date = '2021-01-19'

pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
            plot_image=True, figsize=[20, 4], title=(date + trial))


# %% PLOT LFP WITH STIM TIMINGS FOR ALL-OPTICAL EXPERIMENT

prep = 'PS07'
trial = 't-017'
date = ls(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == (prep + ' ' + trial), 'date'])[0]

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep)
aoplot.plot_lfp_stims(expobj)
