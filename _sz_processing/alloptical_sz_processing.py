# %% 0) IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
from _utils_ import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj
import numpy as np
import matplotlib.pyplot as plt
import _alloptical_utils as Utils
import tifffile as tf



# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Procesing_figs/sz_processing_boundaries_2022-01-06/'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None

# 6.0-dc) ANALYSIS: calculate time delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment
expobj = Utils.import_expobj(prep='RL108', trial='t-013')
# expobj = Utils.import_expobj(prep='RL109', trial='t-017')

# %% plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def plot_sz_invasion(**kwargs):
    expobj: aoutils.Post4ap = kwargs['expobj']

    sz_nums = np.unique([i for i in list(expobj.slmtargets_data.var.seizure_num) if type(i) is int and i > 0])
    fig, axs, counter, ncols, nrows = aoplot.multi_plot_subplots(num_total_plots=len(sz_nums))
    for sz in sz_nums:
        idx = np.where(expobj.slmtargets_data.var.seizure_num == sz)[0][0]  # first seizure invasion frame
        stim_frm = expobj.slmtargets_data.var.stim_start_frame[idx]
        time_del = expobj.slmtargets_data.var.delay_from_sz_onset_sec[idx]

        # plotting
        avg_stim_img_path = f'{expobj.analysis_save_path[:-1]}avg_stim_images/{expobj.metainfo["date"]}_{expobj.metainfo["trial"]}_stim-{stim_frm}.tif'
        bg_img = tf.imread(avg_stim_img_path)
        # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, cells=in_sz, edgecolors='yellowgreen', background=bg_img)
        # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, cells=out_sz, edgecolors='white', background=bg_img)
        ax = aoplot._get_ax_for_multi_plot(axs, counter, ncols)
        fig, ax = aoplot.plot_SLMtargets_Locs(expobj, fig=fig, ax=ax,
                                              title=f"sz #: {sz}, stim_fr: {stim_frm}, time inv.: {time_del}s", show=False,
                                              background=bg_img)

        try:
            inframe_coord1_x = expobj.slmtargets_data.var["seizure location"][idx][0][0]
            inframe_coord1_y = expobj.slmtargets_data.var["seizure location"][idx][0][1]
            inframe_coord2_x = expobj.slmtargets_data.var["seizure location"][idx][1][0]
            inframe_coord2_y = expobj.slmtargets_data.var["seizure location"][idx][1][1]
            ax.plot([inframe_coord1_x, inframe_coord2_x], [inframe_coord1_y, inframe_coord2_y], c='darkorange', linestyle='dashed', alpha=1, lw=2)
        except TypeError:
            print('hitting nonetype error')


    fig.suptitle(f"{expobj.t_series_name} {expobj.date}")
    fig.show()

plot_sz_invasion()


# %% 6.0-dc) ANALYSIS: calculate time delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment
# -- this section has been moved to _ClassExpSeizureAnalysis .22/02/20 -- this copy here is now archived

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def szInvasionTime(**kwargs):
    """
    The general approach to calculate seizure invasion time delay is to calculate the first stim (which are usually every 10 secs)
    which has the seizure wavefront in the FOV relative to the LFP onset of the seizure (which is at the 4ap inj site).



    :param kwargs: no args taken. used only to pipe in experiments from for loop.
    """

    expobj = kwargs['expobj']
    time_delay_sec = [-1]*len(expobj.stim_start_frames)
    sz_num = [-1]*len(expobj.stim_start_frames)
    for i in range(expobj.numSeizures):
        # i = 1
        lfp_onset_fr = expobj.seizure_lfp_onsets[i]
        if lfp_onset_fr != 0:
            start = expobj.seizure_lfp_onsets[i]
            stop = expobj.seizure_lfp_offsets[i]
            _stim_insz = [stim_fr for stim_fr in expobj.stim_start_frames if start < stim_fr < stop]
            stims_wv = [stim_fr for stim_fr in _stim_insz if stim_fr in expobj.stimsWithSzWavefront]
            stims_nowv = [stim_fr for stim_fr in _stim_insz if stim_fr not in expobj.stimsWithSzWavefront]
            if len(stims_wv) > 0:
                for stim in stims_wv:
                    if stim in expobj.stimsWithSzWavefront:
                        sz_start_sec = start / expobj.fps
                        _time_delay_sec = (stim / expobj.fps) - sz_start_sec
                        idx = np.where(expobj.slmtargets_data.var.stim_start_frame == stim)[0][0]
                        time_delay_sec[idx] = round(_time_delay_sec, 3)
                        sz_num[idx] = i
                for stim in stims_nowv:
                    if stim < stims_wv[0]:  # first in seizure stim frame with the seizure wavefront
                        idx = np.where(expobj.slmtargets_data.var.stim_start_frame == stim)[0][0]
                        time_delay_sec[idx] = "bf invasion"  # before seizure invasion to the FOV
                        sz_num[idx] = i
                    elif stim > stims_wv[-1]:  # last in seizure stim frame with the seizure wavefront
                        idx = np.where(expobj.slmtargets_data.var.stim_start_frame == stim)[0][0]
                        time_delay_sec[idx] = "af invasion"  # after seizure wavefront has passed the FOV
                        sz_num[idx] = i

    expobj.slmtargets_data.add_variable(var_name='delay_from_sz_onset_sec', values=time_delay_sec)
    expobj.slmtargets_data.add_variable(var_name='seizure_num', values=sz_num)
    expobj.save()


szInvasionTime()




# %% 5.0-dc) ANALYSIS: cross-correlation between mean FOV 2p calcium trace and LFP seizure trace - incomplete not working yet
import scipy.signal as signal

expobj = aoutils.import_expobj(prep='RL109', trial='t-017')

sznum = 1
slice = np.s_[expobj.convert_frames_to_paqclock(expobj.seizure_lfp_onsets[sznum]): expobj.convert_frames_to_paqclock(expobj.seizure_lfp_offsets[sznum])]

# detrend
detrended_lfp = signal.detrend(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual])[slice]*-1

# downsample LFP signal to the same # of datapoints as # of frames in 2p calcium trace
CaUpsampled1 = signal.resample(expobj.meanRawFluTrace, len(detrended_lfp))[slice]

pj.make_general_plot([CaUpsampled1], figsize=[20,3])
pj.make_general_plot([detrended_lfp], figsize=[20,3])

# use numpy or scipy.signal .correlate to correlate the two timeseries
correlated = signal.correlate(CaUpsampled1, detrended_lfp)
lags = signal.correlation_lags(len(CaUpsampled1), len(detrended_lfp))
correlated /= np.max(correlated)

f, axs = plt.subplots(nrows=3, ncols=1, figsize=[20,9])
axs[0].plot(CaUpsampled1)
axs[1].plot(detrended_lfp)
# axs[2].plot(correlated)
axs[2].plot(lags, correlated)
f.show()


#%% 1) defining trials to run for analysis

# this list should line up with the analysis list for run_post4ap_trials trials
ls = [
    ['RL108 t-013'],
    ['RL108 t-011'],
    ['RL109 t-020'],  # - hasnt been inspected for flip stims requirements
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016'], #, 'RL109 t-017'], - need to redo paired measurements
    ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    # ['PS07 t-017'], - lots of little tiny seizures, not sure if they are propagating through likely not
    # ['PS06 t-014', 'PS06 t-015'], - missing seizure_lfp_onsets
    ['PS06 t-013'],
    # ['PS06 t-016'], - missing seizure_lfp_onsets
    ['PS11 t-016'],
    ['PS11 t-011'],
    # ['PS17 t-011'],
    ['PS17 t-009'],
    # ['PS18 t-008']
]

ls2 = [#['RL108 t-011'],
       ['RL109 t-018'],
       # ['PS04 t-018'],
       # ['RL109 t-020'],
       ]

# %% 2.1) classification of cells in/out of sz boundary

trials_without_flip_stims = []

# expobj, _ = aoutils.import_expobj(trial='t-020', prep='RL109')
# expobj.sz_locations_stims()

# using run_post4ap_trials experiments from allopticalResults attr. in for loop for processing:

# for i in allopticalResults.post_4ap_trials:
for i in ls2:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[-1]
        # j = -1
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nWorking on @ ', prep, trial)
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
        # aoplot.plot_lfp_stims(expobj)

# matlab_pairedmeasurements_path = '%s/paired_measurements/%s_%s_%s.mat' % (expobj.analysis_save_path[:-23], expobj.metainfo['date'], expobj.metainfo['animal prep.'], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
# expobj.paqProcessing()
# expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path)
# expobj.save()

# aoplot.plotSLMtargetsLocs(expobj, background=None)

# ######## CLASSIFY SLM PHOTOSTIM TARGETS AS IN OR OUT OF current SZ location in the FOV
# -- FIRST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv

        if not hasattr(expobj, 'sz_boundary_csv_done'):
            expobj.sz_boundary_csv_done = True
        else:
            AssertionError('confirm that sz boundary csv creation has been completed')
            # sys.exit()

        expobj.sz_locations_stims() if not hasattr(expobj, 'stimsSzLocations') else None

        # specify stims for classifying cells
        on_ = []
        if 0 in expobj.seizure_lfp_onsets:  # this is used to check if 2p imaging is starting mid-seizure (which should be signified by the first lfp onset being set at frame # 0)
            on_ = on_ + [expobj.stim_start_frames[0]]
        on_.extend(expobj.stims_bf_sz)
        if len(expobj.stims_af_sz) != len(on_):
            end = expobj.stims_af_sz + [expobj.stim_start_frames[-1]]
        else:
            end = expobj.stims_af_sz
        print('\-seizure start frames: ', on_)
        print('\-seizure end frames: ', end)

        ##### import the CSV file in and classify cells by their location in or out of seizure

        if not hasattr(expobj, 'not_flip_stims'):
        # if hasattr(expobj, 'not_flip_stims'):
            print(f"|-- expobj {prep} {trial} DOES NOT have previous not_flip_stims attr, so making a new empty list attr")
            expobj.not_flip_stims = []  # specify here the stims where the flip=False leads to incorrect assignment
        else:
            print(f"\-expobj.not_flip_stims: {expobj.not_flip_stims}")

        # break

        print(' \nworking on classifying cells for stims start frames...')
        ## TODO need to implement rest of code for s2prois_szboundary_stim
        expobj.slmtargets_szboundary_stim = {}
        expobj.s2prois_szboundary_stim = {}

        ######## - all stims in sz are classified, with individual sz events labelled

        stims_of_interest = expobj.stimsWithSzWavefront
        print(' \-all stims in seizures: \n \-', stims_of_interest)
        nrows = len(stims_of_interest) // 4 + 1
        if nrows == 1:
            nrows += 1
        ncols = 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
        counter = 0

        for stim in expobj.stimsWithSzWavefront:
            sz_num = expobj.stimsSzLocations.loc[stim, 'sz_num']
            # print(' \-- working sz # %s with stims: \n \---' % (sz_num))

            print(f"considering stim # {stim}")

            ax = axs[counter // ncols, counter % ncols]

            # sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
            if os.path.exists(expobj.sz_border_path(stim=stim)):
                # first round of classifying (dont flip any cells over) - do this in the second round
                if stim not in expobj.not_flip_stims:
                    flip = False
                else:
                    flip = True

                # # classification of suite2p ROIs relative to sz boundary
                # in_sz, out_sz, fig, ax = expobj.classify_cells_sz_bound(stim=stim, to_plot=True,
                #                                                         flip=flip, fig=fig, ax=ax, text='sz %s stim %s' % (sz_num, stim))
                # expobj.s2prois_szboundary_stim[stim] = in_sz

                # # classification of SLM targets relative to sz boundary
                in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(stim=stim, to_plot=True, title=stim, flip=flip, fig=fig, ax=ax)
                expobj.slmtargets_szboundary_stim[stim] = in_sz  # for each stim, there will be a ls of cells that will be classified as in seizure or out of seizure

                axs[counter // ncols, counter % ncols] = ax
                counter += 1
            else:
                print(f"sz border path doesn't exist for stim {stim}: {expobj.sz_border_path(stim=stim)}")

        fig.suptitle('%s %s - Avg img around stims during- all stims' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']), y=0.995)
        save_path_full = f"{save_path_prefix}/{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} {len(expobj.stimsWithSzWavefront)} events.png"
        print(f"saving fig to: {save_path_full}")
        fig.savefig(save_path_full)
        # fig.show()

        expobj.save()
print('end end end.')

sys.exit()

# %% 2.1.1) specify which stims are flipped for boundary assignment and repeat the above code in 2.1.2)
# to correct for mis-assignment of cells (look at results and then find out which stims need to be flipped)

 # ['RL109 t-020'],
    # ['RL109 t-021'],
    # ['RL109 t-018'],
# prep = 'PS11'
# trial = 't-011'
# expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=True)

expobj, _ = aoutils.import_expobj(trial='t-018', prep='RL109')

# expobj.not_flip_stims = [expobj.stims_in_sz[1:]]  # specify here the stims where the flip=False leads to incorrect assignment
expobj.not_flip_stims = expobj.stims_in_sz[1:]  # specify here the stims where the flip=False leads to incorrect assignment

expobj.save()

# %% 2.1.2) re-run with new flip stims
expobj.slmtargets_szboundary_stim = {}
expobj.s2prois_szboundary_stim = {}
sz_num = 0
for sz_num, stims_of_interest in enumerate(expobj.stimsWithSzWavefront):
    print('|-', stims_of_interest)

    nrows = len(stims_of_interest) // 4 + 1
    if nrows == 1:
        nrows += 1
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    counter = 0

    for stim in stims_of_interest:
        ax = axs[counter // ncols, counter % ncols]

        # sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
        if not os.path.exists(expobj.sz_border_path(stim=stim)):
            print(expobj.sz_border_path(stim=stim))
        # first round of classifying (dont flip any cells over) - do this in the second round
        if stim not in expobj.not_flip_stims:
            flip = False
        else:
            flip = True

        in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(expobj.sz_border_path(stim=stim), stim=stim, to_plot=True, title='%s' % stim, flip=flip, fig=fig, ax=ax)
        expobj.slmtargets_szboundary_stim[stim] = in_sz  # for each stim, there will be a ls of cells that will be classified as in seizure or out of seizure

        axs[counter // ncols, counter % ncols] = ax
        counter += 1
    fig.suptitle('%s %s - Avg img around stims during sz - seizure # %s' % (
    expobj.metainfo['animal prep.'], expobj.metainfo['trial'], sz_num + 1), y=0.995)
    fig.show()
    sz_num += 1


# %% archive-3) responses of targets during seizures, but EXCLUDE STIMS WHERE THE CELL IS INSIDE THE SZ BOUND -- should this be in ARCHIVEDalloptical_results_photostim.py???? -- yeah most of this code should probably be retired - and current location is in alloptical_results_photostim_SLMtargets.py

# if hasattr(expobj, 'stims_in_sz'):
#
#     # stims during sz
#     stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
#     if len(stims) > 0:
#         expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
#             aoutils.get_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=stims)
#
#     # stims outside sz
#     stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
#     # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
#     if len(stims) > 0:
#         expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
#             aoutils.get_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=stims)
# else:
#     expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets_dfprestimf = \
#         aoutils.get_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=None)

assert hasattr(expobj, 'stims_in_sz'), AttributeError(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} doesn't have stims_in_sz attr.")

# collect stim responses with stims excluded as necessary
targets_avgresponses_exclude_stims_sz = {}
for row in expobj.insz_responses_SLMtargets.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
    responses = [expobj.insz_responses_SLMtargets.loc[row][stim] for stim in expobj.stims_in_sz if row not in expobj.slmtargets_szboundary_stim[stim]]
    targets_avgresponses_exclude_stims_sz[row] = np.mean(responses)


targets_avgresponses_stims_outsz = {}
for row in expobj.insz_responses_SLMtargets.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    responses = [expobj.outsz_responses_SLMtargets.loc[row][stim] for stim in expobj.stims_out_sz]
    targets_avgresponses_stims_outsz[row] = np.mean(responses)


targets_avgresponses_stims_presz = {}
for row in expobj.responses_SLMtargets_dfprestimf.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    responses = [expobj.responses_SLMtargets_dfprestimf.loc[row][stim] for stim in expobj.stim_start_frames]
    targets_avgresponses_stims_presz[row] = np.mean(responses)


data = [list(targets_avgresponses_exclude_stims_sz.values()), list(targets_avgresponses_stims_outsz.values()), list(targets_avgresponses_stims_presz.values())]
pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ', figsize=(5,5), fill_color=['orange', 'skyblue', 'green'],
                     colors=['black'] * 3, legend_labels=[None] * 3)


# %% 4) PLOT - stim frames for figures with SLM targets inside (yellow) and outside (green) of seizure boundary

prep = 'RL108'
trial = 't-013'
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
aoplot.plot_lfp_stims(expobj, xlims=[0.2e7, 1.0e7], linewidth=1.0)
# aoplot.plotLfpSignal(expobj, downsample=True, figsize=(6,2), x_axis='Time', xlims=[120 * expobj.paq_rate, 480 * expobj.paq_rate],
#                      ylims=[-6, 2], color='slategray', stim_span_color='green', alpha=0.1)

# %%
sz_num = -1
stims_to_plot = [stim for stim in expobj.stim_start_frames if expobj.seizure_lfp_offsets[sz_num] > stim > expobj.seizure_lfp_onsets[sz_num]]

nrows = 2
ncols  = int(len(stims_to_plot) / nrows)

fig, axs = plt.subplots(figsize=[5*ncols, 5*nrows], nrows=2, ncols=ncols)
counter = 0
for i in range(len(stims_to_plot)):
    row = int(counter // (len(stims_to_plot) / nrows))
    col = counter % len(stims_to_plot) - ncols
    ax = axs[row, col]
    stim = stims_to_plot[i]
    # fig, ax = plt.subplots(figsize=[5, 5], nrows=1, ncols=1)
    # stim = stims_to_plot[0]

    # plot SLM targets in sz boundary
    coords_to_plot = expobj.target_coords_all  # all targets
    coords_to_plot2 = [expobj.target_coords_all[cell] for cell in expobj.slmtargets_szboundary_stim[stim]]  # targets in sz bound
    # read in avg stim image to use as the background

    avg_stim_img_path = expobj.analysis_save_path + 'avg_stim_images' + f'/{expobj.t_series_name}_stim-{stim}.tif'
    bg_img = tf.imread(avg_stim_img_path)
    title=f"{prep} {trial} {stim} - SLM targets"
    for (x, y) in coords_to_plot:
        ax.scatter(x=x, y=y, edgecolors='#db6120', facecolors='#db6120', linewidths=2.5, zorder=4)
    for (x, y) in coords_to_plot2:
        ax.scatter(x=x, y=y, edgecolors='#f8cc8f', facecolors='#f8cc8f', linewidths=2.5, zorder=4)
    #
    #
    # fig, ax = aoplot.plot_cells_loc(expobj, cells=cells_to_plot, show_s2p_targets=False, fig=fig, ax=ax, show=False, scatter_only=True)
    # fig, ax = aoplot.plot_cells_loc(expobj, cells=cells_to_plot2, show_s2p_targets=False, fig=fig, ax=ax, show=False, scatter_only=True)
    ax.imshow(bg_img, cmap='Greys_r', zorder=0)
    # ax.set_title(title)
    ax.set_xticks(ticks=[])
    ax.set_xticklabels([])
    ax.set_yticks(ticks=[])
    ax.set_yticklabels([])
    counter += 1
fig.tight_layout(pad=2)
fig.show()


# %%

def _trialProcessing_nontargets(expobj, normalize_to='pre-stim', save=True):
    '''
    Uses dfstdf traces for individual cells and photostim trials, calculate the mean amplitudes of response and
    statistical significance across all trials for all cells

    Inputs:
        plane             - imaging plane n
    '''

    print('\n----------------------------------------------------------------')
    print('running trial Processing for nontargets ')
    print('----------------------------------------------------------------')

    # define non targets from suite2p ROIs (exclude cells in the SLM targets exclusion - .s2p_cells_exclude)
    expobj.s2p_nontargets = [cell for cell in expobj.good_cells if cell not in expobj.s2p_cells_exclude]  ## exclusion of cells that are classified as s2p_cell_targets

    ## collecting nontargets stim traces from in sz imaging frames
    # - - collect stim traces as usual for all stims, then use the sz boundary dictionary to nan cells/stims insize sz boundary
    # make trial arrays from dff data shape: [cells x stims x frames]
    # stim_timings_outsz = [stim for stim in expobj.stim_start_frames if stim not in expobj.seizure_frames]; stim_timings=expobj.stims_out_sz
    expobj._makeNontargetsStimTracesArray(stim_timings=expobj.stim_start_frames, normalize_to=normalize_to, save=False)

    if hasattr(expobj, 'slmtargets_szboundary_stim'):
        stim_timings_insz = [(x, stim) for x, stim in enumerate(expobj.stim_start_frames) if stim in list(expobj.slmtargets_szboundary_stim.keys())]
        # expobj._makeNontargetsStimTracesArray(stim_timings=stim_timings_insz, normalize_to=normalize_to,
        #                                       save=False)
        print('\nexcluding cells for stims inside sz boundary')
        for x, stim in stim_timings_insz:
            # stim = stim_timings_insz[0]
            exclude_list = [idx for idx, cell in enumerate(expobj.s2p_nontargets) if cell in expobj.slmtargets_szboundary_stim[stim]]

            expobj.dff_traces[exclude_list, x, :] = [np.nan] * expobj.dff_traces.shape[2]
            expobj.dfstdF_traces[exclude_list, x, :] = [np.nan] * expobj.dfstdF_traces.shape[2]
            expobj.raw_traces[exclude_list, x, :] = [np.nan] * expobj.raw_traces.shape[2]

        ## need to redefine _avg arrays post exclusion for Post4ap expobj's
        expobj.dff_traces_avg = np.nanmean(expobj.dff_traces, axis=1)
        expobj.dfstdF_traces_avg = np.nanmean(expobj.dfstdF_traces, axis=1)
        expobj.raw_traces_avg = np.nanmean(expobj.raw_traces, axis=1)

    else:
        AttributeError(
            'no slmtargets_szboundary_stim attr, so classify cells in sz boundary hasnot been saved for this expobj')


    # create parameters, slices, and subsets for making pre-stim and post-stim arrays to use in stats comparison
    # test_period = expobj.pre_stim_response_window_msec / 1000  # sec
    # expobj.test_frames = int(expobj.fps * test_period)  # test period for stats
    expobj.pre_stim_frames_test = np.s_[expobj.pre_stim - expobj.pre_stim_response_frames_window: expobj.pre_stim]
    stim_end = expobj.pre_stim + expobj.stim_duration_frames
    expobj.post_stim_frames_slice = np.s_[stim_end: stim_end + expobj.post_stim_response_frames_window]

    # mean pre and post stimulus (within post-stim response window) flu trace values for all cells, all trials
    expobj.analysis_array = expobj.dfstdF_traces  # NOTE: USING dF/stdF TRACES
    expobj.pre_array = np.mean(expobj.analysis_array[:, :, expobj.pre_stim_frames_test],
                               axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
    expobj.post_array = np.mean(expobj.analysis_array[:, :, expobj.post_stim_frames_slice],
                                axis=1)  # [cells x poststim frames] (avg'd taken over all stims)

    # ar2 = expobj.analysis_array[18, :, expobj.post_stim_frames_slice]
    # ar3 = ar2[~np.isnan(ar2).any(axis=1)]
    # assert np.nanmean(ar2) == np.nanmean(ar3)
    # expobj.analysis_array = expobj.analysis_array[~np.isnan(expobj.analysis_array).any(axis=1)]

    # measure avg response value for each trial, all cells --> return array with 3 axes [cells x response_magnitude_per_stim (avg'd taken over response window)]
    expobj.post_array_responses = []  ### this and the for loop below was implemented to try to root out stims with nan's but it's likley not necessary...
    for i in np.arange(expobj.analysis_array.shape[0]):
        a = expobj.analysis_array[i][~np.isnan(expobj.analysis_array[i]).any(axis=1)]
        responses = a.mean(axis=1)
        expobj.post_array_responses.append(responses)

    expobj.post_array_responses = np.mean(expobj.analysis_array[:, :, expobj.post_stim_frames_slice], axis=2)

    expobj._runWilcoxonsTest(save=False)

    expobj.save() if save else None

# %% 4.2) measure seizure legnths across all imaging trials (including any spont imaging you might have)

for key in list(allopticalResults.trial_maps['post'].keys()):
    # import initial expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.0' % key, verbose=False)
    prep = expobj.metainfo['animal prep.']
    # look at all run_post4ap_trials trials in expobj
    if 'post-4ap trials' in expobj.metainfo.keys():
        a = 'post-4ap trials'
    elif 'post4ap_trials' in expobj.metainfo.keys():
        a = 'post4ap_trials'
    # for loop over all of those run_post4ap_trials trials
    for trial in expobj.metainfo['%s' % a]:
        # import expobj
        expobj, experiment = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
        # count the average length of each seizure
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
            counter = 0
            sz_lengths_total = 0
            if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                    if sz_onset != 0:
                        sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] -
                                             expobj.frame_clock_actual[sz_onset]) / expobj.paq_rate
                        counter += 1
                avg_len = sz_lengths_total / counter
                expobj.avg_sz_len = avg_len

                print('Avg. seizure length (secs) for %s, %s, %s: ' % (prep, trial, expobj.metainfo['exptype']),
                      np.round(expobj.avg_sz_len, 2))

        else:
            n_seizures = 0
            print('no sz events for %s, %s, %s ' % (prep, trial, expobj.metainfo['exptype']))

# 4.2.1) measure seizure incidence across onePstim trials
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[
                onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[
        0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path, verbose=False)
        # count the average length of each seizure
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
            counter = 0
            sz_lengths_total = 0
            if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                    if sz_onset != 0:
                        sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] -
                                             expobj.frame_clock_actual[sz_onset]) / expobj.paq_rate
                        counter += 1
                avg_len = sz_lengths_total / counter
                expobj.avg_sz_len = avg_len
                print('Avg. seizure length (secs) for %s, %s, %s: ' % (
                    expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
                      np.round(expobj.avg_sz_len, 2))

        else:
            n_seizures = 0
            print('Avg. seizure length (secs) for %s, %s, %s ' % (
                expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']))

# 4.2.2) plot seizure length across onePstim and twoPstim trials
twop_trials = [24.0, 93.73, 38.86, 84.77, 17.16, 83.78, 15.78, 36.88]
onep_trials = [30.02, 34.25, 114.53, 35.57]

pj.plot_bar_with_points(data=[twop_trials, onep_trials], x_tick_labels=['2p stim', '1p stim'],
                        colors=['purple', 'green'], y_label='seizure length (secs)',
                        title='Avg. length of sz', expand_size_x=0.4, expand_size_y=1, ylims=[0, 120], title_pad=15,
                        shrink_text=0.8)

pj.plot_bar_with_points(data=[twop_trials + onep_trials], x_tick_labels=['Experiments'],
                        colors=['green'], y_label='Seizure length (secs)', alpha=0.7, bar=False,
                        title='Avg sz length', expand_size_x=0.7, expand_size_y=1, ylims=[0, 120],
                        shrink_text=0.8)


# %% 4.1) counting seizure incidence across all imaging trials

for key in list(allopticalResults.trial_maps['post'].keys()):
    # import initial expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.0' % key, verbose=False)
    prep = expobj.metainfo['animal prep.']
    # look at all run_post4ap_trials trials in expobj
    if 'post-4ap trials' in expobj.metainfo.keys():
        a = 'post-4ap trials'
    elif 'post4ap_trials' in expobj.metainfo.keys():
        a = 'post4ap_trials'
    # for loop over all of those run_post4ap_trials trials
    for trial in expobj.metainfo['%s' % a]:
        # import expobj
        expobj, experiment = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
        total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

        # count seizure incidence (avg. over mins) for each experiment (animal)
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
        else:
            n_seizures = 0

        print('Seizure incidence for %s, %s, %s: ' % (prep, trial, expobj.metainfo['exptype']),
              np.round(n_seizures / total_time_recording, 2))

# 4.1.1) measure seizure incidence across onePstim trials
for pkl_path in onePresults.mean_stim_responses['pkl_list']:
    if list(onePresults.mean_stim_responses.loc[
                onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[
        0] != '-':

        expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path, verbose=False)
        total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

        # count seizure incidence (avg. over mins) for each experiment (animal)
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
        else:
            n_seizures = 0

        print('Seizure incidence for %s, %s, %s: ' % (
        expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
              np.round(n_seizures / total_time_recording, 2))

# 4.1.2) plot seizure incidence across onePstim and twoPstim trials
twop_trials = [0.35, 0.251666667, 0.91, 0.33, 0.553333333, 0.0875, 0.47, 0.33, 0.52]
onep_trials = [0.38, 0.26, 0.19, 0.436666667, 0.685]

pj.plot_bar_with_points(data=[twop_trials, onep_trials], x_tick_labels=['2p stim', '1p stim'],
                        colors=['purple', 'green'], y_label='sz incidence (events/min)',
                        title='rate of seizures during exp', expand_size_x=0.4, expand_size_y=1, ylims=[0, 1],
                        shrink_text=0.8)


pj.plot_bar_with_points(data=[twop_trials + onep_trials], x_tick_labels=['Experiments'],
                        colors=['#2E3074'], y_label='Seizure incidence (events/min)', alpha=0.7, bar=False,
                        title='rate of seizures during exp', expand_size_x=0.7, expand_size_y=1, ylims=[0, 1],
                        shrink_text=0.8)



# %% ARCHIVE

# %% temp fixing getting sz locations per stim

expobj.stimsSzLocations = pd.DataFrame(data=None, index=expobj.stims_in_sz, columns=['sz_num', 'coord1', 'coord2', 'wavefront_in_frame'])

# specify stims for classifying cells
on_ = []
_end = []
for start, stop in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):

    stims_frames = [stim for stim in expobj.stim_start_frames if start < stim < stop]
    if len(stims_frames) > 0:
        on_.append(stims_frames[0])
        _end.append(stims_frames[-1])
        print(f"start: {start}, on_: {stims_frames[0]}")
        print(f"stop: {stop}, on_: {stims_frames[-1]}\n")
    #
    # if 0 in expobj.seizure_lfp_onsets and expobj.seizure_lfp_offsets[0] > expobj.stim_start_frames[0]:
    #     frame = expobj.stim_start_frames[0]
    #     # print(f"{frame} [1]")
    #     on_.append(frame)
    # else:
    #     frame = [stim_frame for stim_frame in expobj.stim_start_frames if start < stim_frame < stop]
    #     # print(f"{frame} [2]")
    #     on_.append(frame[0]) if len(frame) > 0 else None
    #
    # _end.extend(stop)
    # if expobj.stim_start_frames[-1] > expobj.seizure_lfp_offsets[-1]:
    #     pass


# on_ = []
# if 0 in expobj.seizure_lfp_onsets:  # this is used to check if 2p imaging is starting mid-seizure (which should be signified by the first lfp onset being set at frame # 0)
#     on_ = on_ + [expobj.stim_start_frames[0]]
# on_.extend(expobj.stims_bf_sz)
# if len(expobj.stims_af_sz) != len(on_):
#     end = expobj.stims_af_sz + [expobj.stim_start_frames[-1]]
# else:
#     end = expobj.stims_af_sz
# print(f'\n\t\- seizure start frames: {on_} [{len(on_)}]')
# print(f'\t\- seizure end frames: {end} [{len(end)}]\n')

sz_num = 0
for on, off in zip(on_, _end):
    stims_of_interest = [stim for stim in expobj.stim_start_frames if on < stim < off if
                         stim != expobj.stims_in_sz[0]]
    # stims_of_interest_ = [stim for stim in stims_of_interest if expobj._sz_wavefront_stim(stim=stim)]
    # expobj.stims_sz_wavefront.append(stims_of_interest_)

    print(f"{stims_of_interest} [1]")

    for _, stim in enumerate(stims_of_interest):
        print(f"\t{stim} [2]")
        if os.path.exists(expobj.sz_border_path(stim=stim)):
            xline, yline = pj.xycsv(csvpath=expobj.sz_border_path(stim=stim))
            expobj.stimsSzLocations.loc[stim, :] = [sz_num, [xline[0], yline[0]], [xline[1], yline[1]], None]

            j = expobj._close_to_edge(tuple(yline))
            expobj.stimsSzLocations.loc[stim, 'wavefront_in_frame'] = j
        else:
            print(f'\t\tno csv coords for stim: {stim}')

    sz_num += 1