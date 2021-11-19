# %% 0) IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
import seaborn as sns
import tifffile as tf

from skimage import draw

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Procesing_figs/sz_processing_boundaries_2021-11-17/'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


## TODO need to do paired measurements for seizures, and then run classifying sz boundaries for trials in ls2:
ls2 = [
    ['PS17 t-009'],
    # ['RL108 t-011'],
    # ['RL109 t-017'],
    # ['PS06 t-014'],
    # ['PS06 t-015'],
    # ['PS06 t-016']
]

#%% 1) defining trials to run for analysis

# this list should line up with the analysis list for post4ap trials
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


# %% 2.1) classification of cells in/out of sz boundary

trials_without_flip_stims = []

# using post4ap experiments from allopticalResults attr. in for loop for processing:

# for i in allopticalResults.post_4ap_trials:
for i in ls2:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[-1]
        # j = -1
        # prep = 'RL109'
        # trial = 't-016'
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
            print(f"|-- expobj {prep} {trial} DOES NOT have not_flip_stims made")
            trials_without_flip_stims.append(f"{prep} {trial}")
            break  # move on to the next for loop instance without plotting this expobj
        else:
            print(f"\-expobj.not_flip_stims: {expobj.not_flip_stims}")

        # break

        print(' \nworking on classifying cells for stims start frames...')
        expobj.slmtargets_szboundary_stim = {}
        expobj.s2prois_szboundary_stim = {}

        ######## - all stims in sz are classified, with individual sz events labelled

        stims_of_interest = [stim for stim in expobj.stims_in_sz[1:]]
        print(' \-all stims in seizures: \n \-', stims_of_interest)
        nrows = len(stims_of_interest) // 4 + 1
        if nrows == 1:
            nrows += 1
        ncols = 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
        counter = 0

        sz_num = 0
        for on, off in zip(on_, end):
            stims_of_interest = [stim for stim in expobj.stim_start_frames if on < stim < off if stim != expobj.stims_in_sz[0]]
            print(' \-- working sz # %s with stims: \n \---' % (sz_num), stims_of_interest)

            for stim in stims_of_interest:
                print(f"considering stim # {stim}")

                ax = axs[counter // ncols, counter % ncols]

                sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
                if os.path.exists(sz_border_path):
                    # first round of classifying (dont flip any cells over) - do this in the second round
                    if stim not in expobj.not_flip_stims:
                        flip = False
                    else:
                        flip = True

                    # classification of suite2p ROIs relative to sz boundary
                    in_sz, out_sz, fig, ax = expobj.classify_cells_sz_bound(sz_border_path, stim=stim, to_plot=True,
                                                                            flip=flip, fig=fig, ax=ax, text='sz %s stim %s' % (sz_num, stim))
                    expobj.s2prois_szboundary_stim[stim] = in_sz
                    # classification of SLM targets relative to sz boundary
                    in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(sz_border_path, stim=stim, to_plot=True, title='%s' % stim, flip=flip, fig=fig, ax=ax)
                    expobj.slmtargets_szboundary_stim[stim] = in_sz  # for each stim, there will be a ls of cells that will be classified as in seizure or out of seizure

                    axs[counter // ncols, counter % ncols] = ax
                    counter += 1
                else:
                    print(f"sz border path doesn't exist for stim {stim}: {sz_border_path}")

            sz_num += 1

        fig.suptitle('%s %s - Avg img around stims during- all stims' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']), y=0.995)
        save_path_full = f"{save_path_prefix}/{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} {sz_num} events.png"
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


# expobj.not_flip_stims = [expobj.stims_in_sz[1:]]  # specify here the stims where the flip=False leads to incorrect assignment
expobj.not_flip_stims = expobj.stims_in_sz[1:]  # specify here the stims where the flip=False leads to incorrect assignment

expobj.save()

# %% 2.1.2) re-run with new flip stims
expobj.slmtargets_szboundary_stim = {}
sz_num = 0
for on, off in zip(on_, end):
    stims_of_interest = [stim for stim in expobj.stim_start_frames if on < stim < off if stim != expobj.stims_in_sz[0]]
    print('|-', stims_of_interest)

    nrows = len(stims_of_interest) // 4 + 1
    if nrows == 1:
        nrows += 1
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    counter = 0

    for stim in stims_of_interest:
        ax = axs[counter // ncols, counter % ncols]

        sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path[:-17], expobj.metainfo['date'], trial, stim)
        if not os.path.exists(sz_border_path):
            print(sz_border_path)
        # first round of classifying (dont flip any cells over) - do this in the second round
        if stim not in expobj.not_flip_stims:
            flip = False
        else:
            flip = True

        in_sz, out_sz, fig, ax = expobj.classify_slmtargets_sz_bound(sz_border_path, stim=stim, to_plot=True, title='%s' % stim, flip=flip, fig=fig, ax=ax)
        expobj.slmtargets_szboundary_stim[stim] = in_sz  # for each stim, there will be a ls of cells that will be classified as in seizure or out of seizure

        axs[counter // ncols, counter % ncols] = ax
        counter += 1
    fig.suptitle('%s %s - Avg img around stims during sz - seizure # %s' % (
    expobj.metainfo['animal prep.'], expobj.metainfo['trial'], sz_num + 1), y=0.995)
    fig.show()
    sz_num += 1


# %% archive-3) responses of targets during seizures, but EXCLUDE STIMS WHERE THE CELL IS INSIDE THE SZ BOUND -- should this be in ARCHIVEDalloptical_results_photostim.py???? -- yeah most of this code should probably be retired - and current location is in alloptical_results_photostim_SLMtargets.py

if hasattr(expobj, 'stims_in_sz'):

    # stims during sz
    stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
    if len(stims) > 0:
        expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
            aoutils.get_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=stims)

    # stims outside sz
    stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
    if len(stims) > 0:
        expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
            aoutils.get_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=stims)


else:
    expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
        aoutils.get_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=None)

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
for row in expobj.responses_SLMtargets.index:
    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
    responses = [expobj.responses_SLMtargets.loc[row][stim] for stim in expobj.stim_start_frames]
    targets_avgresponses_stims_presz[row] = np.mean(responses)


data = [list(targets_avgresponses_exclude_stims_sz.values()), list(targets_avgresponses_stims_outsz.values()), list(targets_avgresponses_stims_presz.values())]
pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ', figsize=(5,5), fill_color=['orange', 'skyblue', 'green'],
                     colors=['black'] * 3, legend_labels=[None] * 3)


# %% 4) PLOT - stim frames for figures with SLM targets inside (yellow) and outside (green) of seizure boundary

prep = 'RL108'
trial = 't-013'
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
# aoplot.plot_lfp_stims(expobj, xlims=[0.2e7, 1.0e7], linewidth=1.0)
# aoplot.plotLfpSignal(expobj, downsample=True, figsize=(6,2), x_axis='Time', xlims=[120 * expobj.paq_rate, 480 * expobj.paq_rate],
#                      ylims=[-6, 2], color='slategray', stim_span_color='green', alpha=0.1)

# %%
sz_num = 2
stims_to_plot = [stim for stim in expobj.stim_start_frames if expobj.seizure_lfp_offsets[sz_num] > stim > expobj.seizure_lfp_onsets[sz_num]]

fig, axs = plt.subplots(figsize=[5*len(stims_to_plot), 5], nrows=1, ncols=len(stims_to_plot))
for i in range(len(stims_to_plot)):

    ax = axs[i]
    stim = stims_to_plot[i]
    # fig, ax = plt.subplots(figsize=[5, 5], nrows=1, ncols=1)
    # stim = stims_to_plot[0]

    # plot SLM targets in sz boundary
    coords_to_plot = [expobj.target_coords_all[cell] for cell in expobj.slmtargets_szboundary_stim[stim]]
    # read in avg stim image to use as the background
    avg_stim_img_path = '%s/%s_%s_stim-%s.tif' % (expobj.analysis_save_path + 'avg_stim_images', expobj.metainfo['date'], expobj.metainfo['trial'], stim)
    bg_img = tf.imread(avg_stim_img_path)
    title=f"{prep} {trial} {stim} - SLM targets"
    for (x, y) in coords_to_plot:
        ax.scatter(x=x, y=y, edgecolors='red', facecolors='none', linewidths=2.5, zorder=4)
    cells_to_plot = expobj.s2prois_szboundary_stim[stim]
    cells_to_plot2 = [cell for cell in expobj.cell_id if cell not in expobj.s2prois_szboundary_stim[stim]]
    fig, ax = aoplot.plot_cells_loc(expobj, cells=cells_to_plot, show_s2p_targets=False, fig=fig, ax=ax, show=False, scatter_only=True)
    fig, ax = aoplot.plot_cells_loc(expobj, cells=cells_to_plot2, show_s2p_targets=False, fig=fig, ax=ax, show=False, scatter_only=True,
                                    edgecolor='gray')
    ax.imshow(bg_img, cmap='Greys_r', zorder=0)
    ax.set_title(title)
    ax.set_xticks(ticks=[])
    ax.set_xticklabels([])
    ax.set_yticks(ticks=[])
    ax.set_yticklabels([])
fig.tight_layout(pad=2)
fig.show()



# %%
# plots for SLM targets responses
def slm_targets_responses(expobj, experiment, trial, y_spacing_factor=2, figsize=[20, 20], smooth_overlap_traces=5, linewidth_overlap_traces=0.2, dff_threshold=0.15,
                          y_lims_periphotostim_trace=[-0.5, 2.0], v_lims_periphotostim_heatmap=[-5, 5], save_results=True, force_redo=False, cmap=None):
    # plot SLM photostim individual targets -- individual, full traces, dff normalized

    # make rolling average for these plots to smooth out the traces a little more

    # force_redo = False
    if force_redo:
        # expobj._findTargets()
        # expobj.raw_traces_from_targets(force_redo=force_redo, save=True)
        # expobj.save()

        if hasattr(expobj, 'stims_in_sz'):
            seizure_filter = True

            stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
            # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
            if len(stims) > 0:
                expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=dff_threshold, stims_to_use=stims)

                # expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_hits_SLMtargets, expobj.outsz_responses_SLMtargets = \
                #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=raw_traces_stims,
                #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
                #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
                #                               verbose=True, plot=False)

            stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
            # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
            if len(stims) > 0:
                expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=dff_threshold, stims_to_use=stims)

                # expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_hits_SLMtargets, expobj.insz_responses_SLMtargets = \
                #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=raw_traces_stims,
                #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
                #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
                #                               verbose=True, plot=False)

        else:
            seizure_filter = False
            print('\n Calculating stim success rates and response magnitudes ***********')
            expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
                aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=dff_threshold, stims_to_use=expobj.stim_start_frames)

            # expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
            #     calculate_StimSuccessRate(expobj, cell_ids=SLMtarget_ids, raw_traces_stims=expobj.SLMTargets_stims_raw,
            #                               dff_threshold=10, post_stim_response_frames_window=expobj.post_stim_response_frames_window,
            #                               pre_stim_sec=expobj.pre_stim_sec, sz_filter=seizure_filter,
            #                               verbose=True, plot=False)

        expobj.save()


    ####################################################################################################################
    w = smooth_overlap_traces
    to_plot = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets])
    # to_plot = expobj.dff_SLMTargets
    # aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Time (secs.)',
    #                              y_label='dFF Flu', title=experiment)


    # initialize figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(4, 8)

    ax9 = fig.add_subplot(gs[0, 0])
    fig, ax9 = aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered, title=None, fig=fig, ax=ax9, show=False)


    ax0 = fig.add_subplot(gs[0, 1:])
    ax0 = aoplot.plot_lfp_stims(expobj, fig=fig, ax=ax0, show=False, x_axis='Time (secs.)')

    ax1 = fig.add_subplot(gs[1:3, :])
    aoplot.plot_photostim_traces_overlap(array=expobj.dff_SLMTargets, expobj=expobj, x_axis='Time (secs.)',
                                         y_spacing_factor=y_spacing_factor, fig=fig, ax=ax1, show=False,
                                         title='%s - dFF Flu photostims' % experiment, linewidth=linewidth_overlap_traces,
                                         figsize=(2 * 20, 2 * len(to_plot) * 0.15))

    ax2 = fig.add_subplot(gs[-1, 0:2])
    y_label = 'dF/F'
    cell_avg_stim_traces = expobj.SLMTargets_stims_dffAvg
    aoplot.plot_periphotostim_avg(arr=cell_avg_stim_traces, expobj=expobj,
                                  stim_duration=expobj.stim_duration_frames,
                                  figsize=[5, 4], y_lims=y_lims_periphotostim_trace, fig=fig, ax=ax2, show=False,
                                  title=('responses of all photostim targets'),
                                  y_label=y_label, x_label='Time post-stimulation (seconds)')

    # fig.show()

    if hasattr(expobj, 'stims_in_sz'):

        # make response magnitude and response success rate figure
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=((5 * 4), 5), nrows=1, ncols=4)
        # stims out sz
        ax3 = fig.add_subplot(gs[-1, 2:4])
        data = [[np.mean(expobj.outsz_responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        fig, ax3 = pj.plot_hist_density(data, x_label='response magnitude (dF/F)', title='stims_out_sz - ',
                                     fig=fig, ax=ax3, show=False)
        ax4 = fig.add_subplot(gs[-1, 4])
        fig, ax4 = pj.plot_bar_with_points(data=[list(expobj.outsz_StimSuccessRate_SLMtargets.values())],
                                           x_tick_labels=[trial],
                                           ylims=[0, 100], bar=False, y_label='% success stims.',
                                           title='target success rate (stims out sz)', expand_size_x=2,
                                           show=False, fig=fig, ax=ax4)
        # stims in sz
        ax5 = fig.add_subplot(gs[-1, 5:7])
        data = [[np.mean(expobj.insz_responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        fig, ax5 = pj.plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='stims_in_sz - ',
                                        fig=fig, ax=ax5, show=False)
        ax6 = fig.add_subplot(gs[-1, 7])
        fig, ax6 = pj.plot_bar_with_points(data=[list(expobj.insz_StimSuccessRate_SLMtargets.values())],
                                        x_tick_labels=[trial],
                                        ylims=[0, 100], bar=False, y_label='% success stims.',
                                        title='target success rate (stims in sz)', expand_size_x=2,
                                        show=False, fig=fig, ax=ax6)
        fig.tight_layout()
        if save_results:
            save = expobj.analysis_save_path[:-17] + '/results/' + '%s_%s_slm_targets_responses_dFF' % (expobj.metainfo['animal prep.'], trial)
            if not os.path.exists(expobj.analysis_save_path[:-17] + '/results'):
                os.makedirs(expobj.analysis_save_path[:-17] + '/results')
            print('saving png and svg to: %s' % save)
            fig.savefig(fname=save + '.png', transparent=True, format='png')
            fig.savefig(fname=save + '.svg', transparent=True, format='svg')

        fig.show()

    else:
        # no sz
        # fig, (ax1, ax2) = plt.subplots(figsize=((5 * 2), 5), nrows=1, ncols=2)
        data = [[np.mean(expobj.responses_SLMtargets.loc[i]) for i in range(expobj.n_targets_total)]]
        ax3 = fig.add_subplot(gs[-1, 2:4])
        fig, ax3 = pj.plot_hist_density(data, x_label='response magnitude (dF/F)', title='no sz', show=False, fig=fig, ax=ax3)
        ax4 = fig.add_subplot(gs[-1, 4])
        fig, ax4 = pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                                           ylims=[0, 100], bar=False, show=False, fig=fig, ax=ax4,
                                           y_label='% success stims.', title='target success rate (pre4ap)',
                                           expand_size_x=2)

        zero_point = abs(v_lims_periphotostim_heatmap[0]/v_lims_periphotostim_heatmap[1])
        c = ColorConverter().to_rgb
        if cmap is None:
            cmap = pj.make_colormap([c('blue'), c('white'), zero_point - 0.20, c('white'), c('red')])
        ax5 = fig.add_subplot(gs[-1, 5:])
        fig, ax5 = aoplot.plot_traces_heatmap(data=cell_avg_stim_traces, expobj=expobj, vmin=v_lims_periphotostim_heatmap[0], vmax=v_lims_periphotostim_heatmap[1],
                                              stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames + 1, cbar=False,
                                              title=(expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo[
                                              'trial'] + ' - SLM targets raw Flu'), show=False, fig=fig, ax=ax5, x_label='Frames', y_label='Neurons',
                                              xlims=(0, expobj.pre_stim + expobj.stim_duration_frames + expobj.post_stim),
                                              cmap=cmap)

        fig.tight_layout()
        if save_results:
            save = expobj.analysis_save_path[:-17] + '/results/' + '%s_%s_slm_targets_responses_dFF' % (expobj.metainfo['animal prep.'], trial)
            if not os.path.exists(expobj.analysis_save_path[:-17] + '/results'):
                os.makedirs(expobj.analysis_save_path[:-17] + '/results')
            print('saving png and svg to: %s' % save)
            fig.savefig(fname=save+'.png', transparent=True, format='png')
            fig.savefig(fname=save+'.svg', transparent=True,  format='svg')

        fig.show()

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
    # look at all post4ap trials in expobj
    if 'post-4ap trials' in expobj.metainfo.keys():
        a = 'post-4ap trials'
    elif 'post4ap_trials' in expobj.metainfo.keys():
        a = 'post4ap_trials'
    # for loop over all of those post4ap trials
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
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
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
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
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

# %% 4.1) counting seizure incidence across all imaging trials

for key in list(allopticalResults.trial_maps['post'].keys()):
    # import initial expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.0' % key, verbose=False)
    prep = expobj.metainfo['animal prep.']
    # look at all post4ap trials in expobj
    if 'post-4ap trials' in expobj.metainfo.keys():
        a = 'post-4ap trials'
    elif 'post4ap_trials' in expobj.metainfo.keys():
        a = 'post4ap_trials'
    # for loop over all of those post4ap trials
    for trial in expobj.metainfo['%s' % a]:
        # import expobj
        expobj, experiment = aoutils.import_expobj(prep=prep, trial=trial, verbose=False)
        total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

        # count seizure incidence (avg. over mins) for each experiment (animal)
        if hasattr(expobj, 'seizure_lfp_onsets'):
            n_seizures = len(expobj.seizure_lfp_onsets)
        else:
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
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
            expobj.seizure_frames = []
            expobj.seizure_lfp_onsets = []
            expobj.seizure_lfp_offsets = []
            expobj.save()
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



