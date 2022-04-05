from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _analysis_._utils import Quantification
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

# %% function definitions from alloptical_utils_pplot file - 04/05/22
###### NON TARGETS analysis + plottings
def run_allopticalAnalysisNontargets(expobj, normalize_to='pre_stim', to_plot=True, save_plot_suffix='',
                                     do_processing=True):
    if do_processing:

        # set the correct test response windows
        if not expobj.pre_stim == int(1.0 * expobj.fps):
            print('updating expobj.pre_stim_sec to 1 sec')
            expobj.pre_stim = int(1.0 * expobj.fps)  # length of pre stim trace collected (in frames)
            expobj.post_stim = int(3.0 * expobj.fps)  # length of post stim trace collected (in frames)
            expobj.post_stim_response_window_msec = 500  # msec
            expobj.post_stim_response_frames_window = int(
                expobj.fps * expobj.post_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)
            expobj.pre_stim_response_window_msec = 500  # msec
            expobj.pre_stim_response_frames_window = int(
                expobj.fps * expobj.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)

        expobj._trialProcessing_nontargets(normalize_to, save=False)
        expobj.sig_units = expobj._sigTestAvgResponse_nontargets(p_vals=expobj.wilcoxons, alpha=0.1, save=False)

        expobj.save()

    # make figure containing plots showing average responses of nontargets to photostim
    # save_plot_path = expobj.analysis_save_path[:30] + 'Results_figs/' + save_plot_suffix
    fig_non_targets_responses(expobj=expobj, plot_subset=False, save_fig_suffix=save_plot_suffix) if to_plot else None

    print('\n** FIN. * allopticalAnalysisNontargets * %s %s **** ' % (
        expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    print(
        '-------------------------------------------------------------------------------------------------------------\n\n')


def fig_non_targets_responses(expobj, plot_subset: bool = True, save_fig_suffix=None):
    print('\n----------------------------------------------------------------')
    print('plotting nontargets responses ')
    print('----------------------------------------------------------------')

    if plot_subset:
        selection = np.random.randint(0, expobj.dff_traces_nontargets_avg.shape[0], 100)
    else:
        selection = np.arange(expobj.dff_traces_nontargets_avg.shape[0])

    #### SUITE2P NON-TARGETS - PLOTTING OF AVG PERI-PHOTOSTIM RESPONSES
    if sum(expobj.sig_units) > 0:
        f = plt.figure(figsize=[25, 10])
        gs = f.add_gridspec(2, 9)
    else:
        f = plt.figure(figsize=[25, 5])
        gs = f.add_gridspec(1, 9)

    # PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
    from _utils_ import alloptical_plotting as aoplot

    a1 = f.add_subplot(gs[0, 0:2])
    x = expobj.dff_traces_nontargets_avg[selection]
    y_label = 'pct. dFF (normalized to prestim period)'
    aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                                  title='Average photostim all trials response', y_label=y_label, fig=f, ax=a1,
                                  show=False,
                                  x_label='Time (seconds)', y_lims=[-50, 200])
    # PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
    a2 = f.add_subplot(gs[0, 2:4])
    x = expobj.dfstdF_traces_avg[selection]
    y_label = 'dFstdF (normalized to prestim period)'
    aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                                  title='Average photostim all trials response', y_label=y_label, fig=f, ax=a2,
                                  show=False,
                                  x_label='Time (seconds)', y_lims=[-1, 3])
    # PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/f
    a3 = f.add_subplot(gs[0, 4:6])
    vmin = -1
    vmax = 1
    aoplot.plot_traces_heatmap(arr=expobj.dfstdF_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax,
                               stim_on=int(1 * expobj.fps),
                               stim_off=int(1 * expobj.fps + expobj.stim_duration_frames),
                               xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                               title='dF/stdF heatmap for all nontargets', x_label='Time', cbar=True,
                               fig=f, ax=a3, show=False)
    # PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/stdf
    a4 = f.add_subplot(gs[0, -3:-1])
    vmin = -100
    vmax = 100
    aoplot.plot_traces_heatmap(arr=expobj.dff_traces_nontargets_avg, expobj=expobj, vmin=vmin, vmax=vmax,
                               stim_on=int(1 * expobj.fps),
                               stim_off=int(1 * expobj.fps + expobj.stim_duration_frames),
                               xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                               title='dF/F heatmap for all nontargets', x_label='Time', cbar=True,
                               fig=f, ax=a4, show=False)
    # bar plot of avg post stim response quantified between responders and non-responders
    a04 = f.add_subplot(gs[0, -1])
    sig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)
    nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
    data = np.asarray([sig_responders_avgresponse, nonsig_responders_avgresponse])
    pplot.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'gray'],
                            y_label='avg dF/stdF', bar=False,
                            text_list=['%s pct' % (np.round(
                                (len(sig_responders_avgresponse) / expobj.post_array_responses.shape[0]), 2) * 100),
                                       '%s pct' % (np.round(
                                           (len(nonsig_responders_avgresponse) / expobj.post_array_responses.shape[0]),
                                           2) * 100)],
                            text_y_pos=1.43, text_shift=1.7, x_tick_labels=['significant', 'non-significant'],
                            ylims=[-2, 3],
                            expand_size_y=1.5, expand_size_x=0.6,
                            fig=f, ax=a04, show=False)

    ## PLOTTING STATISTICALLY SIGNIFICANT RESPONDERS
    if sum(expobj.sig_units) > 0:
        # plot PERI-STIM AVG TRACES of sig nontargets
        a10 = f.add_subplot(gs[1, 0:2])
        x = expobj.dfstdF_traces_avg[expobj.sig_units]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a10, show=False,
                                      title='significant responders', y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of nonsig nontargets
        a11 = f.add_subplot(gs[1, 2:4])
        x = expobj.dfstdF_traces_avg[~expobj.sig_units]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a11, show=False,
                                      title='non-significant responders',
                                      y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of sig. positive responders
        a12 = f.add_subplot(gs[1, 4:6])
        x = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a12, show=False,
                                      title='positive signif. responders',
                                      y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of sig. negative responders
        a13 = f.add_subplot(gs[1, -3:-1])
        x = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a13, show=False,
                                      title='negative signif. responders',
                                      y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # bar plot of avg post stim response quantified between responders and non-responders
        a14 = f.add_subplot(gs[1, -1])
        possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
        data = np.asarray([possig_responders_avgresponse, negsig_responders_avgresponse, nonsig_responders_avgresponse])
        pplot.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells',
                                colors=['green', 'blue', 'gray'],
                                y_label='avg dF/stdF', bar=False,
                                text_list=['%s pct' % (np.round(
                                    (len(possig_responders_avgresponse) / expobj.post_array_responses.shape[0]) * 100,
                                    1)),
                                           '%s pct' % (np.round((len(negsig_responders_avgresponse) /
                                                                 expobj.post_array_responses.shape[0]) * 100, 1)),
                                           '%s pct' % (np.round((len(nonsig_responders_avgresponse) /
                                                                 expobj.post_array_responses.shape[0]) * 100, 1))],
                                text_y_pos=1.43, text_shift=1.2, ylims=[-2, 3],
                                x_tick_labels=['pos. significant', 'neg. significant', 'non-significant'],
                                expand_size_y=1.5, expand_size_x=0.5,
                                fig=f, ax=a14, show=False)

    f.suptitle(
        ('%s %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'])))
    f.tight_layout()
    f.show()

    Utils.save_figure(f, save_fig_suffix) if save_fig_suffix is not None else None
    # _path = save_fig_suffix[:[i for i in re.finditer('/', save_fig_suffix)][-1].end()]
    # os.makedirs(_path) if not os.path.exists(_path) else None
    # print('saving figure output to:', save_fig_suffix)
    # plt.savefig(save_fig_suffix)

######

class PhotostimResponsesQuantificationNonTargets(Quantification):
    """class for quanitying responses of non-targeted cells at photostimulation trials.
    non-targeted cells classified as Suite2p ROIs that were not SLM targets.


    Goals:
    [ ] scatter plot of individual stim trials: response magnitude of targets vs. response magnitude of all nontargets
                                                                                - (maybe, response magnitude of sig. responders only?)
                                                                                - (maybe, summed magnitude of sig. responders only? - check previous plots on this to see if there's some insights there....)

    """

    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationNonTargets.pkl'
    mean_photostim_responses_baseline: List[float] = None
    mean_photostim_responses_interictal: List[float] = None
    mean_photostim_responses_ictal: List[float] = None

    def __init__(self, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimResponsesNonTargets MODULE to expobj: {expobj.t_series_name}')

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
    def run__initPhotostimResponsesQuantificationNonTargets(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.PhotostimResponsesNonTargets = PhotostimResponsesQuantificationNonTargets(expobj=expobj)
        expobj.save()

    def __repr__(self):
        return f"PhotostimResponsesNonTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # 0) ANALYSIS OF NON-TARGETS IN ALL OPTICAL EXPERIMENTS.

    def run_allopticalAnalysisNontargets(self, expobj: Union[alloptical, Post4ap], normalize_to='pre_stim', to_plot=True, save_plot_suffix='',
                                         do_processing=True):
        if do_processing:

            self.responses, self.wilcoxons = expobj._trialProcessing_nontargets(normalize_to,
                                                                                stims='all', save=False)
            self.sig_units = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1, save=False)

            expobj.save()

        # # make figure containing plots showing average responses of nontargets to photostim
        # # save_plot_path = expobj.analysis_save_path[:30] + 'Results_figs/' + save_plot_suffix
        # fig_non_targets_responses(expobj=expobj, plot_subset=False,
        #                           save_fig_suffix=save_plot_suffix) if to_plot else None

        print('\n** FIN. * allopticalAnalysisNontargets * %s %s **** ' % (
            expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
        print(
            '-------------------------------------------------------------------------------------------------------------\n\n')

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False)
    def run_allopticalNontargets(**kwargs):
        expobj = kwargs['expobj']
        if not hasattr(expobj, 's2p_nontargets'):
            expobj._parseNAPARMgpl()
            expobj._findTargetsAreas()
            expobj._findTargetedS2pROIs(force_redo=True, plot=False)
            expobj.save()

        run_allopticalAnalysisNontargets(expobj=expobj, normalize_to='pre-stim', do_processing=True,
                                                 to_plot=True,
                                                 save_plot_suffix=f"{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-pre4ap.png")

    run_allopticalNontargets()
