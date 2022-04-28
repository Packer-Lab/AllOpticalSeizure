import os
from typing import Union

from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_ import alloptical_plotting as aoplot
from _utils_._anndata import create_anndata_SLMtargets


def run_photostim_preprocessing(trial, exp_type, tiffs_loc, naparms_loc, paqs_loc, metainfo,
                                new_tiffs, matlab_pairedmeasurements_path=None, processed_tiffs=True, discard_all=False,
                                quick=False, analysis_save_path=''):
    print('----------------------------------------')
    print('-----Processing trial # %s------' % trial)
    print('----------------------------------------\n')

    os.makedirs(analysis_save_path, exist_ok=True)

    paths = {'tiffs_loc': tiffs_loc,
        'naparms_loc': naparms_loc,
        'paqs_loc': paqs_loc,
        'analysis_save_path': analysis_save_path,
        'matlab_pairedmeasurement_path': matlab_pairedmeasurements_path
             }

    # print(paths)

    # paths = [[tiffs_loc, naparms_loc, paqs_loc, analysis_save_path, matlab_pairedmeasurements_path]]
    # print(
    #     'tiffs_loc, '
    #     'naparms_loc, '
    #     'paqs_loc, '
    #     'analysis_save_path paths, and '
    #     'matlab_pairedmeasurement_path:\n',
    #     paths)
    # for path in paths[0]:  # check that all paths required for processing run are legit and active
    #     if path is not None:
    #         try:
    #             assert os.path.exists(path)
    #         except AssertionError:
    #             print('we got an invalid path at: ', path)

    if 'post' in exp_type and '4ap' in exp_type:
        expobj = Post4ap(paths, metainfo=metainfo, stimtype='2pstim', discard_all=discard_all)
    else:
        expobj = alloptical(paths, metainfo=metainfo, stimtype='2pstim', quick=quick)

    # for key, values in vars(expobj).items():
    #     print(key)

    # # these functions are moved to the alloptical class init() location
    # expobj._parseNAPARMxml()
    # expobj._parseNAPARMgpl()
    # expobj._parsePVMetadata()
    # expobj.stimProcessing(stim_channel='markpoints2packio')
    # expobj._findTargets()
    # expobj.find_photostim_frames()

    # # collect information about seizures
    # if 'post' in exp_type and '4ap' in exp_type:
    #     expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path, discard_all=discard_all)

    if expobj.bad_frames:
        print('***  Collected a total of ', len(expobj.bad_frames),
              'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy  ***')

    # if matlab_pairedmeasurements_path is not None or discard_all is True:
    #     paq = paq_read(file_path=paqs_loc, plot=False)
    #     # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
    #     bad_frames, expobj.seizure_frames, _, _ = \
    #         frames_discard(paq=paq[0], input_array=matlab_pairedmeasurements_path,
    #                        total_frames=expobj.n_frames, discard_all=discard_all)
    #     print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
    #     print('|\n -- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])
    #     expobj.append_bad_frames(
    #         bad_frames=bad_frames)  # here only need to append the bad frames to the expobj.bad_frames property
    #
    # else:
    #     expobj.seizure_frames = []
    #     print('\nNo additional bad (seizure) frames needed for', tiffs_loc_dir)
    #
    # if len(expobj.bad_frames) > 0:
    #     print('***Saving a total of ', len(expobj.bad_frames),
    #           'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy***')
    #     np.save('%s/bad_frames.npy' % tiffs_loc_dir,
    #             expobj.bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p

    # Pickle the expobject output to save it for analysis

    # with open(save_path, 'wb') as f:
    #     pickle.dump(expobj, f)
    # print("\nPkl saved to %s" % save_path)

    # make processed tiffs
    if processed_tiffs:
        expobj.rm_artifacts_tiffs(expobj, tiffs_loc=tiffs_loc, new_tiffs=new_tiffs)


    # MAKE AVG STIM IMAGES AROUND EACH PHOTOSTIM TIMINGS
    expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=False, save_img=True)

    print('\n----- COMPLETED RUNNING run_photostim_preprocessing() *******')
    print(metainfo)
    expobj.save()

    return expobj


def run_alloptical_processing_photostim(expobj: Union[alloptical, Post4ap], to_suite2p=None, baseline_trials=None, plots: bool = True,
                                        force_redo: bool = False):
    """
    main function for running processing photostim trace data collecting (e.g. dFF photostim trials pre- post stim).

    :type expobj: Union[alloptical, Post4ap] object
    :param expobj: experimental object (usually from pkl file)
    :param to_suite2p: trials that were used in the suite2p run for this expobj
    :param baseline_trials: trials that were baseline (spontaneous pre-4ap) for this expobj
    :param plots: whether to plot results of processing where sub-function calls are appropriate
    :param force_redo: bool whether to redo some functions
    :param post_stim_response_window_msec: the length of the time window post stimulation that will be used for measuring the response magnitude.

    :return: n/a
    """

    print(f"\nRunning alloptical_processing_photostim for {expobj.t_series_name} ------------------------------")

    if force_redo:
        expobj._findTargetsAreas()

        if not hasattr(expobj, 'meanRawFluTrace'):
            expobj.mean_raw_flu_trace(plot=True)

        if plots:
            aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
            # aoplot.plotLfpSignal(expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
            aoplot.plot_SLMtargets_Locs(expobj)
            aoplot.plot_lfp_stims(expobj)

        # prep for importing data from suite2p for this whole experiment
        ####################################################################################################################

        if not hasattr(expobj, 'suite2p_trials'):
            if to_suite2p is None:
                AttributeError(
                    'need to provide which trials were used in suite2p for this expobj, the attr. hasnt been set')
            if baseline_trials is None:
                AttributeError(
                    'need to provide which trials were baseline (spont imaging pre-4ap) for this expobj, the attr. hasnt been set')
            expobj.suite2p_trials = to_suite2p
            expobj.baseline_trials = baseline_trials
            expobj.save()

        # determine which frames to retrieve from the overall total s2p output
        expobj.subset_frames_current_trial(trial=expobj.metainfo['trial'], to_suite2p=expobj.suite2p_trials,
                                           baseline_trials=expobj.baseline_trials, force_redo=force_redo)

        ####################################################################################################################
        # collect raw Flu data from SLM targets
        expobj.collect_traces_from_targets(force_redo=force_redo)

        #####
        if 'post' in expobj.exptype:
            expobj.MeanSeizureImages(frames_last=1000)

    if plots:
        aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg, title=f'SLM targets location w/ mean Flu img - {expobj.t_series_name}')
        aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered,
                                    title=f'SLM targets location w/ registered mean Flu img - {expobj.t_series_name}')

    # # collect SLM photostim individual targets -- individual, full traces, dff normalized
    # expobj.dff_SLMTargets = normalize_dff(np.array(expobj.raw_SLMTargets))
    # expobj.save()

    # collect and plot peri- photostim traces for individual SLM target, incl. individual traces for each stim
    # all stims (use for pre-4ap trials)
    if 'pre' in expobj.metainfo['exptype']:
        # Collecting stim trace snippets of SLM targets
        expobj.SLMTargets_stims_dff, expobj.SLMTargets_stims_dffAvg, expobj.SLMTargets_stims_dfstdF, \
        expobj.SLMTargets_stims_dfstdF_avg, expobj.SLMTargets_stims_raw, expobj.SLMTargets_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(process='trace raw', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                   post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr, stims=expobj.stim_start_frames)

        expobj.SLMTargets_tracedFF_stims_dff, expobj.SLMTargets_tracedFF_stims_dffAvg, expobj.SLMTargets_tracedFF_stims_dfstdF, \
        expobj.SLMTargets_tracedFF_stims_dfstdF_avg, expobj.SLMTargets_tracedFF_stims_raw, expobj.SLMTargets_tracedFF_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                   post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr, stims=expobj.stim_start_frames)

        expobj.fake_SLMTargets_tracedFF_stims_dff, expobj.fake_SLMTargets_tracedFF_stims_dffAvg, expobj.fake_SLMTargets_tracedFF_stims_dfstdF, \
        expobj.fake_SLMTargets_tracedFF_stims_dfstdF_avg, expobj.fake_SLMTargets_tracedFF_stims_raw, expobj.fake_SLMTargets_tracedFF_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=expobj.pre_stim,
                                                 post_stim=expobj.post_stim, stims=expobj.fake_stim_start_frames)
        
        SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))

    # filtering of stims in / outside sz period (use for post-4ap trials)
    elif 'post' in expobj.metainfo['exptype']:

        # all stims
        expobj.SLMTargets_stims_dff, expobj.SLMTargets_stims_dffAvg, expobj.SLMTargets_stims_dfstdF, \
        expobj.SLMTargets_stims_dfstdF_avg, expobj.SLMTargets_stims_raw, expobj.SLMTargets_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(process='trace raw', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                   post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr,
                                                   stims=expobj.stim_start_frames)

        expobj.SLMTargets_tracedFF_stims_dff, expobj.SLMTargets_tracedFF_stims_dffAvg, expobj.SLMTargets_tracedFF_stims_dfstdF, \
        expobj.SLMTargets_tracedFF_stims_dfstdF_avg, expobj.SLMTargets_tracedFF_stims_raw, expobj.SLMTargets_tracedFF_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                   post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr,
                                                   stims=expobj.stim_start_frames)

        expobj.fake_SLMTargets_tracedFF_stims_dff, expobj.fake_SLMTargets_tracedFF_stims_dffAvg, expobj.fake_SLMTargets_tracedFF_stims_dfstdF, \
        expobj.fake_SLMTargets_tracedFF_stims_dfstdF_avg, expobj.fake_SLMTargets_tracedFF_stims_raw, expobj.fake_SLMTargets_tracedFF_stims_rawAvg = \
            expobj.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=expobj.pre_stim,
                                                 post_stim=expobj.post_stim, stims=expobj.fake_stim_start_frames)

        if hasattr(expobj, 'stims_in_sz'):
            # out of sz stims:
            stims = [stim for stim in expobj.stim_start_frames if stim not in expobj.seizure_frames]
            expobj.SLMTargets_stims_dff_outsz, expobj.SLMTargets_stims_dffAvg_outsz, expobj.SLMTargets_stims_dfstdF_outsz, \
            expobj.SLMTargets_stims_dfstdF_avg_outsz, expobj.SLMTargets_stims_raw_outsz, expobj.SLMTargets_stims_rawAvg_outsz = \
                expobj.get_alltargets_stim_traces_norm(process='trace raw', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                       post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr, stims=stims)

            expobj.SLMTargets_tracedFF_stims_dff_outsz, expobj.SLMTargets_tracedFF_stims_dffAvg_outsz, expobj.SLMTargets_tracedFF_stims_dfstdF_outsz, \
            expobj.SLMTargets_tracedFF_stims_dfstdF_avg_outsz, expobj.SLMTargets_tracedFF_stims_raw_outsz, expobj.SLMTargets_tracedFF_stims_rawAvg_outsz = \
                expobj.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                       post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr, stims=stims)

            # only in sz stims (use for post-4ap trials) - includes exclusion of cells inside of sz boundary
            stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
            expobj.SLMTargets_stims_dff_insz, expobj.SLMTargets_stims_dffAvg_insz, expobj.SLMTargets_stims_dfstdF_insz, \
            expobj.SLMTargets_stims_dfstdF_avg_insz, expobj.SLMTargets_stims_raw_insz, expobj.SLMTargets_stims_rawAvg_insz = \
                expobj.get_alltargets_stim_traces_norm(process='trace raw', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                       post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr, stims=stims, filter_sz=True)

            expobj.SLMTargets_tracedFF_stims_dff_insz, expobj.SLMTargets_tracedFF_stims_dffAvg_insz, expobj.SLMTargets_tracedFF_stims_dfstdF_insz, \
            expobj.SLMTargets_tracedFF_stims_dfstdF_avg_insz, expobj.SLMTargets_tracedFF_stims_raw_insz, expobj.SLMTargets_tracedFF_stims_rawAvg_insz = \
                expobj.get_alltargets_stim_traces_norm(process='trace dFF', pre_stim=expobj.PhotostimAnalysisSlmTargets.pre_stim_fr,
                                                       post_stim=expobj.PhotostimAnalysisSlmTargets.post_stim_fr, stims=stims, filter_sz=True)


    else:
        raise Exception('something very weird has happened. exptype for expobj not defined as pre or post 4ap [1.]')

    #### PART 2 OF THIS FUNCTION
    # photostim. SUCCESS RATE MEASUREMENTS and PLOT - SLM PHOTOSTIM TARGETED CELLS
    # measure, for each cell, the pct of trials in which the dF_stdF > 20% post stim (normalized to pre-stim avgF for the trial and cell)
    # can plot this as a bar plot for now showing the distribution of the reliability measurement
    if 'pre' in expobj.metainfo['exptype']:
        seizure_filter = False
        # dF/stdF
        expobj.StimSuccessRate_SLMtargets_dfstdf, expobj.hits_SLMtargets_dfstdf, expobj.responses_SLMtargets_dfstdf, expobj.traces_SLMtargets_successes_dfstdf = \
            expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3, stims_to_use=expobj.stim_start_frames)

        # dF/prestimF
        expobj.StimSuccessRate_SLMtargets_dfprestimf, expobj.hits_SLMtargets_dfprestimf, expobj.responses_SLMtargets_dfprestimf, expobj.traces_SLMtargets_successes_dfprestimf = \
            expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                               stims_to_use=expobj.stim_start_frames)
        # dF/stdF
        # expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
        expobj.StimSuccessRate_SLMtargets_dfstdf, expobj.traces_SLMtargets_successes_avg_dfstdf, expobj.traces_SLMtargets_failures_avg_dfstdf = \
            expobj.calculate_SLMTarget_SuccessStims(process='dF/stdF', hits_slmtargets_df=expobj.hits_SLMtargets_dfstdf,
                                                    stims_idx_l=expobj.stims_idx)
        # dF/prestimF
        expobj.StimSuccessRate_SLMtargets_dfprestimf, expobj.traces_SLMtargets_successes_avg_dfprestimf, expobj.traces_SLMtargets_failures_avg_dfprestimf = \
            expobj.calculate_SLMTarget_SuccessStims(process='dF/prestimF', hits_slmtargets_df=expobj.hits_SLMtargets_dfprestimf,
                                                    stims_idx_l=expobj.stims_idx)
        # trace dFF
        expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.hits_SLMtargets_tracedFF, expobj.responses_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes = \
            expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10, stims_to_use=expobj.stim_start_frames)
        # trace dFF
        expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes_avg, expobj.traces_SLMtargets_tracedFF_failures_avg = \
            expobj.calculate_SLMTarget_SuccessStims(process='trace dFF',
                                                    hits_slmtargets_df=expobj.hits_SLMtargets_tracedFF,
                                                    stims_idx_l=expobj.stims_idx)


        ## GET NONTARGETS TRACES - not changed yet to handle the trace dFF processing
        # expobj.dff_traces_nontargets, expobj.dff_traces_nontargets_avg, expobj.dfstdF_traces_nontargets, expobj.dfstdF_traces_nontargets_avg, expobj.raw_traces_nontargets, expobj.raw_traces_nontargets_avg = \
        #     get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
        #                                     post_stim=expobj.post_stim)

        expobj._makeNontargetsStimTracesArray(normalize_to='pre-stim')

    elif 'post' in expobj.metainfo['exptype']:
        seizure_filter = True
        print('|- calculating stim responses (all trials) - %s stims [2.2.1]' % len(expobj.stim_start_frames))
        # dF/stdF
        expobj.StimSuccessRate_SLMtargets_dfstdf, expobj.hits_SLMtargets_dfstdf, expobj.responses_SLMtargets_dfstdf, expobj.traces_SLMtargets_successes_dfstdf = \
            expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
                                               stims_to_use=expobj.stim_start_frames)
        # dF/prestimF
        expobj.StimSuccessRate_SLMtargets_dfprestimf, expobj.hits_SLMtargets_dfprestimf, expobj.responses_SLMtargets_dfprestimf, expobj.traces_SLMtargets_successes_dfprestimf = \
            expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                               stims_to_use=expobj.stim_start_frames)
        # trace dFF
        expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.hits_SLMtargets_tracedFF, expobj.responses_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes = \
            expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10, stims_to_use=expobj.stim_start_frames)

        ### STIMS OUT OF SEIZURE
        if expobj.stims_out_sz:
            stims_outsz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
            if stims_outsz_idx:
                print('|- calculating stim responses (outsz) - %s stims [2.2.1]' % len(stims_outsz_idx))
                # dF/stdF
                expobj.StimSuccessRate_SLMtargets_dfstdf_outsz, expobj.hits_SLMtargets_dfstdf_outsz, expobj.responses_SLMtargets_dfstdf_outsz, expobj.traces_SLMtargets_successes_dfstdf_outsz = \
                    expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
                                                       stims_to_use=expobj.stims_out_sz)
                # dF/prestimF
                expobj.StimSuccessRate_SLMtargets_dfprestimf_outsz, expobj.hits_SLMtargets_dfprestimf_outsz, expobj.responses_SLMtargets_dfprestimf_outsz, expobj.traces_SLMtargets_successes_dfprestimf_outsz = \
                    expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                                       stims_to_use=expobj.stims_out_sz)
                # trace dFF
                expobj.StimSuccessRate_SLMtargets_tracedFF_outsz, expobj.hits_SLMtargets_tracedFF_outsz, expobj.responses_SLMtargets_tracedFF_outsz, expobj.traces_SLMtargets_tracedFF_successes_outsz = \
                    expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10,
                                                       stims_to_use=expobj.stims_out_sz)

                print('|- calculating stim success rates (outsz) - %s stims [2.2.0]' % len(stims_outsz_idx))
                # dF/stdF
                expobj.outsz_StimSuccessRate_SLMtargets_dfstdf, expobj.outsz_traces_SLMtargets_successes_avg_dfstdf, \
                expobj.outsz_traces_SLMtargets_failures_avg_dfstdf =  expobj.calculate_SLMTarget_SuccessStims(process='dF/stdF',
                                                                                                              hits_slmtargets_df=expobj.hits_SLMtargets_dfstdf_outsz,
                                                                                                              stims_idx_l=stims_outsz_idx)
                # dF/prestimF
                expobj.outsz_StimSuccessRate_SLMtargets_dfprestimf, expobj.outsz_traces_SLMtargets_successes_avg_dfprestimf, \
                expobj.outsz_traces_SLMtargets_failures_avg_dfprestimf = expobj.calculate_SLMTarget_SuccessStims(process='dF/prestimF',
                                                                                                                 hits_slmtargets_df=expobj.hits_SLMtargets_dfprestimf_outsz,
                                                                                                                 stims_idx_l=stims_outsz_idx)
                # trace dFF
                expobj.outsz_StimSuccessRate_SLMtargets_tracedFF, expobj.outsz_traces_SLMtargets_tracedFF_successes_avg, \
                expobj.outsz_traces_SLMtargets_tracedFF_failures_avg = expobj.calculate_SLMTarget_SuccessStims(process='trace dFF',
                                                                                                               hits_slmtargets_df=expobj.hits_SLMtargets_tracedFF_outsz,
                                                                                                               stims_idx_l=stims_outsz_idx)

        ### STIMS IN SEIZURE
        if expobj.stims_in_sz:
            if hasattr(expobj, 'slmtargets_szboundary_stim'):
                stims_insz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                if stims_insz_idx:
                    print('|- calculating stim responses (insz) - %s stims [2.3.1]' % len(stims_insz_idx))
                    # dF/stdF
                    expobj.StimSuccessRate_SLMtargets_dfstdf_insz, expobj.hits_SLMtargets_dfstdf_insz, expobj.responses_SLMtargets_dfstdf_insz, expobj.traces_SLMtargets_successes_dfstdf_insz = \
                        expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
                                                           stims_to_use=expobj.stims_in_sz)
                    # dF/prestimF
                    expobj.StimSuccessRate_SLMtargets_dfprestimf_insz, expobj.hits_SLMtargets_dfprestimf_insz, expobj.responses_SLMtargets_dfprestimf_insz, expobj.traces_SLMtargets_successes_dfprestimf_insz = \
                        expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                                           stims_to_use=expobj.stims_in_sz)
                    # trace dFF
                    expobj.StimSuccessRate_SLMtargets_tracedFF_insz, expobj.hits_SLMtargets_tracedFF_insz, expobj.responses_SLMtargets_tracedFF_insz, expobj.traces_SLMtargets_tracedFF_successes_insz = \
                        expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10,
                                                           stims_to_use=expobj.stims_in_sz)

                    print('|- calculating stim success rates (insz) - %s stims [2.3.0]' % len(stims_insz_idx))
                    # dF/stdF
                    expobj.insz_StimSuccessRate_SLMtargets_dfstdf, expobj.insz_traces_SLMtargets_successes_avg_dfstdf, expobj.insz_traces_SLMtargets_failures_avg_dfstdf = \
                        expobj.calculate_SLMTarget_SuccessStims(process='dF/stdF',
                                                                hits_slmtargets_df=expobj.hits_SLMtargets_dfstdf_insz,
                                                                stims_idx_l=stims_insz_idx)
                    # dF/prestimF
                    expobj.insz_StimSuccessRate_SLMtargets_dfprestimf, expobj.insz_traces_SLMtargets_successes_avg_dfprestimf, expobj.insz_traces_SLMtargets_failures_avg_dfprestimf = \
                        expobj.calculate_SLMTarget_SuccessStims(process='dF/prestimF',
                                                                hits_slmtargets_df=expobj.hits_SLMtargets_dfprestimf_insz,
                                                                stims_idx_l=stims_insz_idx)
                    # trace dFF
                    expobj.insz_StimSuccessRate_SLMtargets_tracedFF, expobj.insz_traces_SLMtargets_tracedFF_successes_avg, expobj.insz_traces_SLMtargets_tracedFF_failures_avg = \
                        expobj.calculate_SLMTarget_SuccessStims(process='trace dFF',
                                                                hits_slmtargets_df=expobj.hits_SLMtargets_tracedFF_insz,
                                                                stims_idx_l=stims_insz_idx,
                                                                exclude_stims_targets=expobj.slmtargets_szboundary_stim)

                else:
                    print('******* No stims in sz for: %s %s' % (
                        expobj.metainfo['animal prep.'], expobj.metainfo['trial']), ' [*2.3] ')


            else:
                print('******* No slmtargets_szboundary_stim (sz boundary classification not done) for: %s %s' % (
                    expobj.metainfo['animal prep.'], expobj.metainfo['trial']), ' [*2.3] ')

        expobj.avgResponseSzStims_SLMtargets()
    create_anndata_SLMtargets(expobj)

    if plots:
        aoplot.plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dffAvg, pre_stim_sec=1.0, post_stim_sec=3.0,
                                      title=f'{expobj.t_series_name} - all stims', expobj=expobj,
                                      x_label='Time (secs)', y_label='dFF')

    expobj.save()

    print(f"\nFINISHED alloptical_processing_photostim for {expobj.metainfo['animal prep.']}, {expobj.metainfo['trial']} ------------------------------\n\n\n\n")

