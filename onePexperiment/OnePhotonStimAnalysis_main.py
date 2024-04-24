import os
from typing import Any

import numpy as np
import pandas as pd
from funcsforprajay import funcs as pj
from funcsforprajay.stats import tukey_hsd
from funcsforprajay.plotting.plotting import plot_bar_with_points
from scipy import stats

from _analysis_._utils import Results
from _exp_metainfo_.data_paths import SAVE_LOC
from _exp_metainfo_.exp_metainfo import fontsize_extraplot, interictal_color, baseline_color
from onePexperiment.OnePhotonStimMain import OnePhotonStim, onePresults



# %% ANALYSIS FUNCTIONS

class OnePhotonStimResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__OnePhotonStim.pkl'

    # by experiments:
    expids = ['PS07', 'PS11', 'PS18', 'PS09', 'PS16']

    interictal_decay_constant = {
        'PS07': [1.26471, 0.32844],
        'PS11': [1.05109, 0.45986],
        'PS18': [0.65681],
        'PS09': [0.52549, 0.59118, 0.39412],
        'PS16': [0.78811, 0.98517]
    }

    baseline_decay_constant = {
        # 'PS17': [0.1686],
        'PS07': [0.45972, 0.46595],
        'PS11': [0.45980, 0.52553],
        'PS18': [0.45975],
        'PS09': [0.52546],
        'PS16': [0.45964, 0.39399, 0.46595]
    }

    # by experiments:
    interictal_response_magnitude = {
        'PS07': [0.2654, 0.1296],
        'PS11': [0.9005, 0.9208],
        'PS18': [1.5309],
        'PS09': [0.2543, 0.6789, 0.763],
        'PS16': [0.1858, 0.2652]
    }

    baseline_response_magnitude = {
        'PS17': [0.1686],  # can't be used for paird pre4ap to interictal plot
        'PS07': [0.3574, 0.3032],
        'PS11': [0.4180, 0.3852],
        'PS18': [0.3594],
        'PS09': [0.2662],
        'PS16': [0.1217, 0.093, 0.087]
    }

    def __init__(self):
        super().__init__()
        self.exp_sz_occurrence: dict = None
        self.total_sz_occurrence: dict = None
        self.response_magnitudes = None  #: avg response magnitudes across all 1p stim trials
        self.response_decay = None  #: avg response magnitudes across all 1p stim trials
        self.photostim_responses = {'baseline': Any,
                                    'interictal': Any,
                                    'interictal - sz excluded': Any,
                                    'interictal - presz': Any,
                                    'interictal - postsz': Any}  #: individual post-stim response magnitudes across all 1p stim trials. includes separate set of values for excluded stims that induced seizure.
        self.decay_constants = {'baseline': Any,
                               'interictal': Any,
                               'interictal - sz excluded': Any,
                               'interictal - presz': Any,
                               'interictal - postsz': Any}  #: individual post-stim decay constants across all 1p stim trials. includes separate set of values for excluded stims that are induced seizure.

        self.save_results()

    def collect_response_magnitudes(self):
        data = [[rp for rp in onePresults.mean_stim_responses.iloc[:, 1] if rp != '-'],
                [rp for rp in onePresults.mean_stim_responses.iloc[:, 2] if rp != '-']]
        # data.append([rp for rp in onePresults.mean_stim_responses.iloc[:,3] if rp != '-'])
        self.response_magnitudes = data
        self.save_results()

    def collect_response_decay(self):
        data = [
            list(onePresults.mean_stim_responses[onePresults.mean_stim_responses.iloc[:, -3].notnull()].iloc[:, -3]),
            list(onePresults.mean_stim_responses[onePresults.mean_stim_responses.iloc[:, -2].notnull()].iloc[:, -2])]
        # data.append(ls(onePresults.mean_stim_responses[onePresults.mean_stim_responses.iloc[:, -1].notnull()].iloc[:, -1]))
        self.response_decay = data
        self.save_results()


REMAKE = False
if not os.path.exists(OnePhotonStimResults.SAVE_PATH) or REMAKE:
    resultsobj = OnePhotonStimResults()
else:
    pass


class OnePhotonStimAnalysisFuncs(OnePhotonStim):

    @staticmethod
    def collectPhotostimResponses(pre_stim=None, post_stim=None, response_len=None, response_type: str = 'pre-stim dFF',
                                  run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False, run_trials=[],
                                  skip_trials=[]):
        """I think collecting all photostimulation trials' responses for all experiments."""

        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials,
                                          ignore_cache=ignore_cache,
                                          run_trials=run_trials, skip_trials=skip_trials)
        def _collectPhotostimResponses(self: OnePhotonStim = None, pre_stim=pre_stim, post_stim=post_stim,
                                       response_len=response_len, response_type: str = response_type, **kwargs):
            """calculates and returns photostim reponse magnitudes and time decay constants."""

            self = self if not 'expobj' in [*kwargs] else kwargs['expobj']
            pre_stim = pre_stim if not 'expobj' in [*kwargs] else self.pre_stim
            post_stim = post_stim if not 'expobj' in [*kwargs] else self.post_stim
            response_len = response_len if not 'expobj' in [*kwargs] else self.response_len

            if not response_type in OnePhotonStim.compatible_responses_process:
                raise ValueError(f"{response_type} is not a compatible response_type")

            self.photostim_results = pd.DataFrame(index=['stim type', 'photostim responses', 'decay constant'],
                                                  columns=self.stim_start_frames)

            #### all stims for all exp types ###########################################################################

            stims_to_analyze = self.stim_start_frames

            flu_list = [self.meanRawFluTrace[stim - int(pre_stim * self.fps): stim + int(post_stim * self.fps)] for
                        stim in stims_to_analyze]

            if 'post' in self.exptype:
                self.photostim_results.loc[
                    'stim type', self.stims_out_sz] = 'interictal'  # set stim type to interictal for stim starts outsz in post4ap experiment
                self.photostim_results.loc[
                    'stim type', self.stims_in_sz] = 'ictal'  # set stim type to ictal for stim starts insz in post4ap experiment
            elif 'pre' in self.exptype:
                self.photostim_results.loc[
                    'stim type', self.stim_start_frames] = 'baseline'  # set stim type to baseline for all stim frames in pre4ap experiment

            # convert to dFF normalized to pre-stim F
            if response_type == 'pre-stim dFF':  # otherwise default param is raw Flu
                flu_list = [pj.dff(trace, baseline=np.mean(trace[:int(pre_stim * self.fps) - 2])) for trace in
                            flu_list]
            else:
                raise ValueError(f"{response_type} is not a compatible response_type")
            self.photostim_flu_snippets = np.asarray(flu_list)

            # measure magnitude of response
            if response_type == 'pre-stim dFF':  # otherwise default param is raw Flu
                poststim_1 = int(
                    pre_stim * self.fps) + self.stim_duration_frames + 2  # starting just after the end of the shutter opening
                poststim_2 = poststim_1 + int(response_len * self.fps)
                baseline = int(pre_stim * self.fps) - 2

                for idx, flu_snippet in enumerate(self.photostim_flu_snippets):
                    response = np.mean(flu_snippet[poststim_1:poststim_2]) - np.mean(flu_snippet[:baseline])
                    stim = stims_to_analyze[idx]
                    self.photostim_results.loc['photostim responses', stim] = response
                # self.photostim_results = response_list

            # measure the timescale of the decay
            if response_type == 'pre-stim dFF':  # otherwise default param is raw Flu
                if len(self.photostim_results) > 0:
                    poststim_1 = int(
                        pre_stim * self.fps) + self.stim_duration_frames + 2  # starting just after the end of the shutter opening

                    for idx, flu_snippet in enumerate(self.photostim_flu_snippets):
                        max_value = max(flu_snippet[poststim_1:])  # peak Flu value after stim
                        threshold = np.exp(-1) * max_value  # set threshod to be at 1/e x peak
                        try:
                            x_ = np.where(flu_snippet[poststim_1:] < threshold)[0][
                                0]  # find frame # where, after the stim period, avg_flu_trace reaches the threshold
                            decay_constant = x_ / self.fps  # convert frame # to time
                        except IndexError:
                            decay_constant = None  # cases where the trace doesn't return to threshold after the max value
                        stim = stims_to_analyze[idx]
                        self.photostim_results.loc['decay constant', stim] = decay_constant

                    # self.decay_constants = decay_constant_list
                else:
                    self.decay_constants = [None]

            if hasattr(self, 'photostim_results'):
                self.save()
                return True
            else:
                print(f"\t***** {self.t_series_name} no attr: 'photostim_results'")
                return False

        return _collectPhotostimResponses()

    @staticmethod
    def collectPhotostimResponsesIndivual(resultsobj: OnePhotonStimResults, run_pre4ap_trials=True,
                                          run_post4ap_trials=True, run_trials=[], skip_trials=[], ignore_cache=False):
        """collecting all photostimulation responses for individual trials.
        """
        if not hasattr(resultsobj, 'photostim_responses') or ignore_cache:
            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials,
                                              run_post4ap_trials=run_post4ap_trials,
                                              ignore_cache=ignore_cache, run_trials=run_trials, skip_trials=skip_trials,
                                              supress_print=False)
            def _collect_photostimResponse(**kwargs):
                print('start')
                expobj = kwargs['expobj']

                if 'pre' in expobj.exptype:
                    stims = expobj.stim_start_frames
                    pre4ap_resposnes[expobj.t_series_name] = expobj.photostim_results.loc['photostim responses', stims]

                elif 'post' in expobj.exptype:
                    stims = expobj.stims_out_sz
                    post4ap_resposnes[expobj.t_series_name] = expobj.photostim_results.loc['photostim responses', stims]

            pre4ap_resposnes = {}
            post4ap_resposnes = {}

            _collect_photostimResponse()

            resultsobj.photostim_responses = {'baseline': pre4ap_resposnes,
                                              'interictal': post4ap_resposnes}
            resultsobj.save_results()


    @staticmethod
    def collectPhotostimResponsesAndDecay_szexcluded(resultsobj: OnePhotonStimResults, ignore_cache=False):
        """collect all photostimulation responses for individual experiments - excluding stims that occured near seizure.
        idea is to exclude specifically the stims that led to seizure induction since these stim responses will be elevated
        - also collect all photostimulation decay constants for individual trials.


        """

        if 'interictal - sz excluded' in [*resultsobj.photostim_responses] and not ignore_cache: pass
        else:
            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=False, run_post4ap_trials=True, supress_print=False,
                                              ignore_cache=ignore_cache)
            def _collect_photostimResponse_szexclude(**kwargs):
                expobj = kwargs['expobj']

                responses = resultsobj.photostim_responses['interictal'][expobj.t_series_name]
                resultsobj.photostim_responses['interictal - sz excluded'][expobj.t_series_name] = None
                stims = list(responses.index)
                responses = list(responses.values)
                stims_keep = []
                responses_keep = []
                for i, stim in enumerate(stims):
                    exclude = False
                    for fr in expobj.seizure_frames:
                        if np.abs((stim - fr)) < 1 * expobj.fps:  # exclude if stim is within 1 sec of a seizure onset time
                            exclude = True
                            print('\txx -- excluding', i, stim)
                            # break
                    if not exclude:
                        stims_keep.append(stim)
                        print('\too --keeping', i, stim)
                        responses_keep.append(responses[i])

                post4ap_responses_sz_excluded[expobj.t_series_name] = pd.Series(index=stims_keep, data=responses_keep)

            post4ap_responses_sz_excluded = {}

            _collect_photostimResponse_szexclude()

            resultsobj.photostim_responses['interictal - sz excluded'] = post4ap_responses_sz_excluded
            resultsobj.save_results()

        if 'interictal - sz excluded' in [*resultsobj.decay_constants] and not ignore_cache: pass
        else:
            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=True, run_post4ap_trials=True, supress_print=False,
                                              ignore_cache=ignore_cache)
            def _collect_decay_constant(**kwargs):
                print('start')
                expobj: OnePhotonStim = kwargs['expobj']


                if 'post' in expobj.exptype:
                    stims = expobj.stims_out_sz
                    assert ['interictal' in i for i in expobj.photostim_results.loc['stim type', stims]], 'a non-interictal stim type was found in the post4ap trials selected'
                    post4ap_decays[expobj.t_series_name] = expobj.photostim_results.loc['decay constant', stims]

                    decays = expobj.photostim_results.loc['decay constant', stims]
                    # resultsobj.decay_constants['interictal - sz excluded'][expobj.t_series_name] = None

                    stims = list(decays.index)
                    decays = list(decays.values)
                    stims_keep = []
                    decays_keep = []
                    for i, stim in enumerate(stims):
                        exclude = False
                        for fr in expobj.seizure_frames:
                            if np.abs((stim - fr)) < 1 * expobj.fps:  # exclude if stim is within 1 sec of a seizure onset time
                                exclude = True
                                print('\txx -- excluding', i, stim)
                                # break
                        if not exclude:
                            stims_keep.append(stim)
                            print('\too --keeping', i, stim)
                            decays_keep.append(decays[i])

                    post4ap_decays[expobj.t_series_name] = pd.Series(index=stims_keep, data=decays_keep)


                elif 'pre' in expobj.exptype:
                    stims = expobj.stim_start_frames


                    # write new code for calculating decay constant for each stim below:

                    stims_keep = stims
                    decays_keep = expobj.photostim_results.loc['decay constant', stims]



                    # pre4ap_decays[expobj.t_series_name] = expobj.photostim_results.loc['decay constant', stims]
                    # responses = resultsobj.decay_constants['interictal'][expobj.t_series_name]
                    # resultsobj.decay_constants['interictal - sz excluded'][expobj.t_series_name] = None
                    # stims = list(responses.index)
                    # responses = list(responses.values)
                    # stims_keep = []
                    # responses_keep = []
                    # for i, stim in enumerate(stims):
                    #     exclude = False
                    #     for fr in expobj.seizure_frames:
                    #         if np.abs((stim - fr)) < 1 * expobj.fps:  # exclude if stim is within 1 sec of a seizure onset time
                    #             exclude = True
                    #             print('\txx -- excluding', i, stim)
                    #             # break
                    #     if not exclude:
                    #         stims_keep.append(stim)
                    #         print('\too --keeping', i, stim)
                    #         responses_keep.append(responses[i])

                    pre4ap_decays[expobj.t_series_name] = pd.Series(index=stims_keep, data=decays_keep)

            resultsobj.decay_constants = {}

            pre4ap_decays = {}
            post4ap_decays = {}

            _collect_decay_constant()

            resultsobj.decay_constants = {'baseline': pre4ap_decays,
                                         'interictal - sz excluded': post4ap_decays}
            resultsobj.save_results()


    @staticmethod
    def collectPhotostimResponses_PrePostSz(resultsobj: OnePhotonStimResults, ignore_cache=False):
        """collect all photostimulation responses for individual experiments -
        and group by bin of stim relative to seizure onset/offset."""

        if 'interictal - mid' in [*resultsobj.photostim_responses] and not ignore_cache:
            pass
        else:
            resultsobj.photostim_responses['interictal - mid'] = {}

            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=False, run_post4ap_trials=True, supress_print=False,
                                              ignore_cache=ignore_cache)
            def _collect_photostimResponse_mid(**kwargs):
                expobj: OnePhotonStim = kwargs['expobj']

                responses = resultsobj.photostim_responses['interictal'][expobj.t_series_name]
                resultsobj.photostim_responses['interictal - mid'][expobj.t_series_name] = None
                stims = list(responses.index)
                responses = list(responses.values)
                stims_keep = []
                responses_keep = []
                for i, stim in enumerate(stims):
                    exclude = False
                    for fr in expobj.seizure_frames:
                        if np.abs((
                                          stim - fr)) < 5 * expobj.fps:  # exclude if stim is within 5 sec of a seizure onset time
                            exclude = True
                            print(f'\txx -- excluding {i} {stim}')
                            break
                    if not exclude:
                        include = True
                        for sz_on, sz_off in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
                            if not (not (0 < (sz_on - stim) < 30 * expobj.fps) and not (0 < (
                                    stim - sz_off) < 30 * expobj.fps)):  # keep stim if >30secs of seizure onset AND >30secs after seizure offset
                                print(f'\too --kicking out {i} {stim}, sz on fr: {sz_on}, sz off fr: {sz_off}')
                                include = False
                        if include:
                            print(f'\too --keeping {i} {stim}')
                            stims_keep.append(stim)
                            responses_keep.append(responses[i])

                post4ap_responses_mid[expobj.t_series_name] = pd.Series(index=stims_keep, data=responses_keep)

            post4ap_responses_mid = {}
            _collect_photostimResponse_mid()
            resultsobj.photostim_responses['interictal - mid'] = post4ap_responses_mid
            resultsobj.save_results()

        if 'interictal - presz' in [*resultsobj.photostim_responses] and not ignore_cache:
            pass
        else:
            resultsobj.photostim_responses['interictal - presz'] = {}

            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=False, run_post4ap_trials=True, supress_print=False,
                                              ignore_cache=ignore_cache)
            def _collect_photostimResponse_presz(**kwargs):
                expobj: OnePhotonStim = kwargs['expobj']

                responses = resultsobj.photostim_responses['interictal'][expobj.t_series_name]
                resultsobj.photostim_responses['interictal - presz'][expobj.t_series_name] = None
                stims = list(responses.index)
                responses = list(responses.values)
                stims_keep = []
                responses_keep = []
                for i, stim in enumerate(stims):
                    exclude = False
                    for fr in expobj.seizure_frames:
                        if np.abs((
                                          stim - fr)) < 5 * expobj.fps:  # exclude if stim is within 5 sec of a seizure onset time
                            exclude = True
                            print(f'\txx -- excluding {i} {stim}')
                            break
                    if not exclude:
                        for sz_on in expobj.seizure_lfp_onsets:
                            if 0 < (sz_on - stim) < 30 * expobj.fps:  # keep stim if within 30secs of seizure onset
                                print(f'\too --keeping {i} {stim}, sz on fr: {sz_on}')
                                stims_keep.append(stim)
                                responses_keep.append(responses[i])

                post4ap_responses_presz[expobj.t_series_name] = pd.Series(index=stims_keep, data=responses_keep)

            post4ap_responses_presz = {}
            _collect_photostimResponse_presz()
            resultsobj.photostim_responses['interictal - presz'] = post4ap_responses_presz
            resultsobj.save_results()

        if 'interictal - postsz' in [*resultsobj.photostim_responses] and not ignore_cache:
            pass
        else:
            resultsobj.photostim_responses['interictal - postsz'] = {}

            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=False, run_post4ap_trials=True, supress_print=False,
                                              ignore_cache=ignore_cache)
            def _collect_photostimResponse_postsz(**kwargs):
                expobj: OnePhotonStim = kwargs['expobj']

                responses = resultsobj.photostim_responses['interictal'][expobj.t_series_name]
                resultsobj.photostim_responses['interictal - postsz'][expobj.t_series_name] = None
                stims = list(responses.index)
                responses = list(responses.values)
                stims_keep = []
                responses_keep = []
                for i, stim in enumerate(stims):
                    exclude = False
                    for fr in expobj.seizure_frames:
                        if np.abs((
                                          stim - fr)) < 5 * expobj.fps:  # exclude if stim is within 5 sec of a seizure onset time
                            exclude = True
                            print(f'\txx -- excluding {i} {stim}')
                            break
                    if not exclude:
                        for sz_off in expobj.seizure_lfp_offsets:
                            if 0 < (stim - sz_off) < 30 * expobj.fps:  # keep stim if within 30secs of seizure offset
                                print(f'\too --keeping {i} {stim}')
                                stims_keep.append(stim)
                                responses_keep.append(responses[i])

                post4ap_responses_postsz[expobj.t_series_name] = pd.Series(index=stims_keep, data=responses_keep)

            post4ap_responses_postsz = {}
            _collect_photostimResponse_postsz()

            resultsobj.photostim_responses['interictal - postsz'] = post4ap_responses_postsz
            resultsobj.save_results()

    @staticmethod
    def collectPreStimFluAvgs(run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False, run_trials=[],
                              skip_trials=[]):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials,
                                          ignore_cache=ignore_cache,
                                          run_trials=run_trials, skip_trials=skip_trials)
        def _collectPreStimFluAvgs(self=None, **kwargs):
            "returns the prestim Flu avg values"
            pre_stim = 1  # seconds
            self = self if not 'expobj' in [*kwargs] else kwargs['expobj']

            new_result = 'pre stim flu avg'
            if new_result not in self.photostim_results.index:
                df_ = pd.DataFrame(index=[new_result], columns=self.stim_start_frames)
                self.photostim_results = self.photostim_results.append(df_)

            #### all stim for all exp types ########################################################################
            pre_stim_flu_list = [self.meanRawFluTrace[stim - int(pre_stim * self.fps): stim] for stim in
                                 self.stim_start_frames]
            _pre_stim_flu_list = np.mean(np.asarray(pre_stim_flu_list), axis=1)
            self.photostim_results.loc['pre stim flu avg', self.stim_start_frames] = np.round(_pre_stim_flu_list, 3)
            ############################################################################ all stim for all exp types #

            self.save()
            return True

        return _collectPreStimFluAvgs()

    @staticmethod
    def collectTimeToSzOnset(self=None, run_pre4ap_trials=True, run_post4ap_trials=True, ignore_cache=False,
                             run_trials=[],
                             skip_trials=[], supress_print=False):
        @OnePhotonStim.runOverExperiments(run_pre4ap_trials=run_pre4ap_trials, run_post4ap_trials=run_post4ap_trials,
                                          ignore_cache=ignore_cache, run_trials=run_trials, skip_trials=skip_trials,
                                          supress_print=supress_print)
        def _collectTimeToSzOnset(self=self, **kwargs):
            "returns time to sz onset for each stim"
            self = self if not 'expobj' in [*kwargs] else kwargs['expobj']

            if 'post' in self.exptype:
                # create and append empty pd dataframe
                if 'time to seizure onset (secs)' not in self.photostim_results.index:
                    df2 = pd.DataFrame(index=['time to seizure onset (secs)'], columns=self.stim_start_frames)
                    self.photostim_results = self.photostim_results.append(df2)

                #### inter-ictal and ictal stims #######################################################################
                # transform each stim in photostim flu responses into time relative value:
                for stim_idx, stim_frame in enumerate(self.stim_start_frames):
                    closest_sz_onset = pj.findClosest(arr=self.seizure_lfp_onsets, input=stim_frame)[0]
                    time_diff = (closest_sz_onset - stim_frame) / self.fps  # time difference in seconds

                    if stim_frame in self.stims_out_sz and time_diff < 0:
                        time_diff = None
                    elif stim_frame in self.stims_in_sz and time_diff > 0:
                        time_diff = None
                    else:
                        time_diff = round(time_diff, 3)

                    self.photostim_results.loc['time to seizure onset (secs)', stim_frame] = time_diff

                ########################################################################### inter-ictal and ictal stims #

                self.save()
                return True
            elif 'pre' in self.exptype:
                df2 = pd.DataFrame(index=['time to seizure onset (secs)'], columns=self.stim_start_frames)
                self.photostim_results = self.photostim_results.append(
                    df2)  # only append NaN's for pre4ap (no seizure) trials

                return True

        return _collectTimeToSzOnset()

    @staticmethod
    def collectSzOccurrenceRelativeStim(Results: OnePhotonStimResults, rerun=0):

        if not hasattr(Results, 'total_sz_occurrence') or rerun:
            @OnePhotonStim.runOverExperiments(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
            def __function(**kwargs):
                expobj: OnePhotonStim = kwargs['expobj']

                if expobj.t_series_name == 'PS16 t-009' or expobj.t_series_name == 'PS07 t-015':
                    print(
                        'break here and investigate further where this exp"s seizures are occuring relative to precise timing of stims.')
                    # from _utils_.alloptical_plotting import plotLfpSignal
                    # plotLfpSignal(expobj, x_axis='time', figsize=(30, 3), linewidth=0.5, downsample=True,
                    #                      sz_markings=True, color='black')

                # print(f'{expobj.t_series_name}: {np.sum(expobj.sz_occurrence_stim_intervals2)}')

                return (expobj.t_series_name, expobj.sz_occurrence_stim_intervals2)

            func_collector = __function()
            print(func_collector)

            unique_exps = np.unique([exp[:4] for exp in OnePhotonStim.oneP_post4ap_exp_list])
            results = {}
            for i in unique_exps:
                results[i] = None

            for i, exp_sz_prob in enumerate(func_collector):
                exp = exp_sz_prob[0][:4]

                if results[exp] is None:
                    results[exp] = exp_sz_prob[1]
                else:
                    results[exp] = np.mean(np.vstack([results[exp], exp_sz_prob[1]]), axis=0)
                # array_sz[i] = exp_sz_prob[1]

            sz_occurrence = np.array([list(results.items())[i][1] for i, _ in enumerate(unique_exps)])

            print(results)
            # sz_occurence_relative = [func_collector[0]]
            # for sz_occurrence in func_collector[1:]:
            #     sz_occurence_relative += sz_occurrence

            # return sz_occurence_relative / len(func_collector)
            Results.exp_sz_occurrence, Results.total_sz_occurrence = results, sz_occurrence
            Results.save_results()
            return Results.exp_sz_occurrence, Results.total_sz_occurrence
        else:
            return Results.exp_sz_occurrence, Results.total_sz_occurrence


if __name__ == '__main__':
    Results: OnePhotonStimResults = OnePhotonStimResults.load()

    # plot photostim responses
    # individual trials photostim responses

    # 1) interictal - comparing presz vs. postsz photostim responses

    interictal_response_magnitudes_presz = Results.photostim_responses['interictal - presz']
    interictal_response_magnitudes_postsz = Results.photostim_responses['interictal - postsz']
    baseline_response_magnitudes = Results.photostim_responses['baseline']
    interictal_response_magnitudes_szexclude = Results.photostim_responses['interictal - sz excluded']

    baseline_resposnes = []
    presz_responses = []
    postsz_responses = []
    interictal_resposnes_szexclude = []
    for trial, responses in interictal_response_magnitudes_szexclude.items():
        interictal_resposnes_szexclude.extend(list(responses))
    for trial, responses in baseline_response_magnitudes.items():
        baseline_resposnes.extend(list(responses))
    for trial, responses in interictal_response_magnitudes_presz.items():
        presz_responses.extend(list(responses))
    for trial, responses in interictal_response_magnitudes_postsz.items():
        postsz_responses.extend(list(responses))

    # STATS
    # t-test - individual sessions
    print(f"P(t-test - (indiv. trials) response: presz ({len(presz_responses)} trials) vs. postsz ({len(postsz_responses)} trials)): \n\t\t{stats.ttest_ind(presz_responses, postsz_responses)[1]:.3e}")
    print(f"P(t-test - (indiv. trials) response: interictal ({len(interictal_resposnes_szexclude)} trials) vs. postsz ({len(postsz_responses)} trials)): \n\t\t{stats.ttest_ind(interictal_resposnes_szexclude, postsz_responses)[1]:.3e}")
    print(f"P(t-test - (indiv. trials) response: interictal ({len(interictal_resposnes_szexclude)} trials) vs. presz ({len(presz_responses)} trials)): \n\t\t{stats.ttest_ind(interictal_resposnes_szexclude, presz_responses)[1]:.3e}")
    print(f"P(t-test - (indiv. trials) response: baseline ({len(baseline_resposnes)} trials) vs. postsz ({len(postsz_responses)} trials)): \n\t\t{stats.ttest_ind(baseline_resposnes, postsz_responses)[1]:.3e}")

    # run ANOVA of photostim response magnitudes across baseline, presz, postsz, interictal
    stats.kruskal(baseline_resposnes, presz_responses, postsz_responses, interictal_resposnes_szexclude)
    # run post hoc tukey comparisons
    tukey_hsd({'baseline': baseline_resposnes, 'presz': presz_responses,
               'postsz': postsz_responses, 'interictal': interictal_resposnes_szexclude})


    # print mean and stdev of response magnitudes
    print(f"Mean response magnitude (presz): {np.mean(presz_responses):.3f} +/- {np.std(presz_responses):.3f}")
    print(f"Mean response magnitude (postsz): {np.mean(postsz_responses):.3f} +/- {np.std(postsz_responses):.3f}")

    plot_bar_with_points(data=[baseline_resposnes, presz_responses, postsz_responses, interictal_resposnes_szexclude], x_tick_labels=['baseline', 'presz', 'postsz', 'interictal'], fontsize=fontsize_extraplot,
                         points=False, bar=True, figsize=[3, 3], colors=[baseline_color, 'red', 'violet', interictal_color], s=10, show=True,
                         x_label='', y_label='Avg. dFF', alpha=0.7, lw=1)



