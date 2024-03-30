import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from _utils_ import _alloptical_utils as Utils
import funcsforprajay.funcs as pj

from _analysis_._utils import Quantification, Results
from _main_.Post4apMain import Post4ap

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"
SAVE_PATH = SAVE_LOC + 'TargetsPhotostimResponsesInterictal.pkl'



# %% results

class TargetsPhotostimResponsesInterictalResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__TargetsStimsSzOnsetTime.pkl'

    def __init__(self):
        super().__init__()
        self.binned__szonsettime_vs_photostimresponses = {}


# %%
class TargetsPhotostimResponsesInterictal(Quantification):
    """class for collecting analysis for time to nearest LFP seizure onset for each stim in experiment.
    Then plot photostims responses in relation to this time to LFP onset.
    TODO: analyse photostim response from last stim before seizure and first stim after seizure termination.
    """

    photostim_responses_zscore_type = 'dFF (zscored) (interictal)'

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\- ADDING NEW TargetsPhotostimResponsesInterictal MODULE to expobj: {expobj.t_series_name}')
        self.collect__stims_time_to_szonset(expobj=expobj)
        self._create_anndata(expobj=expobj)
        self.collect__responses_szonsettime_df()

    def __repr__(self):
        return f"TargetsPhotostimResponsesInterictal <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"
    

    # 0) CALCULATE DELAY TO NEAREST SZ LFP ONSET FOR EACH STIM FRAME IN EXP - SAVE TO ANNDATA
    # calculate delay to nearest sz LFP onset for each stim frame in exp
    def collect__stims_time_to_szonset(self, expobj: Post4ap):
        """
        calculate delay to nearest sz LFP onset for each stim frame in exp
        """

        # transform stims to relative time to seizure lfp onset
        stims_relative_sz = [np.nan] * expobj.n_stims
        for stim_idx in expobj.stim_idx_outsz:
            stim_frame = expobj.stim_start_frames[stim_idx]
            closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
            time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
            stims_relative_sz[stim_idx] = round(time_diff, 3)

        self.stim_times_szonset = stims_relative_sz

    def _create_anndata(self, expobj: Post4ap):
        """
        Creates annotated data by extending the adata table from .PhotostimResponsesSLMTargets.adata and adding the
        time to sz onset for each stim frame as a variable.

        """
        assert hasattr(self, 'stim_times_szonset')
        self.adata = expobj.PhotostimResponsesSLMTargets.adata
        # add time to sz time onset for stims as adata var
        self.adata.add_variable(var_name='time_szonset', values=self.stim_times_szonset)
        print(self.adata)

    # 1) make df of photostim responses and time to sz LFP onset
    def collect__responses_szonsettime_df(self):
        df = pd.DataFrame(columns=['target_id', 'stim_id', 'time_toszonset', 'target_dFF', self.photostim_responses_zscore_type])

        # stim_ids = [stim_frame for stim_frame, idx in enumerate(self.adata.var['stim_start_frame']) if self.adata.var['time_szonset'][idx].notnull()]

        index_ = 0
        for idx, target in enumerate(self.adata.obs.index):
            for idxstim, stim in enumerate(self.adata.var['stim_start_frame']):
                time_szonset = self.adata.var['time_szonset'][idxstim]
                response = self.adata.X[idx, idxstim]
                zscored_response = self.adata.layers[self.photostim_responses_zscore_type][idx, idxstim]
                if not np.isnan(time_szonset):
                    df = pd.concat(
                        [df, pd.DataFrame({'target_id': target, 'stim_id': stim, 'time_szonset': time_szonset,
                                           self.photostim_responses_zscore_type: zscored_response, 'target_dFF': response},
                                          index=[index_])])  # TODO need to update the idx to use a proper index range
                    index_ += 1

        self.timeszonset_v_photostimresponses_zscored_df = df

    #2) collect binned time to sz onset vs photostim responses
    @staticmethod
    def collect__binned__szonsettime_v_responses():
        """collect time to sz onset vs. respnses for time bins"""
        bin_width = 5  # sec
        bins = np.arange(-100, 0, bin_width)  # -60 --> 0 secs, split in bins
        num = [0 for _ in range(len(bins))]  # num of datapoints in binned sztemporalinv
        y = [0 for _ in range(len(bins))]  # avg responses at distance bin
        responses = [[] for _ in range(len(bins))]  # collect all responses at distance bin

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=False)
        def add_time_responses(bins, num, y, responses, **kwargs):
            expobj = kwargs['expobj']

            szonset = expobj.TargetsPhotostimResponsesInterictal

            for _, row in szonset.timeszonset_v_photostimresponses_zscored_df.iterrows():
                sztime = row['time_szonset']
                # response = row['target_dFF']
                response = row[szonset.photostim_responses_zscore_type]
                for i, bin in enumerate(bins[:-1]):
                    if bins[i] < sztime < (bins[i + 1]):
                        num[i] += 1
                        y[i] += response
                        responses[i].append(response)

            return num, y, responses

        func_collector = add_time_responses(bins=bins, num=num, y=y, responses=responses)

        num, y, responses = func_collector[-1][0], func_collector[-1][1], func_collector[-1][2]

        szonset_time = bins + bin_width / 2

        avg_responses = [np.mean(responses_) for responses_ in responses]

        # calculate 95% ci for avg responses
        import scipy.stats as stats

        conf_int = np.array(
            [stats.t.interval(alpha=0.95, df=len(responses_) - 1, loc=np.mean(responses_), scale=stats.sem(responses_))
             for responses_ in responses])

        return bin_width, szonset_time, num, avg_responses, conf_int

    @staticmethod
    def plot__responses_v_szinvtemporal(results: TargetsPhotostimResponsesInterictalResults):
        """plot time to sz onset vs. respnses over time bins"""
        # sztemporalinv_bins = results.binned__distance_vs_photostimresponses['sztemporal_bins']
        szonset_time = results.binned__szonsettime_vs_photostimresponses['szonset_time']
        avg_responses = results.binned__szonsettime_vs_photostimresponses['avg_responses']
        conf_int = results.binned__szonsettime_vs_photostimresponses['conf_int']
        num2 = results.binned__szonsettime_vs_photostimresponses['num']

        conf_int_sztemporalinv = pj.flattenOnce(
            [[szonset_time[i], szonset_time[i + 1]] for i in range(len(szonset_time) - 1)])
        conf_int_values_neg = pj.flattenOnce([[val, val] for val in conf_int[1:, 0]])
        conf_int_values_pos = pj.flattenOnce([[val, val] for val in conf_int[1:, 1]])

        fig, axs = plt.subplots(figsize=(6, 6), nrows=2, ncols=1)
        ax = axs[0]
        ax2 = axs[1]
        ax.plot(szonset_time, avg_responses, c='cornflowerblue', zorder=2)
        # ax.step(szonset_time, avg_responses, c='cornflowerblue', zorder=2)
        ax.fill_between(x=szonset_time[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)

        # ax.fill_between(x=conf_int_sztemporalinv, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
        #                 zorder=0)

        # ax.scatter(sztemporalinv[:-1], avg_responses, c='orange', zorder=4)
        ax.set_ylim([-2, 2])
        ax.set_title(
            f'photostim responses vs. time to sz LFP onset (binned every {results.binned__szonsettime_vs_photostimresponses["bin_width"]}sec)',
            wrap=True)
        ax.set_xlabel('time to sz onset (secs)')
        ax.set_ylabel(TargetsPhotostimResponsesInterictal.photostim_responses_zscore_type)
        ax.margins(0)

        pixels = [np.array(num2)] * 10
        ax2.imshow(pixels, cmap='Greys', vmin=-5, vmax=150, aspect=0.1)
        # ax.show()

        fig.tight_layout(pad=1)
        fig.show()


# %% run analysis/results code
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0)
def run__init_TargetsStimsSzOnsetTime(cls, **kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsPhotostimResponsesInterictal = cls(expobj=expobj)
    expobj.save()

if __name__ == '__main__':


    main = TargetsPhotostimResponsesInterictal
    #
    # REMAKE = False
    # if not os.path.exists(TargetsPhotostimResponsesInterictalResults.SAVE_PATH) or REMAKE:
    #     results = TargetsPhotostimResponsesInterictalResults()
    #     results.save_results()
    # else:
    #     results = TargetsPhotostimResponsesInterictalResults.load()
    #
    # run__init_TargetsStimsSzOnsetTime(cls=main)
    # bin_width, szonset_time, num, avg_responses, conf_int = main.collect__binned__szonsettime_v_responses()
    # results.binned__szonsettime_vs_photostimresponses = {'bin_width': bin_width,
    #                                                      'szonset_time': szonset_time,
    #                                                      'num': num,
    #                                                      'avg_responses': avg_responses,
    #                                                      'conf_int': conf_int}
    # results.save_results()
    # main.plot__responses_v_szinvtemporal(results)


