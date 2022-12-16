# read .csv file that contains metainfo about experiments

# # import results superobject that will collect analyses from various individual experiments
import pickle
from dataclasses import dataclass
from funcsforprajay.funcs import save_pkl, load_pkl
import os
import funcsforprajay.funcs as pj

import numpy as np
import pandas as pd

local = "/Users/prajayshah/data/oxford-data/export/"
remote_base = "/home/pshah/mnt/qnap/Analysis/"
remote = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"
current_loc = remote
EXPMETA_path = '/home/pshah/mnt/qnap/Analysis/allopticalseizuresexpmeta.pkl'
CSV_PATH_ao = f'{remote_base}allopticalexpmeta.csv'
CSV_PATH_1p = f'{remote_base}onephotonexpmeta.csv'

# UTILS
def import_meta_from_csv(csv_path=CSV_PATH_ao):
    metainfo = pd.read_csv(csv_path).iloc[:, 1:]
    return metainfo

def import_resultsobj(pkl_path: str):
    assert os.path.exists(pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f"\nimporting resultsobj from: {pkl_path} ... ")
        resultsobj = pickle.load(f)
        print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
    return resultsobj

#

@dataclass
class AllOpticalExpsToAnalyze:
    """Lists of alloptical experiments to use during analysis"""
    csv_path: str = CSV_PATH_ao

    exp_ids = [
        'RL108',
        'RL109',
        'PS04',
        'PS07',
        'PS06',
        'PS11'
    ]

    pre_4ap_trials = [
        ['RL108 t-009'],
        # ['RL108 t-010'],
        # ['RL109 t-007'], - not great targets responses
        # ['RL109 t-008'],
        ['RL109 t-013'],
        # ['RL109 t-014'], -- good targets responses but choosing t-013 because stim length more similar to other experiments
        ['PS04 t-012',
         # , 'PS04 t-014',  # - not sure what's wrong with PS04 t-014, but the photostim and Flu are falling out of sync .21/10/09
         'PS04 t-017'],  # just commented out until t-018 gets fully sorted out again.
        # ['PS05 t-010'],
        ['PS07 t-007'],
        # ['PS07 t-009'],
        # ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],  # matching run_post4ap_trials trial cannot be analysed
        ['PS06 t-011'],
        # ['PS06 t-012'],  # matching run_post4ap_trials trial cannot be analysed
        # ['PS11 t-007'],
        ['PS11 t-010'],
        # ['PS17 t-005'],
        # ['PS17 t-006', 'PS17 t-007'],
        # ['PS18 t-006']
    ]

    post_4ap_trials = [
        ['RL108 t-013'],
        # ['RL108 t-011'],  # -- need to do sz boundary classifying processing - double checking on .22/03/06, should be good to go .22/03/07
        # ['RL109 t-020'],  # -- need to do sz boundary classifying processing - double checking on .22/03/06
        # ['RL109 t-021'],
        ['RL109 t-018'],
        # ['RL109 t-016'],  # 'RL109 t-017'], # -- need to do sz boundary classifying processing for t-017, -- t-016 good targets responses but choosing t-013 because stim length more similar to other experiments
        ['PS04 t-018'],  # -- should be good to go. need to redo sz boundary classifying processing - TODO need to redo once again? some stims seems to be missing .22/02/22
        # ['PS05 t-012'],
        ['PS07 t-011'],
        # ['PS07 t-017'],  -- unclear seizure behaviours
        # ['PS06 t-014', 'PS06 t-015'], - t-014 might have one seizure, t-015 likely not any sz events .22/01/06
        ['PS06 t-013'],
        # ['PS06 t-016'], - no seizures
        # ['PS11 t-016'], - very short seizure
        ['PS11 t-011'],
        # ['PS17 t-011'],
        # ['PS17 t-009'],
        # ['PS18 t-008']
    ]

    trial_maps = {'pre': {
        'a': ['RL108 t-009'],
        # 'b': ['RL108 t-010'],
        # 'c': ['RL109 t-007'],
        # 'd': ['RL109 t-008'],
        'e': ['RL109 t-013'],
        # 'f': ['RL109 t-014'],
        'g': ['PS04 t-012',  # 'PS04 t-014',  # - temp just until PS04 gets reprocessed
              'PS04 t-017'],
        # 'h': ['PS05 t-010'],
        'i': ['PS07 t-007'],
        # 'j': ['PS07 t-009'],
        # 'k': ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
        'l': ['PS06 t-011'],
        # 'm': ['PS06 t-012'],  # - t-016 missing sz lfp onsets
        # 'n': ['PS11 t-007'],
        'o': ['PS11 t-010'],
        # 'p': ['PS17 t-005'],
        # 'q': ['PS17 t-006', 'PS17 t-007'],
        # 'r': ['PS18 t-006']
    }, 'post': {
        'a': ['RL108 t-013'],
        # 'b': ['RL108 t-011'], # -- need to redo sz boundary classifying processing - should be done
        # 'c': ['RL109 t-020'],
        # 'd': ['RL109 t-021'],
        'e': ['RL109 t-018'],
        # 'f': ['RL109 t-016'],  # 'RL109 t-017'], #-- need to do sz boundary classifying processing - should be done
        'g': ['PS04 t-018'],
        # 'h': ['PS05 t-012'],
        'i': ['PS07 t-011'],
        # 'j': ['PS07 t-017'],  # - need to do sz boundary classifying processing
        # 'k': ['PS06 t-014', 'PS06 t-015'],  # - missing seizure_lfp_onsets
        'l': ['PS06 t-013'],
        # 'm': ['PS06 t-016'],  # - missing seizure_lfp_onsets - LFP signal not clear, but there is seizures on avg Flu trace
        # 'n': ['PS11 t-016'],
        'o': ['PS11 t-011'],
        # 'p': ['PS17 t-011'],
        # 'q': ['PS17 t-009'],
        # 'r': ['PS18 t-008']
    }}

    post_ictal_exclude_sz = {
        'RL109 t-018': [2],
        'PS06 t-013': [0, 4, 5],
        'PS04 t-018': [],  # the interseizure interval is short, but it seems to be okay for the post-ictal analysis
        'PS07 t-011': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],  # only one seizure left, therefore leaving out of quantification
        'PS11 t-011': [2, 7]
    }  #: seizures to exclude from analysis for the post ictal phase (most because the seizure termination marking isn't perfect or there is a CSD or other artifact)

    trials_idx_analysis = {
        'baseline': [[0], [1], [2, 3], [4], [5], [6]],
        'interictal': [[0], [1], [2], [3], [4], [5]]
    }

    def __post_init__(self):
        self.metainfo = import_meta_from_csv(csv_path=self.csv_path)

    def save_to_csv(self):
        """saves csv from which this class collects metainfo"""
        pass

    def add_to_csv(self):
        pass

    @classmethod
    def all_pre4ap_trials(cls):
        return pj.flattenOnce(cls.pre_4ap_trials)

    @classmethod
    def all_post4ap_trials(cls):
        return pj.flattenOnce(cls.post_4ap_trials)

    @classmethod
    def find_matched_trial(cls, pre4ap_trial_name = None, post4ap_trial_name = None):
        """
        Returns the matched trial ID depending on whether the pre4ap trial or post4ap trial was given as input.
        
        :param pre4ap_trial_name: 
        :param post4ap_trial_name: 
        """
        # return the pre4ap matched trial
        if post4ap_trial_name:
            for map_key, expid in cls.trial_maps['post'].items():  # find the pre4ap exp that matches with the current post4ap experiment
                if post4ap_trial_name in expid:
                    pre4ap_match_id = cls.trial_maps['pre'][map_key]
                    if map_key == 'g':
                        pre4ap_match_id = cls.trial_maps['pre'][map_key][1]
                        return pre4ap_match_id
                    else:
                        return pre4ap_match_id[0]

        # return the post4ap matched trial
        elif pre4ap_trial_name:
            for map_key, expid in cls.trial_maps['pre'].items():  # find the pre4ap exp that matches with the current post4ap experiment
                if pre4ap_trial_name in expid:
                    post4ap_match_id = cls.trial_maps['post'][map_key]
                    return post4ap_match_id[0]
        else:
            ValueError('no pre4ap or post4ap trial names provided to match.')


assert len(AllOpticalExpsToAnalyze.pre_4ap_trials) == len(AllOpticalExpsToAnalyze.post_4ap_trials), (
    f"# of pre trials: {len(AllOpticalExpsToAnalyze.pre_4ap_trials)}",
    f"# of post trials: {len(AllOpticalExpsToAnalyze.post_4ap_trials)}")

assert len(AllOpticalExpsToAnalyze.trial_maps['pre'].keys()) == len(
    AllOpticalExpsToAnalyze.trial_maps['post'].keys()), (
    f"# of pre trials: {len(AllOpticalExpsToAnalyze.trial_maps['pre'].keys())}",
    f"# of post trials: {len(AllOpticalExpsToAnalyze.trial_maps['post'].keys())}")




@dataclass
class OnePhotonStimExpsToAnalyze:
    """Lists of all one photon stim experiments to use during analysis"""
    csv_path: str = CSV_PATH_1p

    def __post_init__(self):
        onepexpmeta = import_meta_from_csv(csv_path=self.csv_path)
        self.exppkllist = list(onepexpmeta.loc[:, 'pkl_paths'])

        pre4ap_idx = np.where(onepexpmeta['trial_type'] == 'pre-4ap')[0]
        self.pre_4ap_trials = list(onepexpmeta.iloc[pre4ap_idx]['prep_trial'])

        post4ap_idx = np.where(onepexpmeta['trial_type'] == 'post-4ap')[0]
        self.post_4ap_trials = list(onepexpmeta.iloc[post4ap_idx]['prep_trial'])

    def save_to_csv(self):
        """saves csv from which this class collects metainfo"""
        pass

    def add_to_csv(self):
        pass


class ExpMetainfo:
    csv_path: str = CSV_PATH_ao
    alloptical: AllOpticalExpsToAnalyze = AllOpticalExpsToAnalyze()
    onephotonstim: OnePhotonStimExpsToAnalyze = OnePhotonStimExpsToAnalyze()
    figure_settings = {
        "fontsize - extraplot": 10,
        "fontsize - intraplot": 8,
        "fontsize - title": 10,
        "fontsize - label": 12,
        "lfp - lw": 0.15,
        "gcamp - single cell - lw": 0.5,
    }


    def __init__(self):
        pass

    def save(self):
        save_pkl(self, EXPMETA_path)

    @staticmethod
    def load():
        return load_pkl(EXPMETA_path)

try:
    ExpMetainfo = ExpMetainfo.load()
except Exception:
    ExpMetainfo.save(ExpMetainfo)




# # ARCHIVE
# def __resultsmeta_to_csv():
#     # RESULT OBJECT - soon to be phased out
#     def import_resultsobj(pkl_path: str):
#         assert os.path.exists(pkl_path)
#         with open(pkl_path, 'rb') as f:
#             print(f"\nimporting resultsobj from: {pkl_path} ... ")
#             resultsobj = pickle.load(f)
#             print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
#         return resultsobj
#
#     from _results_.alloptical_results_init import AllOpticalResults
#
#     results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
#     allopticalResults: AllOpticalResults = import_resultsobj(pkl_path=results_object_path)
#     allopticalResults.metainfo.to_csv(CSV_PATH_ao)
#     print(f'\n saved allopticalResults.metainfo to {CSV_PATH_ao}')
#
#
# def __1presultsmeta_to_csv():
#     # RESULT OBJECT - soon to be phased out
#     results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
#     onePresults = import_resultsobj(pkl_path=results_object_path)
#
#     data_dict = {}
#     prep_trial = []
#     trial_type = []
#     pkl_list = []
#     for pkl_path in onePresults.mean_stim_responses['pkl_list']:
#         if list(onePresults.mean_stim_responses.loc[
#                     onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'post-4ap response (during sz)'])[0] != '-':
#             idx = np.where(onePresults.mean_stim_responses.loc[:, 'pkl_list'] == pkl_path)[0][0]
#             prep = onePresults.mean_stim_responses.loc[:, 'Prep'].iloc[idx]
#             trial = onePresults.mean_stim_responses.loc[:, 'Trial'].iloc[idx]
#             prep_trial.append(f"{prep} {trial}")
#             trial_type.append('post-4ap')
#             pkl_list.append(pkl_path)
#         elif list(onePresults.mean_stim_responses.loc[
#                     onePresults.mean_stim_responses['pkl_list'] == pkl_path, 'pre-4ap response'])[0] != '-':
#             idx = np.where(onePresults.mean_stim_responses.loc[:, 'pkl_list'] == pkl_path)[0][0]
#             prep = onePresults.mean_stim_responses.loc[:, 'Prep'].iloc[idx]
#             trial = onePresults.mean_stim_responses.loc[:, 'Trial'].iloc[idx]
#             prep_trial.append(f"{prep} {trial}")
#             trial_type.append('pre-4ap')
#             pkl_list.append(pkl_path)
#         else:
#             print(f'Warning: Missing result! for: {pkl_path}')
#
#     data_dict['prep_trial'] = prep_trial
#     data_dict['trial_type'] = trial_type
#     data_dict['pkl_paths'] = pkl_list
#
#     df = pd.DataFrame(data_dict)
#
#     # onePresults.mean_stim_responses['pkl_list'].to_csv(CSV_PATH_1p)
#     df.to_csv(CSV_PATH_1p)
#
#     print(f"\n saved onePresults.mean_stim_responses['pkl_list'] to {CSV_PATH_1p}")



if __name__ == '__main__':
    # __resultsmeta_to_csv()
    # __1presultsmeta_to_csv()
    # ExpMetainfo

    pass

