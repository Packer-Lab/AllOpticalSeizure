# read .csv file that contains metainfo about experiments


# # import results superobject that will collect analyses from various individual experiments
import pickle
from dataclasses import dataclass

import os
import pandas as pd


CSV_PATH = '/home/pshah/mnt/qnap/Analysis/allopticalexpmeta.csv'


# %%
def __resultsmeta_to_csv():
    # %% RESULT OBJECT - soon to be phased out
    def import_resultsobj(pkl_path: str):
        assert os.path.exists(pkl_path)
        with open(pkl_path, 'rb') as f:
            print(f"\nimporting resultsobj from: {pkl_path} ... ")
            resultsobj = pickle.load(f)
            print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
        return resultsobj

    from _results_.alloptical_results_init import AllOpticalResults

    results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
    allopticalResults: AllOpticalResults = import_resultsobj(pkl_path=results_object_path)
    allopticalResults.metainfo.to_csv(CSV_PATH)
    print(f'\n saved allopticalResults.metainfo to {CSV_PATH}')


def import_meta_from_csv(csv_path=CSV_PATH):
    metainfo = pd.read_csv(csv_path)
    return metainfo


# %%

@dataclass
class AllOpticalExpsToAnalyze:
    """Lists of all experiments to use during analyse"""
    csv_path: str = CSV_PATH

    pre_4ap_trials = [
        ['RL108 t-009'],
        # ['RL108 t-010'],
        ['RL109 t-007'],
        # ['RL109 t-008'],
        ['RL109 t-013'],
        ['RL109 t-014'],
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
        # ['RL108 t-011'],  # -- need to do sz boundary classifying processing
        ['RL109 t-020'],  # -- need to do sz boundary classifying processing
        # ['RL109 t-021'],
        ['RL109 t-018'],
        ['RL109 t-016'],  # 'RL109 t-017'], # -- need to do sz boundary classifying processing
        ['PS04 t-018'],
        # -- need to redo sz boundary classifying processing - TODO need to redo once again? some stims seems to be missing .22/02/22
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
        'c': ['RL109 t-007'],
        # 'd': ['RL109 t-008'],
        'e': ['RL109 t-013'],
        'f': ['RL109 t-014'],
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
        'c': ['RL109 t-020'],  # -- need to redo sz boundary classifying processing - should be done
        # 'd': ['RL109 t-021'],
        'e': ['RL109 t-018'],  # -- need to redo sz boundary classifying processing - should be done
        'f': ['RL109 t-016'],  # 'RL109 t-017'], #-- need to do sz boundary classifying processing - should be done
        'g': ['PS04 t-018'],
        # -- need to redo sz boundary classifying processing - TODO need to redo once again? some stims seems to be missing .22/02/22
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

    def __post_init__(self):
        self.metainfo = import_meta_from_csv(csv_path=self.csv_path)

    def save_to_csv(self):
        """saves csv from which this class collects metainfo"""
        pass

    def add_to_csv(self):
        pass


assert len(AllOpticalExpsToAnalyze.pre_4ap_trials) == len(AllOpticalExpsToAnalyze.post_4ap_trials), (
    f"# of pre trials: {len(AllOpticalExpsToAnalyze.pre_4ap_trials)}",
    f"# of post trials: {len(AllOpticalExpsToAnalyze.post_4ap_trials)}")

assert len(AllOpticalExpsToAnalyze.trial_maps['pre'].keys()) == len(
    AllOpticalExpsToAnalyze.trial_maps['post'].keys()), (
    f"# of pre trials: {len(AllOpticalExpsToAnalyze.trial_maps['pre'].keys())}",
    f"# of post trials: {len(AllOpticalExpsToAnalyze.trial_maps['post'].keys())}")


class ExpMetainfo:
    csv_path: str = CSV_PATH
    alloptical: AllOpticalExpsToAnalyze = AllOpticalExpsToAnalyze()



if __name__ == '__main__':
    # __resultsmeta_to_csv()

    pass
