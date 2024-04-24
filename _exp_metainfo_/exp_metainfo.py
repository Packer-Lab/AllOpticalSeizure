# read .csv file that contains metainfo about experiments

# # import results superobject that will collect analyses from various individual experiments
import pickle
from dataclasses import dataclass
from funcsforprajay.funcs import save_pkl, load_pkl
import os
import funcsforprajay.funcs as pj

import numpy as np
import pandas as pd

from _exp_metainfo_.data_paths import CSV_PATH_ao, EXPMETA_path, CSV_PATH_1p


# UTILS
def import_meta_from_csv(csv_path=CSV_PATH_ao):
    metainfo = pd.read_csv(csv_path).iloc[:, 1:]
    return metainfo

def import_resultsobj(pkl_path: str):
    assert os.path.exists(pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f"\nimporting resultsobj from: {pkl_path} ... ")
        try:
            resultsobj = pickle.load(f)
        # except ModuleNotFoundError:
        #     print(f"WARNING: needing to try using CustomUnpickler!")
        #     return CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()
        except ModuleNotFoundError:
            from _utils_.io import CustomUnpicklerModuleNotFoundError, CustomUnpicklerAttributeError
            print(f"WARNING: needing to try using CustomUnpicklerModuleNotFoundError!")
            try:
                resultsobj = CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()
            except AttributeError:
                print(f"WARNING: needing to try using CustomUnpicklerAttributeError! <- from ModuleNotFoundError")
                try:
                    resultsobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()
                except ModuleNotFoundError:
                    print(
                        f"WARNING: needing to try using CustomUnpicklerModuleNotFoundError! <- from AttributeError from ModuleNotFoundError")
                    resultsobj = CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()

        print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
    return resultsobj


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

@dataclass
class FigureSettings:
    """class to hold various values to standardize across all figures"""
    colors = {'baseline': '#5777c6',  # light matte blue
               'interictal': '#77c65a',  # like forest green
               'ictal': 'slateblue',  # like a light matte purple
               'ictal - outsz': '#cd7537',  # dark dark orange
               'ictal - insz': '#f1d49d',  # very pale orange
               'gcamp - FOV': '#208b23',
               'general': '#9f9f9f',  # just like a basic grey
               'stim span': '#ffd9df',  # very pale pinkish
               '1p stim span': '#cbe2e4'  # very pale pinkish
               }

    lw= {
        "gcamp - single cell": 0.5,
        "gcamp - FOV": 0.5,
        "lfp": 0.15}

    fontsize = {
            "title": 9,
            "label": 12,
            "extraplot": 10,
            "intraplot": 10
        }

class ExpMetainfo:
    csv_path: str = CSV_PATH_ao
    alloptical: AllOpticalExpsToAnalyze = AllOpticalExpsToAnalyze()
    onephotonstim: OnePhotonStimExpsToAnalyze = OnePhotonStimExpsToAnalyze()
    figures: FigureSettings = FigureSettings()

    figure_settings = {
        # "fontsize": {
        #     "title": 10,
        #     "label": 12,
        #     "extraplot": 10,
        #     "intraplot": 8
        # },
        # 'lw': {
        #     "gcamp - single cell": 0.5,
        #     "gcamp - FOV": 0.5,
        #     "lfp": 0.15},
        'colors': {'baseline': '#5777c6',  # light matte blue
                    'interictal': '#77c65a',  # like forest green
                    'ictal': 'slateblue',  # like a light matte purple
                   'gcamp - FOV': '#208b23',
                  'general': '#d3bbad',
                  'stim span': '#ffd9df'  # very pale pinkish
                   },
        # 'colors': {'ictal': '#775d90',  # light matte orange
        #             # 'interictal': '#a1d18d',  # like lightish matte green
        #             'interictal': '#ac8feb',  # like lightish matte purple
        #             'baseline': '#eb7625',  # like a light matte purple
        #             'gcamp - FOV': '#208b23',
        #            'general': '#9faad9'
        #             },

        "fontsize - extraplot": 10,
        "fontsize - intraplot": 8,
        "fontsize - title": 10,
        "fontsize - label": 12,
        "lfp - lw": 0.15,
        "gcamp - single cell - lw": 0.5,
        "gcamp - FOV - lw": 0.5,

    }


    def __init__(self):
        pass

    def save(self):
        save_pkl(self, EXPMETA_path)

    @staticmethod
    def load():
        return load_pkl(EXPMETA_path)

    def return_exp_paths(self, exps: list = None, exptype: str = 'All optical'):
        from _utils_.io import import_expobj
        from _utils_.io import import_1pexobj

        if exps is None and exptype == 'All optical':
            exps = pj.flattenOnce(self.alloptical.pre_4ap_trials + self.alloptical.post_4ap_trials)
        elif exptype == 'One P stim':
            exps = self.onephotonstim.pre_4ap_trials + self.onephotonstim.post_4ap_trials
        else:
            AttributeError('exptype is unclear.')
        paths = []
        for exp in exps:
            expobj = import_expobj(exp_prep=exp) if exptype == 'All optical' else import_1pexobj(exp_prep=exp, verbose=False)
            paths.append(expobj.pkl_path)
        print(f'\nPaths of `exps` requested for {exptype} experiments: \n')
        for path in paths: print(f"{path}")
        return paths


# %%

baseline_color = ExpMetainfo.figures.colors['baseline']
interictal_color = ExpMetainfo.figures.colors['interictal']
insz_color = ExpMetainfo.figures.colors['ictal - insz']
outsz_color = ExpMetainfo.figures.colors['ictal - outsz']
general_color = ExpMetainfo.figures.colors['general']

fontsize_extraplot = ExpMetainfo.figures.fontsize['extraplot']
fontsize_intraplot = ExpMetainfo.figures.fontsize['intraplot']

# %%
if __name__ == '__main__':
    try:
        expmeta = ExpMetainfo.load()
    except Exception:
        expmeta = ExpMetainfo()
        expmeta.save()

    expmeta.return_exp_paths(exptype='One P stim')
