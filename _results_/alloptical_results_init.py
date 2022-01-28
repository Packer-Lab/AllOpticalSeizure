# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import os
import sys

sys.path.append('/home/pshah/Documents/code/')
import time
import pandas as pd
import pickle

## SET OPTIONS
pd.set_option('max_columns', None)
pd.set_option('max_rows', 100)

from _main_.TwoPhotonImagingMain import TwoPhotonImaging
import _alloptical_utils as aoutils

# RESULTS OBJECTS
class OnePhotonResults:
    def __init__(self, save_path: str):
        # just create an empty class object that you will throw results and analyses into
        self.pkl_path = save_path

        self.save(pkl_path=self.pkl_path)

    def __repr__(self):
        lastmod = time.ctime(os.path.getmtime(self.pkl_path))
        return repr(f"OnePhotonResults experimental results object, last saved: {lastmod}")


    def save(self, pkl_path: str = None):
        TwoPhotonImaging.save_pkl(self, pkl_path=pkl_path)


class AllOpticalResults:  ## initiated in allOptical-results.ipynb
    def __init__(self, save_path: str):
        print(f"\n {'--'*5} CREATING NEW ALLOPTICAL RESULTS {'--'*5} \n")
        # just create an empty class object that you will throw results and analyses into
        self.pkl_path = save_path

        self.metainfo = pd.DataFrame(columns=['prep_trial', 'date', 'exptype'])  # gets filled in alloptical_results_init.py

        ## DATA CONTAINING ATTRS
        self.slmtargets_stim_responses = pd.DataFrame({'prep_trial': [], 'date': [], 'exptype': [],
                                                       'stim_setup': [],
                                                       'mean response (dF/stdF all targets)': [],
                                                       'mean response delta(trace_dFF) all targets)': [],  # TODO this is the field to fill with mean photostim responses .21/11/25
                                                       'mean reliability (>0.3 dF/stdF)': []})  # gets filled in allOptical-results.ipynb

        # large dictionary containing direct run_pre4ap_trials and run_post4ap_trials trial comparisons for each experiments, and stim responses
        # for run_pre4ap_trials data and stim responses for run_post4ap_trials data (also broken down by outsz and insz) - responses are dF/prestimF
        self.stim_responses = {}  # get defined in alloptical_analysis_photostim

        self.avgTraces = {}  # dictionary containing avg traces for each experiment type (pre4ap, outsz, insz) --> processing type (dfstdf or delta(trace_dFF)) _ response type (success or failures)

        # for run_pre4ap_trials data and stim responses for run_post4ap_trials data (also broken down by outsz and insz) - responses are taken using whole trace dFF
        self.stim_responses_tracedFF = {}  # get defined in alloptical_analysis_photostim

        # responses of targets at each stim (timed relative to the closest sz onset location) - responses are dF/prestimF
        self.stim_relative_szonset_vs_avg_response_alltargets_atstim = {}

        # responses of targets at each stim (timed relative to the closest sz onset location) - using whole trace dFF
        self.stim_relative_szonset_vs_deltatracedFFresponse_alltargets_atstim = {}


        self.stim_responses_zscores = {}  # zscores of photostim responses - zscored to pre4ap trials



        self.save_pkl(pkl_path=self.pkl_path)

    def __repr__(self):
        lastmod = time.ctime(os.path.getmtime(self.pkl_path))
        return repr(f"AllOpticalResults experimental results object, last saved: {lastmod}")

    def save_pkl(self, pkl_path: str = None):
        if pkl_path is None:
            if hasattr(self, 'save_path'):
                pkl_path = self.pkl_path
            else:
                raise ValueError(
                    'pkl path for saving was not found in object attributes, please provide path to save to')
        else:
            self.pkl_path = pkl_path

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self, f)
        print("\n\t -- alloptical results obj saved to %s -- " % pkl_path)

    def save(self):
        self.save_pkl()


# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'

force_remake = False

if not os.path.exists(results_object_path) or force_remake:
    allopticalResults = aoutils.AllOpticalResults(save_path=results_object_path)
    # make a metainfo attribute to store all metainfo types of info for all experiments/trials
    allopticalResults.metainfo = allopticalResults.slmtargets_stim_responses.loc[:, ['prep_trial', 'date', 'exptype']]

allopticalResults = aoutils.import_resultsobj(results_object_path)



# %% 1) lists of trials to analyse for run_pre4ap_trials and run_post4ap_trials trials within experiments,
# note that the commented out trials are used to skip running processing code temporarily

allopticalResults.pre_4ap_trials = [
    ['RL108 t-009'],
    ['RL108 t-010'],
    ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],
    ['RL109 t-014'],
    ['PS04 t-012',  #, 'PS04 t-014',  # - not sure what's wrong with PS04 t-014, but the photostim and Flu are falling out of sync .21/10/09
    'PS04 t-017'],     # just commented out until t-018 gets fully sorted out again.
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

allopticalResults.post_4ap_trials = [
    ['RL108 t-013'],
    ['RL108 t-011'],  # -- need to do sz boundary classifying processing
    ['RL109 t-020'],  # -- need to do sz boundary classifying processing
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016',  'RL109 t-017'], # -- need to do sz boundary classifying processing
    ['PS04 t-018'],  # - need to re run collecting slmtargets_szboundary_stim (sz processing)
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

assert len(allopticalResults.pre_4ap_trials) == len(allopticalResults.post_4ap_trials), (
f"# of pre trials: {len(allopticalResults.pre_4ap_trials)}",
f"# of post trials: {len(allopticalResults.post_4ap_trials)}")

allopticalResults.trial_maps = {'pre': {}, 'post': {}}
allopticalResults.trial_maps['pre'] = {
    'a': ['RL108 t-009'],
    'b': ['RL108 t-010'],
    'c': ['RL109 t-007'],
    'd': ['RL109 t-008'],
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
}

allopticalResults.trial_maps['post'] = {
    'a': ['RL108 t-013'],
    'b': ['RL108 t-011'], # -- need to redo sz boundary classifying processing - should be done
    'c': ['RL109 t-020'], # -- need to redo sz boundary classifying processing - should be done
    'd': ['RL109 t-021'],
    'e': ['RL109 t-018'], # -- need to redo sz boundary classifying processing - should be done
    'f': ['RL109 t-016', 'RL109 t-017'], #-- need to do sz boundary classifying processing - should be done
    'g': ['PS04 t-018'],  # -- need to redo sz boundary classifying processing - should be done
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
}

assert len(allopticalResults.trial_maps['pre'].keys()) == len(allopticalResults.trial_maps['post'].keys()), (
f"# of pre trials: {len(allopticalResults.trial_maps['pre'].keys())}",
f"# of post trials: {len(allopticalResults.trial_maps['post'].keys())}")

allopticalResults.save()
