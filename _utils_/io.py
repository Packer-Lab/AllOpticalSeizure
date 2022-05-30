import os
import re
import pickle
from funcsforprajay import funcs as pj
from _exp_metainfo_.exp_metainfo import ExpMetainfo

# %% HANDLING PICKLING ERRORS


def load_from_backup(prep, trial, date, original_path, backup_path=None):
    ImportWarning(f"\n** FAILED IMPORT OF * {prep} {trial} * from {original_path}\n")
    print(f"\t trying to recover from backup! ****")
    load_backup_path = f'/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}' + f"backups/{date}_{prep}_{trial}.pkl" if backup_path is None else backup_path
    if not os.path.exists(load_backup_path):
        load_backup_path = f'/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}' + f"/backups/{date}_{prep}_{trial}.pkl"
    try:
        with open(load_backup_path, 'rb') as f:
            print(f'\- Loading backup from: {load_backup_path}', end='\r')
            expobj = pickle.load(f)
    except Exception:
        raise ImportError(f"\n** FAILED IMPORT OF * {prep} {trial} * from {original_path}\n")
    print(f'|- Loaded backup of: {expobj.t_series_name} ({load_backup_path}) ... DONE')
    return expobj


# this is used when the unpickler has a problem with finding a class attribute for the file being loaded - note that it is setup manually for each one..
# these are needed when a module or class or attribute gets moved after pickling an object, the new location needs to be provided explicitly
# the solution is to override the find_class method of pickle.Unpickler to provide the new location for the moved attributes/classes/modules
class CustomUnpicklerAttributeError(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PhotostimResponsesQuantificationSLMtargets':
            print(f'\t for: PhotostimResponsesQuantificationSLMtargets')
            from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesQuantificationSLMtargets
            return PhotostimResponsesQuantificationSLMtargets
        # elif name == 'TargetsSzInvasionTemporal':
        #     print(f'\t for: TargetsSzInvasionTemporal')
        #     from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal
        #     return TargetsSzInvasionTemporal
        elif name == 'NonTargetsResponsesSpatialAnalysis':
            print(f'\t for: NonTargetsResponsesSpatialAnalysis')
            from _analysis_.nontargets_analysis._ClassNonTargetsResponsesSpatial import \
                NonTargetsResponsesSpatialAnalysis
            return NonTargetsResponsesSpatialAnalysis

        elif name == 'NonTargetsSzInvasionSpatial':
            print(f'\t for: NonTargetsSzInvasionSpatial')
            from _analysis_.nontargets_analysis._ClassNonTargetsSzInvasionSpatial import NonTargetsSzInvasionSpatial
            return NonTargetsSzInvasionSpatial
        elif name == 'FakeStimsQuantification':
            print(f'\t for: FakeStimsQuantification')
            from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import FakeStimsQuantification
            return FakeStimsQuantification
        elif name == '_analysis_._ClassPhotostimResponsesAnalysisNonTargets':
            print(f'\t for: _analysis_._ClassPhotostimResponsesAnalysisNonTargets')
            from _analysis_.nontargets_analysis import _ClassPhotostimResponsesAnalysisNonTargets
            return _ClassPhotostimResponsesAnalysisNonTargets
        elif name == 'PhotostimResponsesAnalysisNonTargets':
            print(f'\t for: PhotostimResponsesAnalysisNonTargets')
            from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import PhotostimResponsesAnalysisNonTargets
            return PhotostimResponsesAnalysisNonTargets
        elif name == 'PhotostimResponsesQuantificationNonTargets':
            print(f'\t for: PhotostimResponsesQuantificationNonTargets')
            from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
                PhotostimResponsesQuantificationNonTargets
            return PhotostimResponsesQuantificationNonTargets
        elif name == 'PhotostimAnalysisSlmTargets':
            print(f'\t for: PhotostimAnalysisSlmTargets')
            from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
            return PhotostimAnalysisSlmTargets
        elif name == 'TargetsStimsSzOnsetTime':
            print(f'\t for: TargetsPhotostimResponsesInterictal')
            from _analysis_._ClassTargetsPhotostimResponsesInterictal import TargetsPhotostimResponsesInterictal
            return TargetsPhotostimResponsesInterictal
        elif name == 'Suite2pROIsSz':
            print(f'\t for: Suite2pROIsSz')
            from _analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis import Suite2pROIsSz
            return Suite2pROIsSz
        # elif name == 'TargetsPhotostimResponsesInterictalResults':
        #     print(f'\t for: TargetsPhotostimResponsesInterictalResults')
        #     from _analysis_._ClassTargetsSzOnsetTime import TargetsPhotostimResponsesInterictalResults
        #     return TargetsPhotostimResponsesInterictalResults
        elif name == 'TargetsSzInvasionSpatial':
            print(f'\t for: TargetsSzInvasionSpatial')
        #     from _analysis_._ClassTargetsSzInvasionSpatial import TargetsSzInvasionSpatial
        #     return TargetsSzInvasionSpatial
        # elif name == 'TargetsSzInvasionSpatial_codereview':
        #     print(f'\t for: TargetsSzInvasionSpatial_codereview')
            from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview
            return TargetsSzInvasionSpatial_codereview
        elif name == 'ExpSeizureAnalysis':
            print(f'\t for: ExpSeizureAnalysis')
            from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis
            return ExpSeizureAnalysis
        elif name == 'AnnotatedData':
            print(f'\t for: AnnotatedData')
            from _utils_._anndata import AnnotatedData2
            return AnnotatedData2
        elif name == 'OnePhotonStim':
            print(f'\t for: OnePhotonStim')
            from onePexperiment.OnePhotonStimMain import OnePhotonStim
            return OnePhotonStim
        elif name == 'alloptical':
            print(f'\t for: alloptical')
            from _main_.AllOpticalMain import alloptical
            return alloptical
        elif name == 'Post4ap':
            print(f'\t for: Post4ap')
            from _main_.Post4apMain import Post4ap
            return Post4ap
        elif name == 'TwoPhotonImaging':
            print(f'\t for: TwoPhotonImaging')
            from _main_.TwoPhotonImagingMain import TwoPhotonImaging
            return TwoPhotonImaging

        return super().find_class(module, name)


class CustomUnpicklerModuleNotFoundError(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '_analysis_.ClassPhotostimResponseQuantificationSLMtargets':
            renamed_module = "_analysis_._ClassPhotostimResponseQuantificationSLMtargets"

        elif module == '_analysis_._ClassExpSeizureAnalysis':
            renamed_module = "_analysis_.sz_analysis._ClassExpSeizureAnalysis"

        elif module == '_analysis_._ClassSuite2pROIsSzAnalysis':
            renamed_module = "_analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis"

        elif module == '_sz_processing.ClassTargetsSzInvasionTemporal':
            renamed_module = "_analysis_._ClassTargetsSzInvasionTemporal"

        elif module == '_analysis_._ClassTargetsSzInvasionSpatial':
            renamed_module = "_analysis_._ClassTargetsSzInvasionSpatial_codereview"

        elif module == '_analysis_._ClassTargetsSzOnsetTime':
            renamed_module = "_analysis_._ClassTargetsPhotostimResponsesInterictal"

        elif module == '_analysis_._ClassPhotostimResponsesAnalysisNonTargets':
            renamed_module = "_analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets"

        elif module == '_analysis_._ClassPhotostimResponseQuantificationNonTargets':
            renamed_module = "_analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets"

        elif module == '_ClassPhotostimResponseQuantificationNonTargets':
            renamed_module = "_analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets"

        elif module == '_analysis_._ClassNonTargetsSzInvasionSpatial':
            renamed_module = "_analysis_.nontargets_analysis._ClassNonTargetsSzInvasionSpatial"

        elif module == '_analysis_._ClassNonTargetsResponsesSpatial':
            renamed_module = "_analysis_.nontargets_analysis._ClassNonTargetsResponsesSpatial"

        else:
            renamed_module = module

        return super().find_class(renamed_module, name)


# %% CLASS IO

def save_cls_pkl(cls, save_path: str):
    if save_path is None:
        if not hasattr(cls, 'save_path'):
            raise ValueError(
                'pkl path for saving was not found in cls variables, please provide path to save to')
    else:
        cls.pkl_path = save_path

    os.makedirs(pj.return_parent_dir(save_path), exist_ok=True)
    with open(cls.pkl_path, 'wb') as f:
        pickle.dump(cls, f)
    print(f"\- cls saved to {cls.pkl_path} -- ")

def import_cls(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise Exception('pkl path NOT found: ' + pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f'\- Loading {pkl_path}', end='\r')
        try:
            cls = pickle.load(f)
            print(f'|- Loaded {cls} ... DONE')
            return cls
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"\n** FAILED IMPORT from {pkl_path}\n")
        except ModuleNotFoundError:
            print(f"WARNING: needing to try using CustomUnpickler!")
            return CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()


# %% EXPOBJ IO

def import_stripped_expobj(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise Exception('pkl path NOT found: ' + pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f'\- Loading {pkl_path}', end='\r')
        try:
            expobj = pickle.load(f)
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"\n** FAILED IMPORT from {pkl_path}\n")
        print(f'|- Loaded {expobj.t_series_name} ({pkl_path}) .. DONE')
    return expobj




def save_pkl(obj, save_path: str = None):
    if save_path is None:
        if not hasattr(obj, 'pkl_path'):
            raise ValueError(
                'pkl path for saving was not found in object attributes, please provide path to save to')
    else:
        obj.pkl_path = save_path

    os.makedirs(pj.return_parent_dir(save_path), exist_ok=True)
    try:
        with open(obj.pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"\- expobj saved to {obj.pkl_path} -- ")
    except:
        raise IOError(f'failed to save pkl object to: {obj.pkl_path}')
    os.makedirs(pj.return_parent_dir(obj.backup_pkl), exist_ok=True)
    with open(obj.backup_pkl, 'wb') as f:
        pickle.dump(obj, f)



def import_expobj(aoresults_map_id: str = None, trial: str = None, prep: str = None, date: str = None, pkl_path: str = None,
                  exp_prep: str = None, load_backup_path: str = None):
    """
    primary function for importing of saved expobj files saved pickel files.

    :param aoresults_map_id:
    :param trial:
    :param prep:
    :param date:
    :param pkl_path:
    :param verbose:
    :param do_processing: whether to do extra misc. processing steps that are the end of the importing code here.
    :return:
    """

    if aoresults_map_id is not None:
        exp_type = 'post'
        if 'pre' in aoresults_map_id:
            exp_type = 'pre'
        id = aoresults_map_id.split(' ')[1][0]
        if len(ExpMetainfo.alloptical.trial_maps[exp_type][id]) > 1:
            num_ = int(re.search(r"\d", aoresults_map_id)[0])
        else:
            num_ = 0
        prep, trial = ExpMetainfo.alloptical.trial_maps[exp_type][id][num_].split(' ')

    if exp_prep is not None:
        prep = exp_prep[:-6]
        trial = exp_prep[-5:]

    # if need to load from backup path!
    if load_backup_path:
        pkl_path = load_backup_path
        print(f"**** loading from backup path! ****")

    if pkl_path is None:
        if date is None:
            try:
                date = ExpMetainfo.alloptical.metainfo.loc[ExpMetainfo.alloptical.metainfo['prep_trial'] == f"{prep} {trial}", "date"].values[0]
            except KeyError:
                raise KeyError('not able to find date in ExpMetainfo.alloptical.metainfo')

        pkl_path = f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/{date}_{trial}.pkl"
        pkl_path_local = f"/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/expobj/{date}_{trial}.pkl"

        for path in [pkl_path, pkl_path_local]:
            if os.path.exists(path):
                pkl_path = path
                break

    if not os.path.exists(pkl_path):
        raise FileNotFoundError('pkl path NOT found: ' + pkl_path)
    try:
        with open(pkl_path, 'rb') as f:
            print(f'\- Loading {pkl_path}', end='\r')
            expobj = pickle.load(f)
            print(f'|- Loaded {expobj.t_series_name} (from {pkl_path}) ... DONE')
    except EOFError:
        expobj = load_from_backup(prep, trial, date, original_path=pkl_path)
    except pickle.UnpicklingError:
        expobj = load_from_backup(prep, trial, date, original_path=pkl_path)

    except AttributeError:
        print(f"WARNING: needing to try using CustomUnpicklerAttributeError!")
        try:
            expobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()
        except AttributeError:
            print(f"WARNING: needing to try using CustomUnpicklerAttributeError! <- from AttributeError")
            expobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()

    except ModuleNotFoundError:
        print(f"WARNING: needing to try using CustomUnpicklerModuleNotFoundError!")
        try:
            expobj = CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()
        except AttributeError:
            print(f"WARNING: needing to try using CustomUnpicklerAttributeError! <- from ModuleNotFoundError")
            try:
                expobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()
            except ModuleNotFoundError:
                print(f"WARNING: needing to try using CustomUnpicklerModuleNotFoundError! <- from AttributeError from ModuleNotFoundError")
                expobj = CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()

    ### roping in some extraneous processing steps if there's expobj's that haven't completed for them
    try:
        _fps = expobj.fps
    except AttributeError:
        expobj._parsePVMetadata()
        expobj.save()



    # check for existence of backup (if not then make one through the saving func).
    if 'OneDrive' not in pkl_path:
        expobj.save() if not os.path.exists(expobj.backup_pkl) else None

    # save the pkl if loaded from backup path
    expobj.save() if load_backup_path else None

    if expobj.analysis_save_path[-1] != '/':
        expobj.analysis_save_path = expobj.analysis_save_path + '/'
        print(f"updated expobj.analysis_save_path to: {expobj.analysis_save_path}")
        expobj.save()

    # move expobj to the official save_path from the provided save_path that expobj was loaded from (if different)
    if 'OneDrive' not in pkl_path:
        if pkl_path is not None:
            if expobj.pkl_path != pkl_path:
                expobj.save_pkl(save_path=expobj.pkl_path)
                print('saved new copy of expobj to save_path: ', expobj.pkl_path)

    # other misc. things you want to do when importing expobj -- should be temp code basically - not essential for actual importing of expobj

    return expobj


def import_1pexobj(prep=None, trial=None, date=None, pkl_path=None, verbose=False, load_backup_path=None):
    # if need to load from backup path!
    if load_backup_path:
        pkl_path = load_backup_path
        print(f"**** loading from backup path! ****")

    if pkl_path is None:
        if date is None:
            try:
                from onePexperiment.OnePhotonStimMain import onePresults
                date = onePresults.mean_stim_responses.loc[(onePresults.mean_stim_responses.Prep == f"{prep}") & (
                            onePresults.mean_stim_responses.Trial == f'{trial}'), 'pkl_list'].values[0][30:40]
            except ValueError:
                raise ValueError('not able to find date in allopticalResults.metainfo')
        pkl_path = f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/{date}_{trial}.pkl"

    if not os.path.exists(pkl_path):
        raise Exception('pkl path NOT found: ' + pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f'\- Loading {pkl_path}', end='\r')
        try:
            expobj = pickle.load(f)
            if expobj.analysis_save_path != f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/":
                expobj.analysis_save_path = f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/"
                expobj.save_pkl(pkl_path=expobj.pkl_path)
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"\n** FAILED IMPORT OF * {prep} {trial} * from {pkl_path}\n")
        except AttributeError:
            print(f"WARNING: needing to try using CustomUnpicklerAttributeError!")
            expobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()

        experiment = f"{expobj.t_series_name} {expobj.metainfo['exptype']} {expobj.metainfo['comments']}"
        print(f'|- Loaded {expobj.t_series_name} {expobj.metainfo["exptype"]}')
        print(f'|- Loaded {experiment}') if verbose else None


    ### roping in some extraneous processing steps if there's expobj's that haven't completed for them
    try:
        _fps = expobj.fps
    except AttributeError:
        expobj._parsePVMetadata()
        expobj.save()

    return expobj, experiment



# # import results superobject that will collect analyses from various individual experiments
# results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
# local_results_object_path = '/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/expobj/alloptical_results_superobject.pkl'
#
# for path in [results_object_path, local_results_object_path]:
#     if os.path.exists(path):
#         results_path = path
#         try:
#             allopticalResults = import_resultsobj(
#                 pkl_path=results_path)  # this needs to be run AFTER defining the AllOpticalResults class
#         except FileNotFoundError:
#             print(f'not able to get allopticalResults object from {results_object_path}')
#         break

#
# # %%
# import pickle
#
# pkl_path = '/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/expobj/2020-12-18_t-013.pkl'
# with open(pkl_path, 'rb') as f:
#     print(f"\nimporting resultsobj from: {pkl_path} ... ")
#     resultsobj = pickle.load(f)
