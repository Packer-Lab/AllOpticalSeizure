import os
import re
import pickle
from funcsforprajay import funcs as pj
from _exp_metainfo_.exp_metainfo import ExpMetainfo

# %% HANDLING PICKLING ERRORS


# this is used when the unpickler has a problem with finding a class attribute for the file being loaded - note that it is setup manually for each one..
# these are needed when a module or class or attribute gets moved after pickling an object, the new location needs to be provided explicitly
# the solution is to override the find_class method of pickle.Unpickler to provide the new location for the moved attributes/classes/modules


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
    except:
        raise ImportError(f"\n** FAILED IMPORT OF * {prep} {trial} * from {original_path}\n")
    print(f'|- Loaded backup of: {expobj.t_series_name} ({load_backup_path}) ... DONE')
    return expobj

class CustomUnpicklerAttributeError(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PhotostimResponsesQuantificationSLMtargets':
            from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import \
                PhotostimResponsesQuantificationSLMtargets
            return PhotostimResponsesQuantificationSLMtargets
        elif name == '_TargetsSzInvasionTemporal':
            from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal
            return TargetsSzInvasionTemporal
        elif name == 'TargetsSzInvasionSpatial':
            from _analysis_._ClassTargetsSzInvasionSpatial import TargetsSzInvasionSpatial
            return TargetsSzInvasionSpatial

        return super().find_class(module, name)


class CustomUnpicklerModuleNotFoundError(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '_analysis_.ClassPhotostimResponseQuantificationSLMtargets':
            renamed_module = "_analysis_._ClassPhotostimResponseQuantificationSLMtargets"

        elif module == '_sz_processing.ClassTargetsSzInvasionTemporal':
            renamed_module = "_analysis_._ClassTargetsSzInvasionTemporal"

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
        if not hasattr(obj, 'save_path'):
            raise ValueError(
                'pkl path for saving was not found in object attributes, please provide path to save to')
    else:
        obj.pkl_path = save_path

    os.makedirs(pj.return_parent_dir(save_path), exist_ok=True)
    with open(obj.pkl_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"\- expobj saved to {obj.pkl_path} -- ")
    
    os.makedirs(pj.return_parent_dir(obj.backup_pkl), exist_ok=True) if not os.path.exists(backup_dir) else None
    with open(obj.backup_pkl, 'wb') as f:
        pickle.dump(obj, f)



def import_expobj(aoresults_map_id: str = None, trial: str = None, prep: str = None, date: str = None, pkl_path: str = None,
                  exp_prep: str = None, verbose: bool = False, load_backup_path: str = None):
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
        if 'pre' in aoresults_map_id:
            exp_type = 'pre'
        elif 'post' in aoresults_map_id:
            exp_type = 'post'
        id = aoresults_map_id.split(' ')[1][0]
        if len(allopticalResults.trial_maps[exp_type][id]) > 1:
            num_ = int(re.search(r"\d", aoresults_map_id)[0])
        else:
            num_ = 0
        prep, trial = allopticalResults.trial_maps[exp_type][id][num_].split(' ')

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
            # try:
            #     date = allopticalResults.metainfo.loc[
            #         allopticalResults.metainfo['prep_trial'] == f"{prep} {trial}", 'date'].values[0]
            # except KeyError:
            #     raise KeyError('not able to find date in allopticalResults.metainfo')
        pkl_path = f"/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}/{date}_{trial}.pkl"
        pkl_path_local = f"/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/expobj/{date}_{trial}.pkl"

        for path in [pkl_path, pkl_path_local]:
            if os.path.exists(path):
                pkl_path = path
                break

    if not os.path.exists(pkl_path):
        raise Exception('pkl path NOT found: ' + pkl_path)
    try:
        with open(pkl_path, 'rb') as f:
            print(f'\- Loading {pkl_path}', end='\r')
            expobj = pickle.load(f)
            print(f'|- Loaded {expobj.t_series_name} (from {pkl_path}) ... DONE')
    except EOFError:
        expobj = load_from_backup(prep, trial, date, original_path=pkl_path)
    except pickle.UnpicklingError:
        expobj = load_from_backup(prep, trial, date, original_path=pkl_path)

        # ImportWarning(f"\n** FAILED IMPORT OF * {prep} {trial} * from {pkl_path}\n")
        # print(f"\t trying to recover from backup! ****")
        # load_backup_path = f'/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}' + f"backups/{date}_{prep}_{trial}.pkl"
        # if not os.path.exists(load_backup_path):
        #     load_backup_path = f'/home/pshah/mnt/qnap/Analysis/{date}/{prep}/{date}_{trial}' + f"/backups/{date}_{prep}_{trial}.pkl"
        #
        # try:
        #     with open(load_backup_path, 'rb') as f:
        #         print(f'\- Loading backup from: {load_backup_path}', end='\r')
        #         expobj = pickle.load(f)
        # except:
        #     raise ImportError(f"\n** FAILED IMPORT OF * {prep} {trial} * from {pkl_path}\n")
        # experiment = f"{expobj.t_series_name} {expobj.metainfo['exptype']} {expobj.metainfo['comments']}"
        # print(f'|- Loaded {experiment}') if verbose else print(f'|- Loaded {expobj.t_series_name} ({pkl_path}) .. DONE')
    except AttributeError:
        print(f"WARNING: needing to try using CustomUnpickler!")
        expobj = CustomUnpicklerAttributeError(open(pkl_path, 'rb')).load()
    except ModuleNotFoundError:
        print(f"WARNING: needing to try using CustomUnpickler!")
        expobj = CustomUnpicklerModuleNotFoundError(open(pkl_path, 'rb')).load()

    ### roping in some extraneous processing steps if there's expobj's that haven't completed for them

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


# %% RESULT OBJECT IO
def import_resultsobj(pkl_path: str):
    assert os.path.exists(pkl_path)
    with open(pkl_path, 'rb') as f:
        print(f"\nimporting resultsobj from: {pkl_path} ... ")
        resultsobj = pickle.load(f)
        print(f"|-DONE IMPORT of {(type(resultsobj))} resultsobj \n\n")
    return resultsobj


# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
local_results_object_path = '/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/expobj/alloptical_results_superobject.pkl'

for path in [results_object_path, local_results_object_path]:
    if os.path.exists(path):
        results_path = path
        try:
            allopticalResults = import_resultsobj(
                pkl_path=results_path)  # this needs to be run AFTER defining the AllOpticalResults class
        except FileNotFoundError:
            print(f'not able to get allopticalResults object from {results_object_path}')
        break

#
# # %%
# import pickle
#
# pkl_path = '/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/expobj/2020-12-18_t-013.pkl'
# with open(pkl_path, 'rb') as f:
#     print(f"\nimporting resultsobj from: {pkl_path} ... ")
#     resultsobj = pickle.load(f)
