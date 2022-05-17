#
import pickle
import time

import os

from _main_.AllOpticalMain import alloptical


class CustomUnpicklerAttributeError(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PhotostimResponsesNonTargetsResults':
            print(f'\t for: PhotostimResponsesNonTargetsResults')
            from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
                PhotostimResponsesNonTargetsResults
            return PhotostimResponsesNonTargetsResults

        return super().find_class(module, name)


class CustomUnpicklerModuleNotFoundError(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '_analysis_.ClassPhotostimResponseQuantificationSLMtargets':
            renamed_module = "_analysis_._ClassPhotostimResponseQuantificationSLMtargets"

        elif module == '_analysis_._ClassPhotostimResponseQuantificationNonTargets':
            renamed_module = "_analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets"

        elif module == '_analysis_._ClassExpSeizureAnalysis':
            renamed_module = "_analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis"

        else:
            renamed_module = module

        return super().find_class(renamed_module, name)


class Quantification:
    """generic parent for Quantification subclasses """

    save_path: str = None
    _pre_stim_sec = 1
    _post_stim_sec = 3
    pre_stim_response_window_msec = 500  # msec
    post_stim_response_window_msec = 500  # msec

    def __init__(self, expobj: alloptical):
        self._metainfo = expobj.metainfo
        self._fps = expobj.fps
        print(f'\- ADDING NEW Quantification MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"Quantification Analysis submodule for expobj <{self.expobj_id}>"

    @classmethod
    def saveclass(cls):
        assert cls.save_path, print(f"class save path not defined for class: {cls}")
        from _utils_.io import save_cls_pkl
        save_cls_pkl(cls, cls.save_path)

    @property
    def expobj_id(self):
        return f"{self._metainfo['animal prep.']} {self._metainfo['trial']}"

    @property
    def expobj_exptype(self):
        return self._metainfo['exptype']

    # analysis properties
    @property
    def pre_stim_sec(self):
        return self._pre_stim_sec

    @property
    def post_stim_sec(self):
        return self._post_stim_sec

    @property
    def pre_stim_fr(self):
        return int(self._pre_stim_sec * self._fps)  # length of pre stim trace collected (in frames)

    @property
    def post_stim_fr(self):
        return int(self._post_stim_sec * self._fps)  # length of post stim trace collected (in frames)

    @property
    def pre_stim_response_frames_window(self):
        return int(
            self._fps * self.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)

    @property
    def post_stim_response_frames_window(self):
        return int(
            self._fps * self.post_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)


class Results:
    """generic parent for Results subclasses"""

    SAVE_PATH: str = None

    def __init__(self):
        pass

    def __repr__(self):
        return f"Results Analysis Object"

    def __str__(self):

        try:
            lastmod = time.ctime(os.path.getmtime(self.SAVE_PATH))
        except FileNotFoundError:
            lastmod = '-- cannot get --'

        information = ''
        for attr in [*self.__dict__]:
            len_ = len(self.__getattribute__(attr)) if self.__getattribute__(attr) is not None else -1
            information += f"\n\t{attr}: {len_} items" if len_ > 0 else f"\n\t{attr}: {self.__getattribute__(attr)}"

        return f"Results Analysis submodule, last saved: {lastmod}, contains: \n{information}"

    def save_results(self):
        assert self.SAVE_PATH, print(f"save path not defined for: {self}")
        from funcsforprajay.funcs import save_pkl
        print(f'\n Saving {self.__repr__()} ... ')
        save_pkl(self, self.SAVE_PATH)

    @classmethod
    def load(cls):
        from funcsforprajay.funcs import load_pkl
        print(f'Loading Results Analysis object from {cls.SAVE_PATH} ... ')
        try:
            return load_pkl(cls.SAVE_PATH)
        except AttributeError:
            CustomUnpicklerAttributeError(open(cls.SAVE_PATH, 'rb')).load()
        except ModuleNotFoundError:
            CustomUnpicklerModuleNotFoundError(open(cls.SAVE_PATH, 'rb')).load()

    # @property
    # def expobj_id(self):
    #     return f"{self._metainfo['animal prep.']} {self._metainfo['trial']}"
    #
    # @property
    # def expobj_exptype(self):
    #     return self._metainfo['exptype']
