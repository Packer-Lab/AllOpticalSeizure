#
import time

import os

from _main_.AllOpticalMain import alloptical


class Quantification:
    """placeholder for now. some ideas for items to add to this class:

    """

    save_path: str = None

    def __init__(self, expobj: alloptical):
        self._metainfo = expobj.metainfo
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


class Results:
    """placeholder for now. some ideas for items to add to this class:

    """

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
            information += f"\n\t{attr}: {len_}" if len_ > 0 else f"\n\t{attr}: {self.__getattribute__(attr)}"

        return f"Results Analysis submodule, last saved: {lastmod}"

    def save_results(self):
        assert self.SAVE_PATH, print(f"save path not defined for: {self}")
        from funcsforprajay.funcs import save_pkl
        print(f'\n Saving {self.__repr__()}: ')
        save_pkl(self, self.SAVE_PATH)

    @classmethod
    def load(cls):
        from funcsforprajay.funcs import load_pkl
        print(f'Loading Results Analysis object from {cls.SAVE_PATH} ... ')
        return load_pkl(cls.SAVE_PATH)

    # @property
    # def expobj_id(self):
    #     return f"{self._metainfo['animal prep.']} {self._metainfo['trial']}"
    #
    # @property
    # def expobj_exptype(self):
    #     return self._metainfo['exptype']

