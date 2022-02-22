#
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

