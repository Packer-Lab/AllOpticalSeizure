## utilities for creating anndata formatted data storage for each expobj

import anndata as ad
from typing import Optional



class AnnotatedData(ad.AnnData):
    """Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial."""

    def __init__(self, X, obs, var: Optional=None, data_label=None, **kwargs):
        _adata_dict = {'X': X, 'obs': obs, 'var': var}
        for key in [*kwargs]:
            _adata_dict[key] = kwargs[key]

        super().__init__(**_adata_dict)
        self.data_label = data_label if data_label else None

    def __str__(self):
        "extensive information about the AnnotatedData data structure"
        if self.filename:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""

        descr = f"Annotated Data of n_obs (# ROIs) × n_vars (# Frames) = {self.n_obs} × {self.n_vars} {backed_at}"
        descr += f"\navailable attributes: "

        descr += f"\n\t.X (primary datamatrix, with .data_label): \n\t\t|-- {str(self.data_label)}" if self.data_label else f"\n\t.X (primary datamatrix)"
        descr += f"\n\t.obs (ROIs metadata) keys: \n\t\t|-- {str(list(self.obs.keys()))[1:-1]}"
        descr += f"\n\t.var (frames metadata) keys: \n\t\t|-- {str(list(self.var.keys()))[1:-1]}"
        for attr in [
            # "obs",
            # "var",
            ".uns",
            ".obsm",
            ".varm",
            ".layers",
            ".obsp",
            ".varp",
        ]:
            keys = getattr(self, attr[1:]).keys()
            if len(keys) > 0:
                descr += f"\n\t{attr} keys: \n\t\t|-- {str(list(keys))[1:-1]}"
        return descr


    def _gen_repr(self, n_obs, n_vars) -> str:  # overriding base method from AnnData
        """overrides the default anndata _gen_repr_() method for imaging data usage."""

        return f"Annotated Data of n_obs (# ROIs) × n_vars (# Frames) = {n_obs} × {n_vars}"

        # if self.filename:
        #     backed_at = f" backed at {str(self.filename)!r}"
        # else:
        #     backed_at = ""
        #
        # descr = f"Annotated Data of n_obs (# ROIs) × n_vars (# Frames) = {n_obs} × {n_vars} {backed_at}"
        # descr += f"\navailable attributes: "
        #
        # descr += f"\n\t.X (primary datamatrix, with .data_label): \n\t\t|-- {str(self.data_label)}" if self.data_label else f"\n\t.X (primary datamatrix)"
        # descr += f"\n\t.obs (ROIs metadata) keys: \n\t\t|-- {str(list(self.obs.keys()))[1:-1]}"
        # descr += f"\n\t.var (frames metadata) keys: \n\t\t|-- {str(list(self.var.keys()))[1:-1]}"
        # for attr in [
        #     # "obs",
        #     # "var",
        #     ".uns",
        #     ".obsm",
        #     ".varm",
        #     ".layers",
        #     ".obsp",
        #     ".varp",
        # ]:
        #     keys = getattr(self, attr[1:]).keys()
        #     if len(keys) > 0:
        #         descr += f"\n\t{attr} keys: \n\t\t|-- {str(list(keys))[1:-1]}"
        # return descr


    def add_observation(self, obs_name: str, values: list):
        """adds values to the observations of an anndata object, under the key obs_name"""
        assert len(values) == self.obs.shape[0], f"# of values to add doesn't match # of observations in anndata"
        self.obs[obs_name] = values

    def del_observation(self, obs_name: str): # TODO
        "removes a key from observations from an anndata object, of the key obs_name"

    def add_variables(self, var_name: str, values: list):
        """adds values to the variables of an anndata object, under the key var_name"""
        assert len(values) == self.var.shape[0], f"# of values to add doesn't match # of observations in anndata"
        self.var[var_name] = values

    def del_variables(self, obs_name: str): # TODO
        "removes a key from variables from an anndata object, of the key var_name"

    def extend_anndata(self, additional_adata: ad.AnnData, axis: int = 0):
        """
        :param adata_obj: an anndata object of dimensions n obs x m var
        :param additional_adata: an anndata object of dimensions n obs x # var or, # obs x m var (depending on which axis to extend)
        """
        adata = ad.concat([self, additional_adata], axis=axis)
        return adata