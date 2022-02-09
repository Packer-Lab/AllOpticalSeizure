## utilities for creating anndata formatted data storage for each expobj

import anndata as ad
from typing import Optional, Union

# create anndata object for SLMtargets data
import numpy as np
import pandas as pd

import alloptical_utils_pj


def create_anndata_SLMtargets(expobj: Union[alloptical_utils_pj.alloptical, alloptical_utils_pj.Post4ap]):
    """
    Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial.

    """

    if hasattr(expobj, 'dFF_SLMTargets') or hasattr(expobj, 'raw_SLMTargets'):
        # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
        # build dataframe for obs_meta from SLM targets information
        obs_meta = pd.DataFrame(
            columns=['SLM group #', 'SLM target coord'], index=range(expobj.n_targets_total))
        for groupnum, targets in enumerate(expobj.target_coords):
            for target, coord in enumerate(targets):
                obs_meta.loc[target, 'SLM group #'] = groupnum
                obs_meta.loc[target, 'SLM target coord'] = coord

        # build numpy array for multidimensional obs metadata
        obs_m = {'SLM targets areas': []}
        for target, areas in enumerate(expobj.target_areas):
            obs_m['SLM targets areas'].append(np.asarray(areas))
        obs_m['SLM targets areas'] = np.asarray(obs_m['SLM targets areas'])

        # SETUP THE VARIABLES ANNOTATIONS TO USE IN anndata
        # build dataframe for var annot's - based on stim_start_frames
        var_meta = pd.DataFrame(index=['im_time_secs', 'stim_start_frame', 'wvfront in sz', 'seizure location'],
                                columns=range(len(expobj.stim_start_frames)))
        for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
            if 'pre' in expobj.exptype:
                var_meta.loc['wvfront in sz', fr_idx] = None
                var_meta.loc['seizure location', fr_idx] = None
            elif 'post' in expobj.exptype:
                if stim_frame in expobj.stimsWithSzWavefront:
                    var_meta.loc['wvfront in sz', fr_idx] = True
                    # var_meta.loc['seizure location', fr_idx] = '..not-set-yet..'
                    var_meta.loc['seizure location', fr_idx] = (
                    expobj.stimsSzLocations.coord1[stim_frame], expobj.stimsSzLocations.coord2[stim_frame])
                else:
                    var_meta.loc['wvfront in sz', fr_idx] = False
                    var_meta.loc['seizure location', fr_idx] = None
            var_meta.loc['stim_start_frame', fr_idx] = stim_frame
            var_meta.loc['im_time_secs', fr_idx] = stim_frame * expobj.fps

        # BUILD LAYERS TO ADD TO anndata OBJECT
        layers = {'SLM Targets photostim responses (dFstdF)': expobj.responses_SLMtargets_dfprestimf
                  }

        print(f"\n\----- CREATING annotated data object using AnnData:")

        # set primary data
        _data_type = 'SLM Targets photostim responses (tracedFF)'
        # expobj.responses_SLMtargets_tracedFF.columns = expobj.stim_start_frames
        expobj.responses_SLMtargets_tracedFF.columns = range(len(expobj.stim_start_frames))
        photostim_responses = expobj.responses_SLMtargets_tracedFF

        # create anndata object
        adata = AnnotatedData(X=photostim_responses, obs=obs_meta, var=var_meta.T, obsm=obs_m, layers=layers,
                              data_label=_data_type)

        print(f"\n{adata}")
        expobj.slmtargets_data = adata
        expobj.save()
    else:
        Warning(
            'did not create anndata. anndata creation only available if experiments were processed with suite2p and .paq file(s) provided for temporal synchronization')


class AnnotatedData(ad.AnnData):
    """Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial."""

    def __init__(self, X, obs, var: Optional=None, data_label=None, **kwargs):
        _adata_dict = {'X': X, 'obs': obs, 'var': var}
        for key in [*kwargs]:
            _adata_dict[key] = kwargs[key]

        super().__init__(**_adata_dict)
        self.data_label = data_label if data_label else None

    def __str__(self):
        "return more extensive information about the AnnotatedData data structure"
        if self.filename:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""

        descr = f"Annotated Data of n_obs (# ROIs) × n_vars (# Frames) = {self.n_obs} × {self.n_vars} {backed_at}"
        descr += f"\navailable attributes: "

        descr += f"\n\t.X (primary datamatrix) of .data_label: \n\t\t|- {str(self.data_label)}" if self.data_label else f"\n\t.X (primary datamatrix)"
        descr += f"\n\t.obs (ROIs metadata): \n\t\t|- {str(list(self.obs.keys()))[1:-1]}"
        descr += f"\n\t.var (frames metadata): \n\t\t|- {str(list(self.var.keys()))[1:-1]}"
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
                descr += f"\n\t{attr}: \n\t\t|- {str(list(keys))[1:-1]}"
        return descr


    def _gen_repr(self, n_obs, n_vars) -> str:  # overriding base method from AnnData
        """overrides the default anndata _gen_repr_() method for imaging data usage."""

        return f"Annotated Data of n_obs (# ROIs) × n_vars (# Frames) = {n_obs} × {n_vars}"



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