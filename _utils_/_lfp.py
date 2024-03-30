# retrieving and processing on LFP recordings from the .paq file
import os.path
from typing import Union

from _utils_ import _alloptical_utils as Utils
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _main_.TwoPhotonImagingMain import TwoPhotonImaging


class LFP:
    def __init__(self, **kwargs):
        self.lfp_from_paq(kwargs['paq_path']) if 'paq_path' in [*kwargs] else KeyError('no `paq_path` provided to load LFP from.')

    def lfp_from_paq(self, paq_path):

        print('\n----- retrieving LFP from paq file...')

        if not os.path.exists(paq_path):
            raise FileNotFoundError(f"Not found: {paq_path}")

        paq, _ = Utils.paq_read(paq_path, plot=True)
        self.paq_rate = paq['rate']

        # find voltage (LFP recording signal) channel and save as lfp_signal attribute
        voltage_idx = paq['chan_names'].index('voltage')
        self.lfp_signal = paq['data'][voltage_idx]


    @staticmethod
    def downsampled_LFP(expobj: Union[TwoPhotonImaging, alloptical, Post4ap]):
        # option for downsampling of data plot trace
        x = range(len(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual]))
        signal_cropped = expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual]
        down = 1000
        signal_down = signal_cropped[::down]
        x = x[::down]
        assert len(x) == len(signal_down), print('something went wrong with the downsampling')

        return signal_cropped, signal_down
