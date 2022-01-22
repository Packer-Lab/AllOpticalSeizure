# retrieving and processing on LFP recordings from the .paq file
import os.path

import numpy as np
import seaborn as sns
import _alloptical_utils as Utils
import matplotlib.pyplot as plt

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
