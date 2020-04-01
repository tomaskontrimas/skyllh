# -*- coding: utf-8 -*-

import unittest
import numpy as np
import os, sys
sys.path.insert(0, '/home/hans/software/skyllh_tools_dev3/skyllh')
sys.path.insert(0, '/home/hans/software/skyllh_tools_dev3/i3skyllh')
sys.path.insert(0, '/home/hans/software/photosplines')

import logging
import numpy as np

import skyllh
from skyllh.core.random import RandomStateService
from skyllh.core.timing import TimeLord
from skyllh.physics.source import PointLikeSource
from skyllh.core.utils.trials import load_pseudo_data

from i3skyllh.datasets import data_samples
from i3skyllh.datasets import repository
from i3skyllh.analyses.kdepdf_mcbg_ps.analysis import create_analysis

sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
from utils import isAlmostEqual




class LLHRatio_TestCase(unittest.TestCase):
    def setUp(self):
        # setup analysis

        bp = '/home/hans/icecube/projects/ps_llh/skyllh_tests/'
        dsc = data_samples['NorthernTracks_v003p01_KDE_PDF_v005'].create_dataset_collection(base_path=bp)
        datasets = dsc.get_datasets('IC79 IC86 MC')

        # Define the point source at TXS position.
        src_ra  = np.radians(77.358)
        src_dec = np.radians(5.693)
        source = PointLikeSource(src_ra, src_dec)

        tl = TimeLord()



        # Create analysis
        yr=365.25 # days per year
        with tl.task_timer('Creating analysis.') as tt:
            ana = create_analysis(
                datasets,
                source,
                bkg_event_rate_field_names=['astro', 'conv'],
                refplflux_gamma=2.0,
                gamma_seed = 2.0,
                fit_gamma=True,
                livetime_list=[10.0 * yr],
                compress_data=True,
                use_photosplines=True,
                tl=tl)

        rss_seed = 4
        wd = '/home/hans/icecube/projects/ps_llh/skyllh_tests/dev/store_trials/'
        pseudo_data = load_pseudo_data(os.path.join(wd, 'trial_data_rss4_gamma20_nsig40.pkl'))
        print(pseudo_data)


    def test_gradient(self):
