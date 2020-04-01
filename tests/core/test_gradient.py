# -*- coding: utf-8 -*-

"""This test module tests the gradient calculation returned by the
core/LLHRatio.evaluate method.
"""


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

        bp = '/data/user/mwolf/data/'
        dsc = data_samples['NorthernTracks_v003p01_KDE_PDF_v006'].create_dataset_collection(base_path=bp)
        datasets = dsc.get_datasets('IC79 IC86 MC')

        # Define the point source at TXS position.
        src_ra  = np.radians(77.358)
        src_dec = np.radians(5.693)
        source = PointLikeSource(src_ra, src_dec)

        tl = TimeLord()

        # Create analysis
        yr=365.25 # days per year
        with tl.task_timer('Creating analysis.') as tt:
            self.ana = create_analysis(
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
        wd = '/data/user/hmniederhausen/point_sources/skyllh/test_trials/'
        pseudo_data = load_pseudo_data(os.path.join(wd, 'trial_data_rss4_gamma20_nsig40.pkl'))
        mean_n_sig, n_sig, n_events_list, events_list = pseudo_data
        self.ana.initialize_trial(events_list, n_events_list)


    def test_gradient(self, tol=1.e-4):

        def eval_finite_difference(x, func, eps=1.e-8):
            ns, gamma = x

            f_plus, _ = func(np.asarray([ns, gamma+eps]))
            f_minus, _ = func(np.asarray([ns, gamma-eps]))
            dgamma = (f_plus-f_minus)/(2*eps)

            f_plus, _ = func(np.asarray([ns+eps, gamma]))
            f_minus, _ = func(np.asarray([ns-eps, gamma]))
            dns = (f_plus-f_minus) / (2*eps)
            return np.asarray([dns, dgamma])

        # Check gradients at function minimum, near function minimum
        # and far from function minimum.
        xbest = np.asarray([32.31368553,1.986462])
        x0 = xbest + np.asarray([0.0, 1.e-2])
        x1 = xbest + np.asarray([1.e-2, 0.0])
        x2 = xbest + np.asarray([0.0, 0.3])
        x3 = xbest + np.asarray([40, 0.0])
        x4 = xbest + np.asarray([-20, 0.0])

        for tx in [xbest, x0, x1, x2, x3, x4]:
            grad_approx = eval_finite_difference(tx, self.ana.llhratio.evaluate)
            _, grad = self.ana.llhratio.evaluate(tx)
            delta = (grad_approx - grad) / (grad_approx + grad)
            delta_ns, delta_gamma = delta

            self.assertTrue(delta_ns<tol)
            self.assertTrue(delta_gamma<tol)


if(__name__ == '__main__'):
    unittest.main()

