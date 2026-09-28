import pickle
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np

from skyllh.analyses.i3.publicdata_ps.utils import FctSpline1D
from skyllh.core.multiproc import parallelize


def create_spline(scale):
    return FctSpline1D(scale * np.array([1.0, 3.0, 2.0, 5.0, 4.0]), np.linspace(0, 5, 6), norm=True)


class FctSpline1DPickleTestCase(unittest.TestCase):
    def setUp(self):
        self.spline = create_spline(1.0)
        self.x = np.linspace(-1, 6, 71)

    def assert_equal_splines(self, spline):
        np.testing.assert_array_equal(spline(self.x), self.spline(self.x))
        np.testing.assert_array_equal(spline(self.x, oor_value=-1), self.spline(self.x, oor_value=-1))
        self.assertEqual(spline.norm, self.spline.norm)
        self.assertEqual(spline.x_min, self.spline.x_min)
        self.assertEqual(spline.x_max, self.spline.x_max)

    def test_state_does_not_contain_scipy_spline(self):
        # Some scipy versions (e.g. 1.18.0) cannot pickle PchipInterpolator
        # instances, hence the spline object must not be part of the state.
        self.assertNotIn('spl_f', self.spline.__getstate__())

    def test_pickle(self):
        self.assert_equal_splines(pickle.loads(pickle.dumps(self.spline)))

    def test_forking_pickler(self):
        self.assert_equal_splines(pickle.loads(bytes(ForkingPickler.dumps(self.spline))))

    def test_transfer_between_processes(self):
        splines = parallelize(func=create_spline, args_list=[((1.0,), {}), ((1.0,), {})], ncpu=2)
        for spline in splines:
            self.assert_equal_splines(spline)


if __name__ == '__main__':
    unittest.main()
