"""The unit tests in this module test the building blocks of time-dependent
analyses: the tabulated time flux profile, the detector on-time integrals and
time sampling of time flux profiles, the signal time PDF for changing and
non-overlapping time flux profiles, the time-dependent detector signal yield,
and the time window event selection method.
"""

import unittest
from unittest.mock import Mock

import numpy as np
import scipy.integrate
import scipy.stats

from skyllh.core.config import (
    Config,
)
from skyllh.core.dataset import (
    Dataset,
)
from skyllh.core.detsigyield import (
    DetSigYield,
    TimeDependentDetSigYield,
)
from skyllh.core.event_selection import (
    AllEventSelectionMethod,
    TimeWindowEventSelectionMethod,
)
from skyllh.core.flux_model import (
    BoxTimeFluxProfile,
    FluxModel,
    GaussianTimeFluxProfile,
    TabulatedTimeFluxProfile,
    UnityTimeFluxProfile,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.parameters import (
    ParameterModelMapper,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.signalpdf import (
    SignalTimePDF,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    SourceModel,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.utils.flux_model import (
    draw_ontimes_from_TimeFluxProfile,
    get_ontime_integral_of_TimeFluxProfile,
)

from .test_signalpdf import create_tdm


def create_livetime():
    """Creates a Livetime instance with three on-time intervals."""
    return Livetime(
        np.array(
            [
                [0, 1],
                [1.3, 4.6],
                [7.7, 10],
            ]
        )
    )


class LivetimeUptimeIntervalsTestCase(unittest.TestCase):
    def setUp(self):
        self.livetime = create_livetime()

    def test_get_uptime_intervals_between_overlapping(self):
        intervals = self.livetime.get_uptime_intervals_between(0.5, 8)
        np.testing.assert_almost_equal(intervals, np.array([[0.5, 1], [1.3, 4.6], [7.7, 8]]))

    def test_get_uptime_intervals_between_within_gap(self):
        intervals = self.livetime.get_uptime_intervals_between(5, 7)
        self.assertEqual(intervals.shape, (0, 2))

    def test_get_uptime_intervals_between_before_and_after(self):
        self.assertEqual(self.livetime.get_uptime_intervals_between(-5, -1).shape, (0, 2))
        self.assertEqual(self.livetime.get_uptime_intervals_between(11, 20).shape, (0, 2))

    def test_get_uptime_intervals_between_infinite(self):
        intervals = self.livetime.get_uptime_intervals_between(-np.inf, np.inf)
        np.testing.assert_almost_equal(intervals, self.livetime.uptime_mjd_intervals_arr)


class TabulatedTimeFluxProfileTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.t = np.array([10.0, 11.0, 13.0, 14.0])
        self.values = np.array([0.0, 2.0, 1.0, 0.0])
        self.profile = TabulatedTimeFluxProfile(t=self.t, values=self.values, cfg=self.cfg)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            TabulatedTimeFluxProfile(t=[1.0], values=[1.0], cfg=self.cfg)
        with self.assertRaises(ValueError):
            TabulatedTimeFluxProfile(t=[1.0, 1.0], values=[1.0, 1.0], cfg=self.cfg)
        with self.assertRaises(ValueError):
            TabulatedTimeFluxProfile(t=[1.0, 2.0], values=[1.0, -1.0], cfg=self.cfg)
        with self.assertRaises(ValueError):
            TabulatedTimeFluxProfile(t=[1.0, 2.0], values=[1.0, 1.0, 1.0], cfg=self.cfg)

    def test_t_start_t_stop(self):
        self.assertEqual(self.profile.t_start, 10.0)
        self.assertEqual(self.profile.t_stop, 14.0)
        self.assertEqual(self.profile.duration, 4.0)

    def test__call__(self):
        values = self.profile(t=np.array([9.0, 10.0, 10.5, 11.0, 12.0, 13.5, 14.0, 15.0]))
        np.testing.assert_almost_equal(values, np.array([0.0, 0.0, 1.0, 2.0, 1.5, 0.5, 0.0, 0.0]))

    def test_get_integral(self):
        # The total integral is the sum of the trapezoids: 1 + 3 + 0.5.
        np.testing.assert_almost_equal(self.profile.get_total_integral(), 4.5)

        t1 = np.array([9.0, 10.2, 11.5, 12.7, 13.1])
        t2 = np.array([10.5, 12.9, 15.0, 13.3, 13.2])
        expected = np.array(
            [
                scipy.integrate.quad(lambda t: self.profile(t)[0], a, b, limit=200)[0]
                for a, b in zip(t1, t2, strict=True)
            ]
        )
        np.testing.assert_allclose(self.profile.get_integral(t1, t2), expected, rtol=1e-8)

    def test_cdf(self):
        np.testing.assert_almost_equal(self.profile.cdf(t=np.array([9.0, 11.0, 14.0, 15.0])), [0, 1 / 4.5, 1, 1])

    def test_move_and_set_params(self):
        state = self.profile.state
        self.profile.move(5)
        self.assertNotEqual(self.profile.state, state)
        self.assertEqual(self.profile.t_start, 15.0)
        self.assertEqual(self.profile.t_stop, 19.0)
        np.testing.assert_almost_equal(self.profile(t=np.array([16.0])), [2.0])
        np.testing.assert_almost_equal(self.profile.get_total_integral(), 4.5)

        updated = self.profile.set_params({'t_start': 10.0})
        self.assertTrue(updated)
        self.assertEqual(self.profile.state, state)
        np.testing.assert_almost_equal(self.profile(t=np.array([11.0])), [2.0])

        np.testing.assert_almost_equal(self.profile.times, self.t)


class OntimeIntegralTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.livetime = create_livetime()

    def test_box_profile(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=0.5, stop=8, cfg=self.cfg)
        S = get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=self.livetime)
        np.testing.assert_almost_equal(S, 0.5 + 3.3 + 0.3)

    def test_unity_profile_gives_livetime(self):
        profile = UnityTimeFluxProfile(cfg=self.cfg)
        S = get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=self.livetime)
        np.testing.assert_almost_equal(S, self.livetime.livetime)

    def test_no_overlap(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=5, stop=7, cfg=self.cfg)
        self.assertEqual(get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=self.livetime), 0)

        profile = TabulatedTimeFluxProfile(t=[20, 21, 22], values=[0, 1, 0], cfg=self.cfg)
        self.assertEqual(get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=self.livetime), 0)


class DrawOntimesTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.livetime = create_livetime()

    def _assert_distribution(self, profile, times):
        """Checks that the given times are distributed according to the time
        flux profile restricted to the detector on-time via a KS test.
        """
        self.assertTrue(np.all(self.livetime.is_on(times)))
        self.assertTrue(np.all((times >= profile.t_start) & (times <= profile.t_stop)))

        intervals = self.livetime.get_uptime_intervals_between(profile.t_start, profile.t_stop)
        S = np.sum(profile.get_integral(intervals[:, 0], intervals[:, 1]))

        def cdf(x):
            x = np.atleast_1d(x)
            out = np.zeros_like(x, dtype=np.float64)
            for i, xi in enumerate(x):
                lo = intervals[:, 0]
                hi = np.minimum(intervals[:, 1], xi)
                m = hi > lo
                out[i] = np.sum(profile.get_integral(lo[m], hi[m])) / S if np.any(m) else 0
            return out

        result = scipy.stats.kstest(times, cdf)
        self.assertGreater(result.pvalue, 1e-3)

    def test_tabulated_profile(self):
        profile = TabulatedTimeFluxProfile(t=[0.5, 2.0, 8.5, 9.0], values=[0.0, 3.0, 1.0, 0.0], cfg=self.cfg)
        times = draw_ontimes_from_TimeFluxProfile(
            profile=profile, livetime=self.livetime, rss=RandomStateService(seed=1), size=20000
        )
        self.assertEqual(times.shape, (20000,))
        self._assert_distribution(profile, times)

    def test_gaussian_profile(self):
        profile = GaussianTimeFluxProfile(t0=4.5, sigma_t=1.5, cfg=self.cfg)
        times = draw_ontimes_from_TimeFluxProfile(
            profile=profile, livetime=self.livetime, rss=RandomStateService(seed=2), size=20000
        )
        self._assert_distribution(profile, times)

    def test_box_profile(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=0.5, stop=8, cfg=self.cfg)
        times = draw_ontimes_from_TimeFluxProfile(
            profile=profile, livetime=self.livetime, rss=RandomStateService(seed=3), size=20000
        )
        self._assert_distribution(profile, times)

    def test_zero_size(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=5, stop=7, cfg=self.cfg)
        times = draw_ontimes_from_TimeFluxProfile(
            profile=profile, livetime=self.livetime, rss=RandomStateService(seed=1), size=0
        )
        self.assertEqual(len(times), 0)

    def test_no_overlap_raises(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=5, stop=7, cfg=self.cfg)
        with self.assertRaises(ValueError):
            draw_ontimes_from_TimeFluxProfile(
                profile=profile, livetime=self.livetime, rss=RandomStateService(seed=1), size=10
            )


class SignalTimePDFProfileChangeTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.pmm = ParameterModelMapper(models=[SourceModel()])
        self.livetime = create_livetime()

    def test_no_overlap_gives_zero_pd(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=5, stop=7, cfg=self.cfg)
        pdf = SignalTimePDF(pmm=self.pmm, livetime=self.livetime, time_flux_profile=profile, cfg=self.cfg)
        self.assertEqual(pdf.get_S(), 0)

        tdm = create_tdm(n_sources=1, n_selected_events=3)
        pdf.initialize_for_new_trial(tdm=tdm)
        (pd, _) = pdf.get_pd(tdm=tdm)
        np.testing.assert_equal(pd, np.zeros(3))

    def test_precalculated_pd_follows_profile_changes(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=0, stop=10, cfg=self.cfg)
        pdf = SignalTimePDF(pmm=self.pmm, livetime=self.livetime, time_flux_profile=profile, cfg=self.cfg)

        # The event times of the mock tdm are 0, 5, and 9.7.
        tdm = create_tdm(n_sources=1, n_selected_events=3)
        pdf.initialize_for_new_trial(tdm=tdm)
        S = self.livetime.livetime
        np.testing.assert_almost_equal(pdf.get_pd(tdm=tdm)[0], [1 / S, 0, 1 / S])

        # Change the parameters of the profile instance externally.
        profile.set_params({'t0': 8.5, 'tw': 3})
        S = 10 - 7.7
        np.testing.assert_almost_equal(pdf.get_pd(tdm=tdm)[0], [0, 0, 1 / S])

        # Replace the profile instance.
        pdf.time_flux_profile = BoxTimeFluxProfile.from_start_and_stop_time(start=0, stop=1, cfg=self.cfg)
        np.testing.assert_almost_equal(pdf.get_pd(tdm=tdm)[0], [1, 0, 0])


class TimeDependentDetSigYieldTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.livetime = create_livetime()

        # Create a mock time-integrated detector signal yield, which returns 2
        # for each source and a gradient of 3 for the global parameter 1.
        self.base = Mock(spec=DetSigYield)
        self.base.__class__ = DetSigYield  # pyright: ignore[reportAttributeAccessIssue]
        self.base.param_names = ('gamma',)
        self.base.dataset = Mock(spec=Dataset)
        self.base.dataset.__class__ = Dataset
        self.base.fluxmodel = Mock(spec=FluxModel)
        self.base.fluxmodel.__class__ = FluxModel
        self.base.return_value = (np.array([2.0]), {1: np.array([3.0])})

    def _create_detsigyield(self, profile):
        return TimeDependentDetSigYield(detsigyield=self.base, livetime=self.livetime, time_flux_profile=profile)

    def test_unity_profile_gives_time_integrated_yield(self):
        detsigyield = self._create_detsigyield(UnityTimeFluxProfile(cfg=self.cfg))
        np.testing.assert_almost_equal(detsigyield.get_time_factor(), 1)
        (values, grads) = detsigyield(src_recarray=None, src_params_recarray=None)
        np.testing.assert_almost_equal(values, [2.0])
        np.testing.assert_almost_equal(grads[1], [3.0])

    def test_time_factor(self):
        profile = BoxTimeFluxProfile.from_start_and_stop_time(start=0.5, stop=8, cfg=self.cfg)
        detsigyield = self._create_detsigyield(profile)
        f = (0.5 + 3.3 + 0.3) / self.livetime.livetime
        np.testing.assert_almost_equal(detsigyield.get_time_factor(), f)
        (values, grads) = detsigyield(src_recarray=None, src_params_recarray=None)
        np.testing.assert_almost_equal(values, [2.0 * f])
        np.testing.assert_almost_equal(grads[1], [3.0 * f])

        # Changing the profile parameters updates the time factor.
        profile.set_params({'t0': 8.85, 'tw': 2.3})
        np.testing.assert_almost_equal(detsigyield.get_time_factor(), 2.3 / self.livetime.livetime)

        # Replacing the profile updates the time factor.
        detsigyield.time_flux_profile = BoxTimeFluxProfile.from_start_and_stop_time(start=0, stop=1, cfg=self.cfg)
        np.testing.assert_almost_equal(detsigyield.get_time_factor(), 1 / self.livetime.livetime)

    def test_no_overlap_gives_zero_yield(self):
        profile = TabulatedTimeFluxProfile(t=[5, 6, 7], values=[0, 1, 0], cfg=self.cfg)
        detsigyield = self._create_detsigyield(profile)
        (values, grads) = detsigyield(src_recarray=None, src_params_recarray=None)
        np.testing.assert_equal(values, [0.0])
        np.testing.assert_equal(grads[1], [0.0])


class TimeWindowEventSelectionMethodTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.shg_mgr = Mock(spec_set=['__class__', 'source_list', 'n_sources'])
        self.shg_mgr.__class__ = SourceHypoGroupManager  # pyright: ignore[reportAttributeAccessIssue]
        self.shg_mgr.source_list = [SourceModel(), SourceModel()]
        self.shg_mgr.n_sources = 2

        self.events = DataFieldRecordArray(
            np.array(
                [(0.5,), (2.0,), (3.0,), (5.0,), (9.0,)],
                dtype=[('time', np.float64)],
            )
        )
        self.profile = BoxTimeFluxProfile.from_start_and_stop_time(start=1.5, stop=5.0, cfg=self.cfg)

    def test_select_events(self):
        method = TimeWindowEventSelectionMethod(shg_mgr=self.shg_mgr, time_flux_profile=self.profile)
        (events, (src_idxs, evt_idxs), org_idxs) = method.select_events(self.events, ret_original_evt_idxs=True)

        np.testing.assert_equal(events['time'], [2.0, 3.0, 5.0])
        np.testing.assert_equal(org_idxs, [1, 2, 3])
        np.testing.assert_equal(src_idxs, [0, 0, 0, 1, 1, 1])
        np.testing.assert_equal(evt_idxs, [0, 1, 2, 0, 1, 2])

    def test_select_events_with_src_evt_idxs(self):
        method = TimeWindowEventSelectionMethod(shg_mgr=self.shg_mgr, time_flux_profile=self.profile)
        src_evt_idxs = (np.array([0, 0, 1, 1]), np.array([0, 2, 3, 4]))
        (events, (src_idxs, evt_idxs)) = method.select_events(self.events, src_evt_idxs=src_evt_idxs)

        np.testing.assert_equal(events['time'], [2.0, 3.0, 5.0])
        # The event with index 2 (t=3) becomes 1, the one with index 3 (t=5)
        # becomes 2. The events with index 0 and 4 are not selected.
        np.testing.assert_equal(src_idxs, [0, 1])
        np.testing.assert_equal(evt_idxs, [1, 2])

    def test_intersection(self):
        method = AllEventSelectionMethod(shg_mgr=self.shg_mgr) & TimeWindowEventSelectionMethod(
            shg_mgr=self.shg_mgr, time_flux_profile=self.profile
        )
        (_, (src_idxs, evt_idxs), org_idxs) = method.select_events(self.events, ret_original_evt_idxs=True)
        np.testing.assert_equal(org_idxs, [1, 2, 3])
        np.testing.assert_equal(src_idxs, [0, 0, 0, 1, 1, 1])
        np.testing.assert_equal(evt_idxs, [0, 1, 2, 0, 1, 2])

    def test_follows_profile_changes(self):
        method = TimeWindowEventSelectionMethod(shg_mgr=self.shg_mgr, time_flux_profile=self.profile)
        self.profile.set_params({'t0': 8.5, 'tw': 2})
        (events, _) = method.select_events(self.events)
        np.testing.assert_equal(events['time'], [9.0])


if __name__ == '__main__':
    unittest.main()
