"""Validation tests of the time-dependent public data point-source analysis
using synthetic datasets in the file format of the public data releases.

The tests validate:

- that a box time profile covering the entire livetime reproduces the
  time-integrated analysis exactly,
- that the signal is distributed over the datasets according to the integral
  of the time flux profile over the detector on-time of each dataset,
- that the injected signal follows the injection time profile, which can
  differ from the tested signal hypothesis,
- that datasets without overlap with the time profile are handled,
- that the time window event selection does not change the result,
- that splitting a dataset in time, or into two simultaneous datasets, does not
  change the result beyond statistical fluctuations of the data-driven
  background PDFs,
- that the number of injected signal events with a tabulated lightcurve is
  recovered,
- that the expectation maximization based flare search works with several
  datasets.
"""

import os
import shutil
import tempfile
import unittest
from typing import cast

import numpy as np

from skyllh.analyses.i3.publicdata_ps import (
    time_dependent_ps,
    time_integrated_ps,
)
from skyllh.core.config import Config
from skyllh.core.detsigyield import TimeDependentDetSigYield
from skyllh.core.flux_model import (
    BoxTimeFluxProfile,
    GaussianTimeFluxProfile,
    TabulatedTimeFluxProfile,
)
from skyllh.core.livetime import Livetime
from skyllh.core.random import RandomStateService
from skyllh.core.signal_generator import TimeDependentMultiDatasetSignalGenerator
from skyllh.core.source_model import PointLikeSource
from skyllh.core.utils.flux_model import get_ontime_integral_of_TimeFluxProfile

from .synthetic_data import (
    SyntheticSeason,
    create_dataset_collection,
    create_grl,
    write_synthetic_data,
)

SOURCE = PointLikeSource(name='synthetic source', ra=np.radians(40.0), dec=np.radians(10.0))

# Two consecutive seasons with different effective areas, a season spanning
# both seasons, and two seasons recorded at the same time as season S.
SEASONS = [
    SyntheticSeason('S1', 59000.0, 59100.0, aeff_scale=1.0, seed=1),
    SyntheticSeason('S2', 59100.0, 59200.0, aeff_scale=2.0, seed=2),
    SyntheticSeason('S', 59000.0, 59200.0, n_events=40000, aeff_scale=1.0, seed=3),
]


def _fit(ana, events_list, n_events_list=None):
    """Fits the given events and returns the TS value and the best fit
    parameter values.
    """
    ana.initialize_trial(events_list, n_events_list)
    (log_lambda, fitparam_values, _) = ana.llhratio.maximize(RandomStateService(seed=1))
    ts = ana.calculate_test_statistic(log_lambda=log_lambda, fitparam_values=fitparam_values)
    return (ts, fitparam_values)


def _get_sig_generator(ana) -> TimeDependentMultiDatasetSignalGenerator:
    """Returns the signal generator of the given analysis."""
    return cast(TimeDependentMultiDatasetSignalGenerator, ana.sig_generator)


def _get_ds_weights(ana, gamma=2.0):
    """Returns the dataset signal weights of the log-likelihood ratio function
    of the given analysis for the given spectral index.
    """
    src_params_recarray = ana.pmm.create_src_params_recarray(gflp_values=np.array([0.0, gamma]))
    ana.src_detsigyield_weights_service.calculate(src_params_recarray=src_params_recarray)
    ana.ds_sig_weight_factors_service.calculate()
    (weights, _) = ana.ds_sig_weight_factors_service.get_weights()
    return np.array(weights)


class TimeDependentPSSyntheticTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = tempfile.mkdtemp(prefix='skyllh_synthetic_')

        write_synthetic_data(base_path=cls.base_path, seasons=SEASONS)

        # Split the events of season S into two simultaneous datasets S_a and
        # S_b, each with half of the effective area, and into two consecutive
        # datasets S_early and S_late with the same effective area.
        cls._write_split_seasons()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base_path)

    @classmethod
    def _write_split_seasons(cls):
        def read(path):
            with open(path) as f:
                lines = f.read().splitlines()
            return (lines[0], lines[1:])

        def write(path, header, rows):
            with open(path, 'w') as f:
                f.write('\n'.join([header, *rows]) + '\n')

        (evt_header, evt_rows) = read(os.path.join(cls.base_path, 'events', 'S_exp.csv'))
        (grl_header, grl_rows) = read(os.path.join(cls.base_path, 'uptime', 'S_exp.csv'))
        with open(os.path.join(cls.base_path, 'irfs', 'S_effectiveArea.csv')) as f:
            aeff_lines = f.read().splitlines()

        # Simultaneous split: random half of the events, same GRL, half of the
        # effective area.
        rng = np.random.default_rng(42)
        m = rng.random(len(evt_rows)) < 0.5
        for name, mask in (('S_a', m), ('S_b', ~m)):
            write(
                os.path.join(cls.base_path, 'events', f'{name}_exp.csv'),
                evt_header,
                [r for (r, keep) in zip(evt_rows, mask, strict=True) if keep],
            )
            write(os.path.join(cls.base_path, 'uptime', f'{name}_exp.csv'), grl_header, grl_rows)
            rows = []
            for line in aeff_lines[1:]:
                cols = line.split()
                cols[-1] = f'{0.5 * float(cols[-1]):.6e}'
                rows.append(' '.join(cols))
            write(os.path.join(cls.base_path, 'irfs', f'{name}_effectiveArea.csv'), aeff_lines[0], rows)

        # Sequential split at the start of the first run after t_split.
        t_split = 59100.0
        run_starts = np.array([float(r.split()[1]) for r in grl_rows])
        t_split = run_starts[run_starts >= t_split][0]
        evt_times = np.array([float(r.split()[3]) for r in evt_rows])
        for name, evt_mask, grl_mask in (
            ('S_early', evt_times < t_split, run_starts < t_split),
            ('S_late', evt_times >= t_split, run_starts >= t_split),
        ):
            write(
                os.path.join(cls.base_path, 'events', f'{name}_exp.csv'),
                evt_header,
                [r for (r, keep) in zip(evt_rows, evt_mask, strict=True) if keep],
            )
            write(
                os.path.join(cls.base_path, 'uptime', f'{name}_exp.csv'),
                grl_header,
                [r for (r, keep) in zip(grl_rows, grl_mask, strict=True) if keep],
            )
            write(os.path.join(cls.base_path, 'irfs', f'{name}_effectiveArea.csv'), aeff_lines[0], aeff_lines[1:])

    def _create_analysis(self, module, names, **kwargs):
        cfg = Config()
        seasons = [*SEASONS] + [SyntheticSeason(name, 59000.0, 59200.0) for name in ('S_a', 'S_b', 'S_early', 'S_late')]
        dsc = create_dataset_collection(cfg=cfg, base_path=self.base_path, seasons=seasons)
        return module.create_analysis(cfg=cfg, datasets=dsc[names], source=SOURCE, **kwargs)

    def _create_td_analysis(self, names, **kwargs):
        return self._create_analysis(time_dependent_ps, names, **kwargs)

    def _get_livetimes(self, ana) -> list[Livetime]:
        return [
            cast(Livetime, detsigyield.livetime)
            for detsigyield in ana.detsigyield_service.arr[:, 0]
            if isinstance(detsigyield, TimeDependentDetSigYield)
        ]

    def test_full_livetime_box_equals_time_integrated(self):
        """A box profile covering the entire livetime of all datasets must
        reproduce the time-integrated analysis exactly.
        """
        names = ['S1', 'S2']
        ana_ti = self._create_analysis(time_integrated_ps, names)
        ana_td = self._create_td_analysis(names, box={'start': 58990.0, 'stop': 59210.0})

        np.testing.assert_allclose(_get_ds_weights(ana_td), _get_ds_weights(ana_ti), rtol=1e-10)
        np.testing.assert_allclose(
            ana_td.calculate_fluxmodel_scaling_factor(), ana_ti.calculate_fluxmodel_scaling_factor(), rtol=1e-10
        )

        (_, n_events_list, events_list) = ana_td.generate_pseudo_data(rss=RandomStateService(seed=3), mean_n_sig=25)

        (ts_ti, fp_ti) = _fit(ana_ti, events_list, n_events_list)
        (ts_td, fp_td) = _fit(ana_td, events_list, n_events_list)

        self.assertGreater(ts_ti, 10)
        np.testing.assert_allclose(ts_td, ts_ti, rtol=1e-8)
        np.testing.assert_allclose(fp_td, fp_ti, rtol=1e-6)

    def test_dataset_weights_follow_time_profile(self):
        """The dataset weights must be proportional to the time-integrated
        detector signal yield times the fraction of the livetime covered by the
        time profile.
        """
        names = ['S1', 'S2']
        ana_ti = self._create_analysis(time_integrated_ps, names)
        w_ti = _get_ds_weights(ana_ti)

        profile = TabulatedTimeFluxProfile(t=[59040.0, 59110.0, 59190.0], values=[0.0, 1.0, 0.2], cfg=Config())
        ana_td = self._create_td_analysis(names, time_flux_profile=profile)
        livetimes = self._get_livetimes(ana_td)
        fracs = np.array(
            [get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=lt) / lt.livetime for lt in livetimes]
        )
        expected = w_ti * fracs
        expected /= np.sum(expected)
        np.testing.assert_allclose(_get_ds_weights(ana_td), expected, rtol=1e-10)

        # A profile within the second season only.
        ana_td = self._create_td_analysis(names, box={'start': 59150.0, 'stop': 59160.0})
        np.testing.assert_allclose(_get_ds_weights(ana_td), [0.0, 1.0], atol=1e-15)

    def test_injection_follows_injection_time_profile(self):
        """The signal must be injected according to the injection time profile,
        while the signal hypothesis of the likelihood is not changed.
        """
        names = ['S1', 'S2']
        inj_profile = TabulatedTimeFluxProfile(t=[59050.0, 59052.0, 59060.0], values=[0.0, 1.0, 0.0], cfg=Config())
        ana = self._create_td_analysis(
            names, box={'start': 58990.0, 'stop': 59210.0}, injection_time_flux_profile=inj_profile
        )
        w_before = _get_ds_weights(ana)

        (n_signal, sig_events_dict) = _get_sig_generator(ana).generate_signal_events(
            rss=RandomStateService(seed=4), mean=200, poisson=False
        )
        self.assertEqual(n_signal, 200)
        self.assertEqual(list(sig_events_dict.keys()), [0])
        times = sig_events_dict[0]['time']
        self.assertTrue(np.all((times >= 59050.0) & (times <= 59060.0)))
        livetime = self._get_livetimes(ana)[0]
        self.assertTrue(np.all(livetime.is_on(times)))

        # The flux model scaling factor uses the injection time profile.
        ana_inj = self._create_td_analysis(names, time_flux_profile=inj_profile)
        np.testing.assert_allclose(
            ana.calculate_fluxmodel_scaling_factor(), ana_inj.calculate_fluxmodel_scaling_factor(), rtol=1e-10
        )

        # The detector signal yields of the likelihood are not changed.
        np.testing.assert_allclose(_get_ds_weights(ana), w_before, rtol=1e-12)

    def test_dataset_without_overlap(self):
        """A dataset without overlap with the time profile must not contribute
        any signal and must not break the analysis.
        """
        names = ['S1', 'S2']
        ana = self._create_td_analysis(names, gauss={'mu': 59150.0, 'sigma': 2.0})
        np.testing.assert_allclose(_get_ds_weights(ana), [0.0, 1.0], atol=1e-15)

        (ts, params, _) = ana.unblind(minimizer_rss=RandomStateService(seed=1))
        self.assertTrue(np.isfinite(ts))
        self.assertTrue(np.all(np.isfinite(list(params.values()))))

        trials = ana.do_trials(rss=RandomStateService(seed=5), n=5, mean_n_sig=5, ncpu=1)
        self.assertTrue(np.all(np.isfinite(trials['ts'])))
        self.assertTrue(np.all(trials['ns'] >= 0))

    def test_time_window_event_selection(self):
        """The time window event selection must not change the result."""
        names = ['S1', 'S2']
        profile = TabulatedTimeFluxProfile(t=[59095.0, 59100.0, 59110.0], values=[0.0, 1.0, 0.0], cfg=Config())
        ana = self._create_td_analysis(names, time_flux_profile=profile)
        ana_tw = self._create_td_analysis(names, time_flux_profile=profile, evt_sel_time_window=True)

        for seed in (6, 7):
            (_, n_events_list, events_list) = ana.generate_pseudo_data(rss=RandomStateService(seed=seed), mean_n_sig=8)
            (ts, fp) = _fit(ana, events_list, n_events_list)
            (ts_tw, fp_tw) = _fit(ana_tw, events_list, n_events_list)
            self.assertGreater(ts, 1)
            np.testing.assert_allclose(ts_tw, ts, rtol=1e-6)
            np.testing.assert_allclose(fp_tw, fp, rtol=1e-4)

        # The time profile of an analysis with time window event selection
        # cannot be changed.
        with self.assertRaises(RuntimeError):
            time_dependent_ps.change_time_flux_profile(
                ana_tw, BoxTimeFluxProfile.from_start_and_stop_time(start=59000.0, stop=59001.0, cfg=Config())
            )

    def _get_split_trial(self, ana_full, ana_split, split_masks):
        """Creates signal events with the full analysis and adds them to the
        data. The events are split according to the given masks for the split
        analysis.
        """
        (_, sig_events_dict) = _get_sig_generator(ana_full).generate_signal_events(
            rss=RandomStateService(seed=8), mean=20, poisson=False
        )
        sig = sig_events_dict[0]
        data_full = ana_full.data_list[0].exp
        fields = [name for name in data_full.field_name_list if name in sig.field_name_list]
        full = data_full.copy(keep_fields=fields)
        full.append(sig.copy(keep_fields=fields))

        sig_masks = split_masks(sig['time'], np.random.default_rng(9).random(len(sig)) < 0.5)
        split = []
        for ds_idx, sig_mask in enumerate(sig_masks):
            data = ana_split.data_list[ds_idx].exp.copy(keep_fields=fields)
            data.append(sig[np.flatnonzero(sig_mask)].copy(keep_fields=fields))
            split.append(data)
        return ([full], split)

    def test_simultaneous_datasets(self):
        """Two simultaneous datasets, each holding half of the events and half
        of the effective area, must share the signal equally and give the same
        result as the combined dataset within the statistical fluctuations of
        the data-driven background PDFs.
        """
        profile = TabulatedTimeFluxProfile(t=[59090.0, 59095.0, 59120.0], values=[0.0, 1.0, 0.0], cfg=Config())
        ana_full = self._create_td_analysis(['S'], time_flux_profile=profile)
        ana_split = self._create_td_analysis(['S_a', 'S_b'], time_flux_profile=profile)

        # The effective area of the split datasets is written with 7
        # significant digits.
        np.testing.assert_allclose(_get_ds_weights(ana_split), [0.5, 0.5], rtol=1e-6)
        np.testing.assert_allclose(
            ana_split.calculate_fluxmodel_scaling_factor(), ana_full.calculate_fluxmodel_scaling_factor(), rtol=1e-6
        )

        (full, split) = self._get_split_trial(ana_full, ana_split, lambda t, coin: (coin, ~coin))
        (ts_full, fp_full) = _fit(ana_full, full)
        (ts_split, fp_split) = _fit(ana_split, split)

        self.assertGreater(ts_full, 10)
        np.testing.assert_allclose(ts_split, ts_full, rtol=0.1)
        np.testing.assert_allclose(fp_split, fp_full, rtol=0.1)

    def test_consecutive_datasets(self):
        """Splitting a dataset in time must give the same result, also for a
        time profile spanning the split time, within the statistical
        fluctuations of the data-driven background PDFs.
        """
        profile = GaussianTimeFluxProfile(t0=59100.0, sigma_t=3.0, cfg=Config())
        ana_full = self._create_td_analysis(['S'], time_flux_profile=profile)
        ana_split = self._create_td_analysis(['S_early', 'S_late'], time_flux_profile=profile)

        livetimes = self._get_livetimes(ana_split)
        S = np.array([get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=lt) for lt in livetimes])
        np.testing.assert_allclose(_get_ds_weights(ana_split), S / np.sum(S), rtol=1e-10)
        np.testing.assert_allclose(
            ana_split.calculate_fluxmodel_scaling_factor(), ana_full.calculate_fluxmodel_scaling_factor(), rtol=1e-10
        )

        t_split = livetimes[1].time_start
        (full, split) = self._get_split_trial(ana_full, ana_split, lambda t, coin: (t < t_split, t >= t_split))
        (ts_full, fp_full) = _fit(ana_full, full)
        (ts_split, fp_split) = _fit(ana_split, split)

        self.assertGreater(ts_full, 10)
        np.testing.assert_allclose(ts_split, ts_full, rtol=0.1)
        np.testing.assert_allclose(fp_split, fp_full, rtol=0.1)

    def test_injection_and_recovery(self):
        """The number of injected signal events with a tabulated lightcurve
        spanning two datasets must be recovered, and the injected signal events
        must be distributed over the datasets according to the dataset weights.
        """
        profile = TabulatedTimeFluxProfile(
            t=[59097.0, 59099.0, 59100.5, 59104.0], values=[0.0, 1.0, 0.6, 0.0], cfg=Config()
        )
        ana = self._create_td_analysis(['S1', 'S2'], time_flux_profile=profile, evt_sel_time_window=True)

        weights = _get_ds_weights(ana, gamma=2.0)
        self.assertTrue(np.all(weights > 0.1))

        (n_signal, sig_events_dict) = _get_sig_generator(ana).generate_signal_events(
            rss=RandomStateService(seed=10), mean=2000, poisson=False
        )
        n_ds = np.array([len(sig_events_dict.get(ds_idx, [])) for ds_idx in range(2)])
        self.assertEqual(np.sum(n_ds), n_signal)
        # The number of events per dataset is fixed by rounding the weights.
        np.testing.assert_allclose(n_ds / n_signal, weights, atol=1e-3)

        n_inj = 15
        trials = ana.do_trials(rss=RandomStateService(seed=11), n=100, mean_n_sig=n_inj, ncpu=1)
        mean_ns = np.mean(trials['ns'])
        mean_ns_err = np.std(trials['ns']) / np.sqrt(len(trials))
        self.assertLess(abs(mean_ns - np.mean(trials['n_sig'])), max(4 * mean_ns_err, 0.15 * n_inj))

    def test_em_flare_search_with_several_datasets(self):
        """The expectation maximization based flare search must work with
        several datasets.
        """
        gauss = {'mu': 59100.0, 'sigma': 2.0}
        ana = self._create_td_analysis(['S1', 'S2'], gauss=gauss)

        trials = time_dependent_ps.do_trials_with_em(
            ana=ana, n=2, ncpu=1, mean_n_sig=10, gamma_src=2.0, n_gamma=3, gauss=gauss, ppbar=None
        )
        self.assertEqual(len(trials), 2)
        self.assertTrue(np.all(np.isfinite(trials['ts'])))
        self.assertTrue(np.all(np.abs(trials['mu_fit'] - gauss['mu']) < 20))

        # Changing the signal hypothesis changes the signal time PDFs of all
        # datasets.
        time_dependent_ps.change_signal_time_pdf_of_llhratio_function(ana=ana, box={'start': 59050.0, 'stop': 59150.0})
        sig_time_pdfs = time_dependent_ps.get_signal_time_pdfs(ana)
        self.assertEqual(len(sig_time_pdfs), 2)
        for pdf in sig_time_pdfs:
            self.assertIsInstance(pdf.time_flux_profile, BoxTimeFluxProfile)
        np.testing.assert_allclose(np.sum(_get_ds_weights(ana)), 1.0)


class SyntheticDataTestCase(unittest.TestCase):
    def test_create_grl(self):
        grl = create_grl(59000.0, 59001.0, run_length=0.25, gap_fraction=0.2)
        self.assertTrue(np.all(grl[:, 1] > grl[:, 0]))
        self.assertTrue(np.all(grl[1:, 0] > grl[:-1, 1]))
        self.assertLessEqual(grl[-1, 1], 59001.0)


if __name__ == '__main__':
    unittest.main()
