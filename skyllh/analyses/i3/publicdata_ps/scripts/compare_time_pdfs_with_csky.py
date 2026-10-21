"""Cross-checks the time-dependent likelihood ingredients of SkyLLH against the
csky implementation (https://github.com/icecube/csky).

For gaussian and box shaped time profiles and the good-run-lists of one or
several datasets, it compares

- the fraction of the time profile covered by the detector on-time of each
  dataset (csky: ``utils.get_frac_during_livetime``),
- the signal over background time PDF ratio of each event
  (csky: ``pdf.UntriggeredTimePDFRatioEvaluator``),
- the relative signal acceptance of the datasets for a common signal rate
  (csky: ``pdf.SpaceTimePDFRatioModel.AccModel``).

The good-run-lists are taken from the given datasets of a dataset collection,
e.g. IceTracks-DR2, or synthetic good-run-lists are used with ``--synthetic``.

Note: csky truncates gaussian time profiles at ``--cut-n-sigma`` sigma, whereas
SkyLLH truncates them where the profile drops below 1e-12. Hence, small
differences of the order of the gaussian tail beyond ``--cut-n-sigma`` are
expected for gaussian profiles.

Usage example::

    python compare_time_pdfs_with_csky.py --synthetic
    python compare_time_pdfs_with_csky.py --collection IceTracks-DR2 \\
        --names IC86_I IC86_II IC86_III --data-basepath /path/to/data
"""

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import skyllh
from skyllh.analyses.i3.publicdata_ps.utils import clip_grl_start_times
from skyllh.core.backgroundpdf import BackgroundTimePDF
from skyllh.core.config import Config
from skyllh.core.dataset import Dataset
from skyllh.core.detsigyield import DetSigYield, TimeDependentDetSigYield
from skyllh.core.flux_model import (
    BoxTimeFluxProfile,
    FluxModel,
    GaussianTimeFluxProfile,
    TimeFluxProfile,
)
from skyllh.core.livetime import Livetime
from skyllh.core.random import RandomStateService
from skyllh.core.signalpdf import SignalTimePDF
from skyllh.core.utils.flux_model import get_ontime_integral_of_TimeFluxProfile
from skyllh.i3.livetime import I3Livetime


class _TrialData:
    """Minimal trial data manager for a single source providing the event
    times.
    """

    def __init__(self, times):
        self._times = times
        self.n_sources = 1
        self.n_selected_events = len(times)
        self.src_evt_idxs = (np.zeros(len(times), dtype=np.int64), np.arange(len(times)))
        self.trial_data_state_id = 1

    def get_data(self, name):
        if name != 'time':
            raise KeyError(name)
        return self._times

    def get_n_values(self):
        return len(self._times)


def _create_constant_rate_detsigyield(livetime: Livetime, rate: float) -> DetSigYield:
    """Creates a time-integrated detector signal yield of a constant signal
    rate, i.e. a yield proportional to the integrated livetime of the dataset.
    """
    detsigyield = Mock(spec=DetSigYield)
    detsigyield.__class__ = DetSigYield  # pyright: ignore[reportAttributeAccessIssue]
    detsigyield.param_names = ()
    detsigyield.dataset = Mock(spec=Dataset)
    detsigyield.dataset.__class__ = Dataset
    detsigyield.fluxmodel = Mock(spec=FluxModel)
    detsigyield.fluxmodel.__class__ = FluxModel
    detsigyield.return_value = (np.array([rate * livetime.livetime]), {})
    return detsigyield


def create_synthetic_livetimes() -> dict[str, Livetime]:
    """Creates three synthetic livetimes: two consecutive ones and a third one
    overlapping in time with both.
    """
    rng = np.random.default_rng(1)

    def create(t_start, t_stop):
        starts = np.arange(t_start, t_stop, 1 / 3 * 1.1)
        stops = starts + 1 / 3 * rng.uniform(0.8, 1.0, len(starts))
        return Livetime(np.vstack((starts, np.minimum(stops, t_stop))).T)

    return {
        'A': create(59000.0, 59100.0),
        'B': create(59100.0, 59200.0),
        'C': create(59050.0, 59150.0),
    }


def load_livetimes(collection: str, names: list[str], base_path: str | None) -> dict[str, Livetime]:
    """Loads the livetimes of the given datasets from their good-run-lists."""
    cfg = Config()
    datasets = skyllh.create_datasets(collection, cfg=cfg, names=names, base_path=base_path)
    livetimes = {}
    for ds in datasets:
        grl = ds.load_grl()
        clip_grl_start_times(grl_data=grl)
        livetimes[ds.name] = I3Livetime.from_grl_data(grl_data=grl)
    return livetimes


def skyllh_time_ratio(livetime: Livetime, profile: TimeFluxProfile, times: np.ndarray, cfg: Config) -> np.ndarray:
    """Calculates the signal over background time PDF ratio of the given event
    times with the SkyLLH time PDF classes.
    """
    tdm = _TrialData(times)
    sig_pdf = SignalTimePDF(livetime=livetime, time_flux_profile=profile, cfg=cfg)
    bkg_pdf = BackgroundTimePDF(
        livetime=livetime,
        time_flux_profile=BoxTimeFluxProfile.from_start_and_stop_time(
            start=livetime.time_start, stop=livetime.time_stop, cfg=cfg
        ),
        cfg=cfg,
    )
    sig_pdf.initialize_for_new_trial(tdm=tdm)  # pyright: ignore[reportArgumentType]
    bkg_pdf.initialize_for_new_trial(tdm=tdm)  # pyright: ignore[reportArgumentType]
    (sig_pd, _) = sig_pdf.get_pd(tdm=tdm)  # pyright: ignore[reportArgumentType]
    (bkg_pd, _) = bkg_pdf.get_pd(tdm=tdm)  # pyright: ignore[reportArgumentType]
    return sig_pd / bkg_pd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic good-run-lists.')
    parser.add_argument('--collection', default='IceTracks-DR2', help='The dataset collection name.')
    parser.add_argument('--names', nargs='*', default=None, help='The dataset names.')
    parser.add_argument('--data-basepath', default=None, help='The base path of the data.')
    parser.add_argument('--cut-n-sigma', type=float, default=5.0, help='The csky gaussian truncation in sigma.')
    parser.add_argument('--n-events', type=int, default=20000, help='The number of event times per dataset.')
    parser.add_argument('--rtol', type=float, default=1e-5, help='The relative tolerance of the comparison.')
    args = parser.parse_args(argv)

    try:
        from csky import pdf as cy_pdf  # pyright: ignore[reportMissingImports]
        from csky import utils as cy_utils  # pyright: ignore[reportMissingImports]
    except ImportError:
        print('csky is not installed. Install it from https://github.com/icecube/csky.')
        return 2

    if args.synthetic:
        livetimes = create_synthetic_livetimes()
    else:
        livetimes = load_livetimes(args.collection, args.names, args.data_basepath)

    cfg = Config()
    t_min = min(lt.time_start for lt in livetimes.values())
    t_max = max(lt.time_stop for lt in livetimes.values())
    t_mid = 0.5 * (t_min + t_max)

    # Time profiles to test: (name, csky box flag, t0, dt).
    tests = [
        ('gauss narrow', False, t_mid + 0.123, 0.05),
        ('gauss wide', False, t_mid, 0.1 * (t_max - t_min)),
        ('gauss at start', False, t_min + 0.5, 2.0),
        ('box short', True, t_mid + 0.3, 0.7),
        ('box long', True, t_mid, 0.6 * (t_max - t_min)),
    ]

    rss = RandomStateService(seed=2)
    n_failed = 0

    for name, box, t0, dt in tests:
        if box:
            profile = BoxTimeFluxProfile(t0=t0, tw=dt, cfg=cfg)
        else:
            profile = GaussianTimeFluxProfile(t0=t0, sigma_t=dt, cfg=cfg)
        total = float(profile.get_total_integral())

        print(f'\n=== {name}: {profile} ===')
        print(f'{"dataset":>12} {"frac skyllh":>14} {"frac csky":>14} {"max ratio rel diff":>20}')

        fracs_skyllh = []
        fracs_csky = []
        accs_skyllh = []
        for ds_name, livetime in livetimes.items():
            grl = SimpleNamespace(
                start=livetime.uptime_mjd_intervals_arr[:, 0], stop=livetime.uptime_mjd_intervals_arr[:, 1]
            )

            # Event times on detector on-time, half of them close to t0.
            times = np.concatenate(
                (
                    livetime.draw_ontimes(rss=rss, size=args.n_events // 2),
                    livetime.draw_ontimes(
                        rss=rss,
                        size=args.n_events // 2,
                        t_min=max(livetime.time_start, t0 - 3 * dt),
                        t_max=min(livetime.time_stop, t0 + 3 * dt),
                    )
                    if (t0 + 3 * dt > livetime.time_start) and (t0 - 3 * dt < livetime.time_stop)
                    else np.array([]),
                )
            )
            times = np.sort(times)

            ana = SimpleNamespace(
                grl=grl,
                livetime=livetime.livetime * 86400.0,
                data=SimpleNamespace(mjd=times),
            )

            # The fraction of the time profile during the detector on-time.
            S = get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=livetime)
            frac_skyllh = S / total
            frac_csky = float(
                cy_utils.get_frac_during_livetime(
                    ana, np.array([t0]), np.array([dt]), args.cut_n_sigma, 'center' if box else None
                )[0]
            )
            fracs_skyllh.append(frac_skyllh)
            fracs_csky.append(frac_csky)

            # The time PDF ratio of each event.
            ratio_skyllh = skyllh_time_ratio(livetime=livetime, profile=profile, times=times, cfg=cfg)
            model = cy_pdf.UntriggeredTimePDFRatioModel(
                ana, [None], box=box, box_mode='center', cut_n_sigma=args.cut_n_sigma
            )
            evaluator = model(SimpleNamespace(mjd=times))
            (ratio_csky, _) = evaluator.evaluate_sparse(t0=t0, dt=dt)
            ratio_csky = np.asarray(ratio_csky)

            # Compare only events within the csky truncation of the profile.
            half_width = 0.5 * dt * (1 - 1e-9) if box else args.cut_n_sigma * dt
            m = np.abs(times - t0) < half_width
            denom = np.where(ratio_csky[m] > 0, ratio_csky[m], 1)
            rel = np.max(np.abs(ratio_skyllh[m] - ratio_csky[m]) / denom) if np.any(m) else 0.0
            # Outside the truncation csky gives zero, SkyLLH gives the tail.
            if box:
                outside = ~m & (np.abs(times - t0) > 0.5 * dt * (1 + 1e-9))
                if np.any(ratio_skyllh[outside] != 0):
                    rel = np.inf

            print(f'{ds_name:>12} {frac_skyllh:14.8f} {frac_csky:14.8f} {rel:20.3e}')

            if not np.isclose(frac_skyllh, frac_csky, rtol=args.rtol, atol=1e-7) or rel > args.rtol:
                n_failed += 1
                print('    -> MISMATCH')

            # The relative acceptance for a common signal rate.
            detsigyield = TimeDependentDetSigYield(
                detsigyield=_create_constant_rate_detsigyield(livetime=livetime, rate=1.0),
                livetime=livetime,
                time_flux_profile=profile,
            )
            (Y, _) = detsigyield(src_recarray=None, src_params_recarray=None)
            accs_skyllh.append(float(Y[0]))

        # csky: acc = base_acc * frac / livetime, with base_acc = rate * livetime.
        accs_csky = np.array(fracs_csky)
        accs_skyllh = np.array(accs_skyllh)
        if np.sum(accs_csky) > 0:
            w_csky = accs_csky / np.sum(accs_csky)
            w_skyllh = accs_skyllh / np.sum(accs_skyllh)
            print(f'relative acceptance skyllh: {np.array2string(w_skyllh, precision=8)}')
            print(f'relative acceptance csky:   {np.array2string(w_csky, precision=8)}')
            if not np.allclose(w_skyllh, w_csky, rtol=args.rtol, atol=1e-7):
                n_failed += 1
                print('    -> MISMATCH')

    print(f'\n{n_failed} mismatch(es) found.')
    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
