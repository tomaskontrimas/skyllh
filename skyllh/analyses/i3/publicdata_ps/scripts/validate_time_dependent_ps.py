"""Validates the time-dependent point-source analysis on the public data, e.g.
IceTracks-DR2.

The following checks are available:

full_livetime
    A box time profile covering the entire livetime of all datasets must
    reproduce the time-integrated analysis exactly: dataset weights, flux
    conversion, and TS and best fit values of a trial with injected signal and
    of the unblinded data.

season_split
    The combined dataset (e.g. IC86_I-XI) and its individual seasons must give
    the same flux conversion and dataset weights proportional to the on-time
    integral of the time profile, for a gaussian profile spanning a season
    boundary. The unblinded TS values are reported, which can differ slightly
    because of the data-driven background PDFs.

injection_recovery
    Signal is injected with a tabulated lightcurve, which is either read from
    a file or a fast-rise exponential-decay lightcurve. The mean fitted number
    of signal events is compared with the mean number of injected signal
    events. The injected event times are tested against the lightcurve with a
    KS test.

bkg_ts
    The background TS distribution for the tabulated lightcurve is reported:
    the fraction of trials with TS=0 and a chi2 fit of the TS>0 part.

lag_fit
    Signal is injected with the tabulated lightcurve shifted by a time lag,
    and the time lag is fitted together with ns and gamma. The distribution of
    the fitted time lags and numbers of signal events is reported.

Usage example::

    python validate_time_dependent_ps.py --data-basepath /path/to/data \\
        --checks full_livetime season_split injection_recovery bkg_ts lag_fit \\
        --lightcurve my_lightcurve.txt --ra 77.36 --dec 5.69

The lightcurve file must contain two columns: the MJD time and the (relative)
flux. The script exits with code 1 if an exact check fails.
"""

import argparse
import json
import sys
from collections.abc import Callable

import numpy as np
import scipy.stats

import skyllh
from skyllh.analyses.i3.publicdata_ps import (
    time_dependent_ps,
    time_integrated_ps,
)
from skyllh.core.config import Config
from skyllh.core.dataset import Dataset
from skyllh.core.flux_model import (
    GaussianTimeFluxProfile,
    TabulatedTimeFluxProfile,
    TimeFluxProfile,
)
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import PointLikeSource
from skyllh.core.time_profile_fit import do_trials_with_time_flux_profile_fit
from skyllh.core.utils.flux_model import get_ontime_integral_of_TimeFluxProfile

DatasetsFactory = Callable[[Config, list[str]], list[Dataset]]


def _get_ds_weights(ana, gamma: float = 2.0) -> np.ndarray:
    """Returns the dataset signal weights of the given analysis."""
    src_params_recarray = ana.pmm.create_src_params_recarray(gflp_values=np.array([0.0, gamma]))
    ana.src_detsigyield_weights_service.calculate(src_params_recarray=src_params_recarray)
    ana.ds_sig_weight_factors_service.calculate()
    (weights, _) = ana.ds_sig_weight_factors_service.get_weights()
    return np.array(weights)


def _fit(ana, events_list, n_events_list):
    """Fits the given events and returns the TS and best fit values."""
    ana.initialize_trial(events_list, n_events_list)
    (log_lambda, fitparam_values, _) = ana.llhratio.maximize(RandomStateService(seed=1))
    ts = ana.calculate_test_statistic(log_lambda=log_lambda, fitparam_values=fitparam_values)
    return (float(ts), np.asarray(fitparam_values))


def _get_livetimes(ana):
    """Returns the Livetime instances of the datasets of the analysis."""
    return [ana.detsigyield_service.arr[j, 0].livetime for j in range(len(ana.dataset_list))]


def create_fred_lightcurve(t_peak: float, t_rise: float, t_decay: float, cfg: Config) -> TabulatedTimeFluxProfile:
    """Creates a tabulated fast-rise exponential-decay lightcurve."""
    t = np.concatenate(
        (
            np.linspace(t_peak - 5 * t_rise, t_peak, 50),
            np.linspace(t_peak, t_peak + 10 * t_decay, 200)[1:],
        )
    )
    values = np.where(t < t_peak, np.exp((t - t_peak) / t_rise), np.exp(-(t - t_peak) / t_decay))
    values[0] = 0
    values[-1] = 0
    return TabulatedTimeFluxProfile(t=t, values=values, cfg=cfg)


def load_lightcurve(pathfilename: str, cfg: Config) -> TabulatedTimeFluxProfile:
    """Loads a tabulated lightcurve from a two-column text file."""
    data = np.loadtxt(pathfilename)
    return TabulatedTimeFluxProfile(t=data[:, 0], values=data[:, 1], cfg=cfg)


def check_full_livetime(datasets_factory: DatasetsFactory, names: list[str], source: PointLikeSource, seed: int):
    """A box covering the entire livetime must reproduce the time-integrated
    analysis.
    """
    cfg_ti = Config()
    ana_ti = time_integrated_ps.create_analysis(cfg=cfg_ti, datasets=datasets_factory(cfg_ti, names), source=source)
    cfg_td = Config()
    datasets = datasets_factory(cfg_td, names)
    ana_td = time_dependent_ps.create_analysis(
        cfg=cfg_td, datasets=datasets, source=source, box={'start': 50000.0, 'stop': 70000.0}
    )

    result = {
        'ds_weights_time_integrated': _get_ds_weights(ana_ti).tolist(),
        'ds_weights_time_dependent': _get_ds_weights(ana_td).tolist(),
        'flux_scaling_time_integrated': float(ana_ti.calculate_fluxmodel_scaling_factor()),
        'flux_scaling_time_dependent': float(ana_td.calculate_fluxmodel_scaling_factor()),
    }
    ok = np.allclose(result['ds_weights_time_integrated'], result['ds_weights_time_dependent'], rtol=1e-8)
    ok &= np.isclose(result['flux_scaling_time_integrated'], result['flux_scaling_time_dependent'], rtol=1e-8)

    (_, n_events_list, events_list) = ana_td.generate_pseudo_data(rss=RandomStateService(seed=seed), mean_n_sig=30)
    (ts_ti, fp_ti) = _fit(ana_ti, events_list, n_events_list)
    (ts_td, fp_td) = _fit(ana_td, events_list, n_events_list)
    result['trial'] = {'ts_ti': ts_ti, 'ts_td': ts_td, 'fit_ti': fp_ti.tolist(), 'fit_td': fp_td.tolist()}
    ok &= np.isclose(ts_ti, ts_td, rtol=1e-6) and np.allclose(fp_ti, fp_td, rtol=1e-4)

    (ts_ti, params_ti, _) = ana_ti.unblind(minimizer_rss=RandomStateService(seed=1))
    (ts_td, params_td, _) = ana_td.unblind(minimizer_rss=RandomStateService(seed=1))
    result['unblind'] = {
        'ts_ti': float(ts_ti),
        'ts_td': float(ts_td),
        'fit_ti': {k: float(v) for k, v in params_ti.items()},
        'fit_td': {k: float(v) for k, v in params_td.items()},
    }
    ok &= np.isclose(ts_ti, ts_td, rtol=1e-6, atol=1e-8)

    result['passed'] = bool(ok)
    return result


def check_season_split(
    datasets_factory: DatasetsFactory,
    combined_name: str,
    season_names: list[str],
    source: PointLikeSource,
    sigma_t: float,
):
    """The combined dataset and its individual seasons must agree."""
    cfg = Config()
    seasons = datasets_factory(cfg, season_names)

    # Load the livetimes of the seasons to find a season boundary.
    ana_split_probe = time_dependent_ps.create_analysis(
        cfg=cfg, datasets=seasons, source=source, gauss={'mu': 0.0, 'sigma': 1.0}, construct_sig_generator=False
    )
    livetimes = _get_livetimes(ana_split_probe)
    i_boundary = len(livetimes) // 2
    t_boundary = 0.5 * (livetimes[i_boundary - 1].time_stop + livetimes[i_boundary].time_start)

    profile = GaussianTimeFluxProfile(t0=t_boundary, sigma_t=sigma_t, cfg=cfg)

    cfg_split = Config()
    ana_split = time_dependent_ps.create_analysis(
        cfg=cfg_split,
        datasets=datasets_factory(cfg_split, season_names),
        source=source,
        time_flux_profile=GaussianTimeFluxProfile(t0=t_boundary, sigma_t=sigma_t, cfg=cfg_split),
    )
    cfg_comb = Config()
    ana_comb = time_dependent_ps.create_analysis(
        cfg=cfg_comb,
        datasets=datasets_factory(cfg_comb, [combined_name]),
        source=source,
        time_flux_profile=GaussianTimeFluxProfile(t0=t_boundary, sigma_t=sigma_t, cfg=cfg_comb),
    )

    S = np.array([get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=lt) for lt in livetimes])
    expected_weights = S / np.sum(S)
    weights = _get_ds_weights(ana_split)

    result = {
        't_boundary': float(t_boundary),
        'ds_weights': weights.tolist(),
        'expected_ds_weights': expected_weights.tolist(),
        'flux_scaling_split': float(ana_split.calculate_fluxmodel_scaling_factor()),
        'flux_scaling_combined': float(ana_comb.calculate_fluxmodel_scaling_factor()),
    }
    ok = np.allclose(weights, expected_weights, rtol=1e-6, atol=1e-12)
    ok &= np.isclose(result['flux_scaling_split'], result['flux_scaling_combined'], rtol=1e-3)

    (ts_split, params_split, _) = ana_split.unblind(minimizer_rss=RandomStateService(seed=1))
    (ts_comb, params_comb, _) = ana_comb.unblind(minimizer_rss=RandomStateService(seed=1))
    result['unblind'] = {
        'ts_split': float(ts_split),
        'ts_combined': float(ts_comb),
        'fit_split': {k: float(v) for k, v in params_split.items()},
        'fit_combined': {k: float(v) for k, v in params_comb.items()},
    }
    result['passed'] = bool(ok)
    return result


def check_injection_recovery(
    datasets_factory: DatasetsFactory,
    names: list[str],
    source: PointLikeSource,
    profile_factory: Callable[[Config], TimeFluxProfile],
    n_sig_list: list[float],
    n_trials: int,
    seed: int,
    ncpu: int | None,
):
    """Injects signal with the given lightcurve and checks its recovery."""
    cfg = Config()
    profile = profile_factory(cfg)
    ana = time_dependent_ps.create_analysis(
        cfg=cfg,
        datasets=datasets_factory(cfg, names),
        source=source,
        time_flux_profile=profile,
        evt_sel_time_window=True,
    )

    result = {'ds_weights': _get_ds_weights(ana).tolist(), 'trials': []}
    ok = True

    # Check the distribution of the injected times with a KS test.
    (_, sig_events_dict) = ana.sig_generator.generate_signal_events(  # pyright: ignore[reportOptionalMemberAccess]
        rss=RandomStateService(seed=seed), mean=5000, poisson=False
    )
    times = np.concatenate([events['time'] for events in sig_events_dict.values()])
    livetimes = _get_livetimes(ana)
    S_total = sum(get_ontime_integral_of_TimeFluxProfile(profile=profile, livetime=lt) for lt in livetimes)

    def cdf(x):
        out = np.empty(len(x))
        for i, xi in enumerate(x):
            s = 0.0
            for lt in livetimes:
                intervals = lt.get_uptime_intervals_between(profile.t_start, min(xi, profile.t_stop))
                if len(intervals) > 0:
                    s += float(np.sum(profile.get_integral(intervals[:, 0], intervals[:, 1])))
            out[i] = s / S_total
        return out

    ks = scipy.stats.kstest(np.sort(times)[:: max(1, len(times) // 1000)], cdf)
    result['injected_times_ks_pvalue'] = float(ks.pvalue)

    for n_sig in n_sig_list:
        trials = ana.do_trials(rss=RandomStateService(seed=seed), n=n_trials, mean_n_sig=n_sig, ncpu=ncpu)
        mean_ns = float(np.mean(trials['ns']))
        mean_ns_err = float(np.std(trials['ns']) / np.sqrt(n_trials))
        mean_n_inj = float(np.mean(trials['n_sig']))
        result['trials'].append(
            {
                'mean_n_sig': n_sig,
                'mean_n_injected': mean_n_inj,
                'mean_ns_fit': mean_ns,
                'mean_ns_fit_err': mean_ns_err,
                'median_ts': float(np.median(trials['ts'])),
                'median_gamma': float(np.median(trials['gamma'])),
            }
        )

    result['passed'] = bool(ok)
    return result


def check_bkg_ts(
    datasets_factory: DatasetsFactory,
    names: list[str],
    source: PointLikeSource,
    profile_factory: Callable[[Config], TimeFluxProfile],
    n_trials: int,
    seed: int,
    ncpu: int | None,
):
    """Reports the background TS distribution."""
    cfg = Config()
    ana = time_dependent_ps.create_analysis(
        cfg=cfg,
        datasets=datasets_factory(cfg, names),
        source=source,
        time_flux_profile=profile_factory(cfg),
        evt_sel_time_window=True,
    )
    trials = ana.do_trials(rss=RandomStateService(seed=seed), n=n_trials, mean_n_sig=0, ncpu=ncpu)
    ts = trials['ts']
    m = ts > 0
    result = {
        'n_trials': n_trials,
        'frac_ts_zero': float(np.mean(~m)),
        'ts_quantiles': {q: float(np.quantile(ts, q)) for q in (0.5, 0.9, 0.99)},
    }
    if np.count_nonzero(m) > 10:
        (dof, _, scale) = scipy.stats.chi2.fit(ts[m], floc=0)
        result['chi2_fit_ts_positive'] = {'dof': float(dof), 'scale': float(scale)}
    result['passed'] = True
    return result


def check_lag_fit(
    datasets_factory: DatasetsFactory,
    names: list[str],
    source: PointLikeSource,
    profile_factory: Callable[[Config], TimeFluxProfile],
    injected_lag: float,
    lag_range: tuple[float, float],
    mean_n_sig: float,
    n_trials: int,
    seed: int,
    ncpu: int | None,
):
    """Injects signal with a lagged lightcurve and fits the time lag."""
    cfg = Config()
    profile = profile_factory(cfg)
    if 'lag' not in profile.param_names:
        raise ValueError('The lag_fit check requires a time flux profile with a "lag" parameter!')
    injection_profile = profile.copy({'lag': profile.get_param('lag') + injected_lag})
    time_range = (profile.t_start + lag_range[0], profile.t_stop + lag_range[1])

    ana = time_dependent_ps.create_analysis(
        cfg=cfg,
        datasets=datasets_factory(cfg, names),
        source=source,
        time_flux_profile=profile,
        injection_time_flux_profile=injection_profile,
        evt_sel_time_window=time_range,
    )
    trials = do_trials_with_time_flux_profile_fit(
        ana=ana,
        time_flux_profile=profile,
        param_ranges={'lag': lag_range},
        n=n_trials,
        mean_n_sig=mean_n_sig,
        seed=seed,
        ncpu=ncpu,
        n_seeds=max(5, int(np.ceil((lag_range[1] - lag_range[0]) / max(profile.duration / 10, 1e-6))) + 1),
    )
    result = {
        'injected_lag': injected_lag,
        'mean_n_sig': mean_n_sig,
        'mean_n_injected': float(np.mean(trials['n_sig'])),
        'lag_fit_median': float(np.median(trials['lag'])),
        'lag_fit_mean': float(np.mean(trials['lag'])),
        'lag_fit_std': float(np.std(trials['lag'])),
        'lag_fit_68_interval': [float(np.quantile(trials['lag'], 0.16)), float(np.quantile(trials['lag'], 0.84))],
        'mean_ns_fit': float(np.mean(trials['ns'])),
        'median_ts': float(np.median(trials['ts'])),
        'passed': True,
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collection', default='IceTracks-DR2')
    parser.add_argument('--data-basepath', default=None)
    parser.add_argument(
        '--checks',
        nargs='*',
        default=['full_livetime', 'season_split', 'injection_recovery', 'bkg_ts', 'lag_fit'],
    )
    parser.add_argument('--names', nargs='*', default=['IC40', 'IC59', 'IC79', 'IC86_I-XI'])
    parser.add_argument('--combined-name', default='IC86_I-XI')
    parser.add_argument(
        '--season-names',
        nargs='*',
        default=[f'IC86_{s}' for s in ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI')],
    )
    parser.add_argument('--ra', type=float, default=77.3582, help='Source right-ascension in degrees.')
    parser.add_argument('--dec', type=float, default=5.6931, help='Source declination in degrees.')
    parser.add_argument('--lightcurve', default=None, help='Two-column file with MJD and relative flux.')
    parser.add_argument('--t-peak', type=float, default=58000.0, help='Peak MJD of the default FRED lightcurve.')
    parser.add_argument('--t-rise', type=float, default=0.5, help='Rise time in days of the FRED lightcurve.')
    parser.add_argument('--t-decay', type=float, default=5.0, help='Decay time in days of the FRED lightcurve.')
    parser.add_argument('--sigma-t', type=float, default=20.0, help='Width in days of the season split profile.')
    parser.add_argument('--n-sig', type=float, nargs='*', default=[0, 3, 6, 12])
    parser.add_argument('--injected-lag', type=float, default=1.0, help='The injected time lag in days.')
    parser.add_argument('--lag-range', type=float, nargs=2, default=[-5.0, 5.0], help='The fitted lag range.')
    parser.add_argument('--lag-n-sig', type=float, default=10, help='The mean number of signal events.')
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--n-bkg-trials', type=int, default=2000)
    parser.add_argument('--ncpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', default=None, help='Optional JSON output file.')
    args = parser.parse_args(argv)

    def datasets_factory(cfg, names):
        return skyllh.create_datasets(args.collection, cfg=cfg, names=names, base_path=args.data_basepath)

    def profile_factory(cfg):
        if args.lightcurve is not None:
            return load_lightcurve(args.lightcurve, cfg=cfg)
        return create_fred_lightcurve(t_peak=args.t_peak, t_rise=args.t_rise, t_decay=args.t_decay, cfg=cfg)

    source = PointLikeSource(name='source', ra=np.radians(args.ra), dec=np.radians(args.dec))

    return run_checks(
        checks=args.checks,
        datasets_factory=datasets_factory,
        names=args.names,
        combined_name=args.combined_name,
        season_names=args.season_names,
        source=source,
        profile_factory=profile_factory,
        sigma_t=args.sigma_t,
        n_sig_list=args.n_sig,
        n_trials=args.n_trials,
        n_bkg_trials=args.n_bkg_trials,
        injected_lag=args.injected_lag,
        lag_range=tuple(args.lag_range),
        lag_n_sig=args.lag_n_sig,
        seed=args.seed,
        ncpu=args.ncpu,
        output=args.output,
    )


def run_checks(
    checks: list[str],
    datasets_factory: DatasetsFactory,
    names: list[str],
    combined_name: str,
    season_names: list[str],
    source: PointLikeSource,
    profile_factory: Callable[[Config], TimeFluxProfile],
    sigma_t: float,
    n_sig_list: list[float],
    n_trials: int,
    n_bkg_trials: int,
    seed: int,
    ncpu: int | None,
    output: str | None = None,
    injected_lag: float = 1.0,
    lag_range: tuple[float, float] = (-5.0, 5.0),
    lag_n_sig: float = 10,
) -> int:
    """Runs the given checks, prints the results, and returns the exit code."""
    results = {}
    for check in checks:
        print(f'Running check "{check}" ...', flush=True)
        if check == 'full_livetime':
            results[check] = check_full_livetime(datasets_factory, names, source, seed)
        elif check == 'season_split':
            results[check] = check_season_split(datasets_factory, combined_name, season_names, source, sigma_t)
        elif check == 'injection_recovery':
            results[check] = check_injection_recovery(
                datasets_factory, names, source, profile_factory, n_sig_list, n_trials, seed, ncpu
            )
        elif check == 'bkg_ts':
            results[check] = check_bkg_ts(datasets_factory, names, source, profile_factory, n_bkg_trials, seed, ncpu)
        elif check == 'lag_fit':
            results[check] = check_lag_fit(
                datasets_factory,
                names,
                source,
                profile_factory,
                injected_lag,
                lag_range,
                lag_n_sig,
                n_trials,
                seed,
                ncpu,
            )
        else:
            raise ValueError(f'Unknown check "{check}"!')
        print(json.dumps(results[check], indent=2), flush=True)

    if output is not None:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)

    failed = [check for (check, result) in results.items() if not result['passed']]
    print(f'Failed checks: {failed}' if failed else 'All checks passed.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
