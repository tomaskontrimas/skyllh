"""This module provides functions to fit the parameters of the time flux profile
of the signal hypothesis of a time-dependent log-likelihood ratio analysis,
e.g. the time lag of a lightcurve, or the mean and width of a gaussian flare.

The time flux profile parameters are fitted by maximizing the profile
log-likelihood ratio: For each tested set of time flux profile parameter values,
the log-likelihood ratio function is maximized w.r.t. the global fit parameters
of the analysis, e.g. ns and gamma, and the test-statistic value is calculated.
The time flux profile parameters are determined by evaluating the
test-statistic on a grid of seed values, followed by a bounded minimization
starting from the best seeds.

The time flux profile must be the instance, which is used by the signal time
PDFs and the time-dependent detector signal yields of the analysis. Any
parameter of the time flux profile can be fitted, e.g. ``lag`` of a
:class:`~skyllh.core.flux_model.TabulatedTimeFluxProfile`, ``t0`` and
``sigma_t`` of a :class:`~skyllh.core.flux_model.GaussianTimeFluxProfile`, or
the prefixed parameters of composite time flux profiles.
"""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.optimize

from skyllh.core.analysis import (
    LLHRatioAnalysis,
)
from skyllh.core.event_selection import (
    IntersectionEventSelectionMethod,
    TimeWindowEventSelectionMethod,
)
from skyllh.core.flux_model import (
    TimeFluxProfile,
)
from skyllh.core.multiproc import (
    get_ncpu,
    parallelize,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.timing import (
    TaskTimer,
    TimeLord,
)


def create_time_flux_profile_param_seeds(
    param_ranges: dict[str, tuple[float, float]],
    n_seeds: int | dict[str, int] = 5,
) -> list[dict[str, float]]:
    """Creates a grid of seed values for the given time flux profile
    parameters.

    Parameters
    ----------
    param_ranges
        The dictionary mapping the parameter names to their (min, max) range.
    n_seeds
        The number of seed values for each parameter, which are distributed
        uniformly within the parameter range, including its boundaries. It can
        be a dictionary specifying the number of seeds for each parameter.

    Returns
    -------
    seeds
        The list of dictionaries holding the parameter values of each seed.
    """
    names = list(param_ranges.keys())
    grids = []
    for name in names:
        n = n_seeds[name] if isinstance(n_seeds, dict) else n_seeds
        (vmin, vmax) = param_ranges[name]
        grids.append(np.linspace(vmin, vmax, n) if n > 1 else np.array([0.5 * (vmin + vmax)]))

    seeds = [dict(zip(names, (float(v) for v in values), strict=True)) for values in itertools.product(*grids)]

    return seeds


def _uses_profile_time_window(evt_sel_method, time_flux_profile: TimeFluxProfile) -> bool:
    """Checks if the given event selection method is or contains a time window
    event selection method based on the given time flux profile.
    """
    if isinstance(evt_sel_method, TimeWindowEventSelectionMethod):
        return evt_sel_method.time_flux_profile is time_flux_profile
    if isinstance(evt_sel_method, IntersectionEventSelectionMethod):
        return _uses_profile_time_window(evt_sel_method.evt_sel_method1, time_flux_profile) or (
            _uses_profile_time_window(evt_sel_method.evt_sel_method2, time_flux_profile)
        )
    return False


def _assert_time_flux_profile_can_be_fitted(
    ana: LLHRatioAnalysis,
    time_flux_profile: TimeFluxProfile,
    param_ranges: dict[str, tuple[float, float]],
):
    """Checks that the given time flux profile parameters can be fitted with
    the given analysis.
    """
    if not isinstance(time_flux_profile, TimeFluxProfile):
        raise TypeError(
            'The time_flux_profile argument must be an instance of TimeFluxProfile! '
            f'Its current type is {classname(time_flux_profile)}.'
        )

    for name, prange in param_ranges.items():
        if name not in time_flux_profile.param_names:
            raise KeyError(
                f'The time flux profile {classname(time_flux_profile)} has no parameter named "{name}"! '
                f'Its parameters are {time_flux_profile.param_names}.'
            )
        if len(prange) != 2 or prange[0] > prange[1]:
            raise ValueError(f'The range {prange} of the parameter "{name}" must be an ascending (min, max) tuple!')

    for evt_sel_method in getattr(ana, '_event_selection_method_list', []):
        if _uses_profile_time_window(evt_sel_method, time_flux_profile):
            raise RuntimeError(
                'The time flux profile parameters cannot be fitted, because the analysis selects the events within '
                'the time range of the time flux profile. Use a fixed time range covering all tested time flux '
                'profiles for the time window event selection instead.'
            )


def fit_time_flux_profile_params(
    ana: LLHRatioAnalysis,
    time_flux_profile: TimeFluxProfile,
    param_ranges: dict[str, tuple[float, float]],
    seeds: Sequence[dict[str, float]] | None = None,
    n_seeds: int | dict[str, int] = 5,
    n_refine: int = 1,
    minimizer_rss: RandomStateService | None = None,
    method: str = 'Nelder-Mead',
    minimize_options: dict | None = None,
    tl: TimeLord | None = None,
) -> dict[str, Any]:
    """Fits the given parameters of the time flux profile of the signal
    hypothesis by maximizing the profile log-likelihood ratio, i.e. for each
    set of time flux profile parameter values the log-likelihood ratio function
    of the analysis is maximized w.r.t. its global fit parameters.

    The analysis must have been initialized with the trial data, e.g. via its
    ``initialize_trial`` or ``unblind`` method.

    Parameters
    ----------
    ana
        The analysis instance.
    time_flux_profile
        The instance of TimeFluxProfile of the signal hypothesis, which is used
        by the signal time PDFs and the time-dependent detector signal yields of
        the analysis. After the fit, its parameters are set to the best fit
        values.
    param_ranges
        The dictionary mapping the names of the time flux profile parameters,
        which should be fitted, to their (min, max) range.
    seeds
        The optional list of dictionaries holding the parameter values of the
        seeds, e.g. from an expectation maximization. If set to ``None``, a grid
        of seed values is created via
        :func:`create_time_flux_profile_param_seeds`.
    n_seeds
        The number of seed values for each parameter, if ``seeds`` is ``None``.
    n_refine
        The number of best seeds, from which a minimization is started. If set
        to 0, the best seed is the result of the fit.
    minimizer_rss
        The instance of RandomStateService for the minimizer of the
        log-likelihood ratio function. If set to ``None``, a new instance with
        seed 1 is created.
    method
        The minimization method of ``scipy.optimize.minimize`` used for the
        refinement. It must support bounds, e.g. ``'Nelder-Mead'`` or
        ``'Powell'``.
    minimize_options
        The optional options passed to ``scipy.optimize.minimize``. For the
        ``'Nelder-Mead'`` method, the default options are an absolute parameter
        tolerance of 0.1% of the smallest parameter range, and an absolute
        test-statistic tolerance of 0.01.
    tl
        The optional instance of TimeLord to measure timing information.

    Returns
    -------
    result
        The dictionary with the following keys:

        ts : float
            The maximal test-statistic value.
        time_params : dict
            The best fit values of the time flux profile parameters.
        global_params : dict
            The best fit values of the global parameters of the analysis.
        fitparam_values : numpy.ndarray
            The best fit values of the global fit parameters.
        n_evaluations : int
            The number of maximizations of the log-likelihood ratio function.
    """
    _assert_time_flux_profile_can_be_fitted(ana, time_flux_profile, param_ranges)

    if minimizer_rss is None:
        minimizer_rss = RandomStateService(seed=1)

    names = list(param_ranges.keys())
    bounds = [param_ranges[name] for name in names]

    if minimize_options is None and method == 'Nelder-Mead':
        widths = [hi - lo for (lo, hi) in bounds if hi > lo]
        minimize_options = {'xatol': 1e-3 * min(widths) if widths else 1e-4, 'fatol': 0.01}

    if seeds is None:
        seeds = create_time_flux_profile_param_seeds(param_ranges=param_ranges, n_seeds=n_seeds)
    if len(seeds) == 0:
        raise ValueError('At least one seed is required!')

    n_evaluations = 0
    cache: dict[tuple, tuple[float, np.ndarray]] = {}

    def evaluate(x) -> tuple[float, np.ndarray]:
        """Maximizes the log-likelihood ratio function for the given time flux
        profile parameter values and returns the test-statistic value and the
        best fit values of the global fit parameters.
        """
        nonlocal n_evaluations

        x = tuple(float(np.clip(v, lo, hi)) for (v, (lo, hi)) in zip(x, bounds, strict=True))
        if x in cache:
            return cache[x]

        time_flux_profile.set_params(dict(zip(names, x, strict=True)))
        (log_lambda, fitparam_values, _) = ana.llhratio.maximize(rss=minimizer_rss)
        ts = float(ana.calculate_test_statistic(log_lambda=log_lambda, fitparam_values=fitparam_values))
        n_evaluations += 1

        cache[x] = (ts, np.array(fitparam_values, copy=True))
        return cache[x]

    with TaskTimer(tl, 'Evaluate time flux profile parameter seeds.'):
        seed_points = [tuple(float(seed[name]) for name in names) for seed in seeds]
        seed_ts = np.array([evaluate(x)[0] for x in seed_points])

    best_x = seed_points[int(np.argmax(seed_ts))]
    best_ts = float(np.max(seed_ts))

    with TaskTimer(tl, 'Refine time flux profile parameters.'):
        for idx in np.argsort(-seed_ts)[: max(0, n_refine)]:
            res = scipy.optimize.minimize(
                lambda x: -evaluate(x)[0],
                x0=np.array(seed_points[idx]),
                method=method,
                bounds=bounds,
                options=minimize_options,
            )
            x = tuple(float(np.clip(v, lo, hi)) for (v, (lo, hi)) in zip(res.x, bounds, strict=True))
            ts = evaluate(x)[0]
            if ts > best_ts:
                (best_ts, best_x) = (ts, x)

    # Set the time flux profile to the best fit and re-evaluate the
    # log-likelihood ratio function, so that the analysis is in the state of
    # the best fit.
    cache.pop(best_x, None)
    (best_ts, best_fitparam_values) = evaluate(best_x)

    result = {
        'ts': best_ts,
        'time_params': dict(zip(names, best_x, strict=True)),
        'global_params': ana.pmm.create_global_params_dict(gflp_values=best_fitparam_values),
        'fitparam_values': best_fitparam_values,
        'n_evaluations': n_evaluations,
    }

    return result


def do_trial_with_time_flux_profile_fit(
    ana: LLHRatioAnalysis,
    rss: RandomStateService,
    time_flux_profile: TimeFluxProfile,
    param_ranges: dict[str, tuple[float, float]],
    mean_n_sig: float = 0,
    seeds: Sequence[dict[str, float]] | None = None,
    n_seeds: int | dict[str, int] = 5,
    n_refine: int = 1,
    method: str = 'Nelder-Mead',
    minimize_options: dict | None = None,
    tl: TimeLord | None = None,
) -> np.ndarray:
    """Generates pseudo data and fits the time flux profile parameters of the
    signal hypothesis together with the global fit parameters. See
    :func:`fit_time_flux_profile_params` for the description of the fit
    arguments.

    Parameters
    ----------
    ana
        The analysis instance.
    rss
        The instance of RandomStateService to generate the pseudo data.
    time_flux_profile
        The instance of TimeFluxProfile of the signal hypothesis.
    param_ranges
        The dictionary mapping the names of the fitted time flux profile
        parameters to their (min, max) range.
    mean_n_sig
        The mean number of signal events to inject.
    seeds
        The optional list of seeds of the time flux profile parameters.
    n_seeds
        The number of seed values for each parameter, if ``seeds`` is ``None``.
    n_refine
        The number of best seeds, from which a minimization is started.
    method
        The minimization method of ``scipy.optimize.minimize``.
    minimize_options
        The optional options passed to ``scipy.optimize.minimize``.
    tl
        The optional instance of TimeLord to measure timing information.

    Returns
    -------
    trial
        The structured numpy ndarray of length 1 with the fields ``seed``,
        ``mean_n_sig``, ``n_sig``, ``ts``, the global parameters of the analysis,
        and the fitted time flux profile parameters.
    """
    (n_sig, n_events_list, events_list) = ana.generate_pseudo_data(rss=rss, mean_n_sig=mean_n_sig, tl=tl)
    ana.initialize_trial(events_list, n_events_list)  # pyright: ignore[reportArgumentType]

    result = fit_time_flux_profile_params(
        ana=ana,
        time_flux_profile=time_flux_profile,
        param_ranges=param_ranges,
        seeds=seeds,
        n_seeds=n_seeds,
        n_refine=n_refine,
        minimizer_rss=RandomStateService(seed=1),
        method=method,
        minimize_options=minimize_options,
        tl=tl,
    )

    names = list(param_ranges.keys())
    dtype = [
        ('seed', np.int64),
        ('mean_n_sig', np.float64),
        ('n_sig', np.int64),
        ('ts', np.float64),
    ]
    dtype += [(name, np.float64) for name in result['global_params']]
    dtype += [(name, np.float64) for name in names]

    trial = np.empty((1,), dtype=dtype)
    trial['seed'] = rss.seed if rss.seed is not None else -1
    trial['mean_n_sig'] = mean_n_sig
    trial['n_sig'] = n_sig
    trial['ts'] = result['ts']
    for name, value in result['global_params'].items():
        trial[name] = value
    for name in names:
        trial[name] = result['time_params'][name]

    return trial


def do_trials_with_time_flux_profile_fit(
    ana: LLHRatioAnalysis,
    time_flux_profile: TimeFluxProfile,
    param_ranges: dict[str, tuple[float, float]],
    n: int = 1000,
    mean_n_sig: float = 0,
    seed: int = 1,
    ncpu: int | None = None,
    seeds: Sequence[dict[str, float]] | None = None,
    n_seeds: int | dict[str, int] = 5,
    n_refine: int = 1,
    method: str = 'Nelder-Mead',
    minimize_options: dict | None = None,
    tl: TimeLord | None = None,
    ppbar: ProgressBar | None = None,
) -> np.ndarray:
    """Performs ``n`` trials, each with a fit of the time flux profile
    parameters of the signal hypothesis together with the global fit
    parameters. See :func:`do_trial_with_time_flux_profile_fit` for the
    description of the arguments.

    Returns
    -------
    trials
        The structured numpy ndarray of length ``n`` with the results of the
        trials.
    """
    rss = RandomStateService(seed=seed)

    args_list = [
        (
            (),
            {
                'ana': ana,
                'time_flux_profile': time_flux_profile,
                'param_ranges': param_ranges,
                'mean_n_sig': mean_n_sig,
                'seeds': seeds,
                'n_seeds': n_seeds,
                'n_refine': n_refine,
                'method': method,
                'minimize_options': minimize_options,
            },
        )
        for _ in range(n)
    ]

    result_list = parallelize(
        func=do_trial_with_time_flux_profile_fit,
        args_list=args_list,
        ncpu=get_ncpu(cfg=ana.cfg, local_ncpu=ncpu),
        rss=rss,
        tl=tl,
        ppbar=ppbar,
    )

    trials = np.concatenate(result_list)

    return trials
