"""Untriggered flare search for the public data using the expectation
maximization (EM) algorithm.

The EM algorithm determines the mean and width of a gaussian flare from the
signal over background ratios of the events. The flare hypotheses found for a
scan of the spectral index are then tested with the log-likelihood ratio of the
time-dependent analysis created by
:func:`skyllh.analyses.i3.publicdata_ps.time_dependent_ps.create_analysis`,
which is re-exported by this module for convenience.
"""

from typing import cast

import numpy as np

from skyllh.analyses.i3.publicdata_ps.signal_generator import (
    TimeDependentPDDatasetSignalGenerator,
)
from skyllh.analyses.i3.publicdata_ps.time_dependent_ps import (
    change_time_flux_profile,
    create_analysis,
    create_time_flux_profile,
    get_signal_time_pdfs,
)
from skyllh.core.analysis import (
    SingleSourceMultiDatasetLLHRatioAnalysis,
)
from skyllh.core.expectation_maximization import (
    em_fit,
)
from skyllh.core.llhratio import MultiDatasetTCLLHRatio, ZeroSigH0SingleDatasetTCLLHRatio
from skyllh.core.multiproc import (
    get_ncpu,
    parallelize,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)
from skyllh.core.timing import (
    TimeLord,
)

TXS_0506_PLUS056_SOURCE = PointLikeSource(name='TXS 0506+056', ra=np.deg2rad(77.3581851), dec=np.deg2rad(5.69314828))
TXS_0506_PLUS056_ALERT_TIME = 58018.8711856


def change_signal_time_pdf_of_llhratio_function(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    gauss: dict | None = None,
    box: dict | None = None,
):
    """Changes the signal time PDF of the log-likelihood ratio function to a
    gaussian or box shaped time flux profile for all datasets. See
    :func:`change_time_flux_profile` for time flux profiles of any shape.

    Parameters
    ----------
    ana
        The analysis instance.
    gauss
        None or dictionary with {"mu": float, "sigma": float}.
    box
        None or dictionary with {"start": float, "stop": float}.
    """
    time_flux_profile = create_time_flux_profile(cfg=ana.cfg, gauss=gauss, box=box)

    change_time_flux_profile(ana=ana, time_flux_profile=time_flux_profile)


def get_energy_spatial_signal_over_background(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    fitparam_values: np.ndarray,
    tl: TimeLord | None = None,
) -> np.ndarray:
    """Returns the signal over background ratio for
    (spatial_signal * energy_signal) / (spatial_background * energy_background)
    for the events of all datasets. The ratios of the datasets are
    concatenated in the order of the datasets.

    Parameters
    ----------
    ana
        The analysis instance.
    fitparam_values
        The (N_fitparams,)-shaped numpy ndarray holding the values of the global
        fit parameters, e.g. ns and gamma.
    tl
        The optional instance of TimeLord for measuring timing information.

    Returns
    -------
    ratio
        Product of spatial and energy signal over background pdfs.
    """
    src_params_recarray = ana.pmm.create_src_params_recarray(gflp_values=fitparam_values)

    ratios = []
    for tdm, llhratio in zip(ana.tdm_list, cast(MultiDatasetTCLLHRatio, ana.llhratio).llhratio_list, strict=True):
        pdfratio = cast(ZeroSigH0SingleDatasetTCLLHRatio, llhratio).pdfratio

        # pdfratio is an instance of PDFRatioProduct.
        # The first item is the PDF ratio product of the spatial and energy PDF
        # ratios. The second item is the time PDF ratio.
        pdfratio = pdfratio.pdfratio1  # pyright: ignore[reportAttributeAccessIssue]

        ratios.append(pdfratio.get_ratio(tdm=tdm, src_params_recarray=src_params_recarray, tl=tl))

    return np.concatenate(ratios)


def change_fluxmodel_gamma(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    gamma: float,
):
    """Sets the given gamma value to the flux model of the single source.

    Parameters
    ----------
    ana
        The analysis that should be used.
    gamma
        Spectral index for the flux model.
    """
    ana.shg_mgr.shg_list[0].fluxmodel.set_params({'gamma': gamma})
    ana.change_shg_mgr(shg_mgr=ana.shg_mgr)


def change_time_flux_profile_params(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    params: dict,
):
    """Changes the parameters of the time flux profile of the injected signal.

    Parameters
    ----------
    ana
        The analysis that should be used.
    params
        The dictionary with the parameter names and values to be set.
    """
    # Each time flux profile instance must be changed only once, because
    # several signal generators might share the same instance.
    profiles = {}
    for sig_generator in ana.sig_generator_list:
        if isinstance(sig_generator, TimeDependentPDDatasetSignalGenerator):
            profiles[id(sig_generator.time_flux_profile)] = sig_generator.time_flux_profile
    for profile in profiles.values():
        profile.set_params(params)


def calculate_TS(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    em_results: np.ndarray,
    rss: RandomStateService,
) -> tuple[float | None, dict | None, np.ndarray | None]:
    """Calculate the best TS value from the expectation maximization gamma scan
    results.

    Parameters
    ----------
    ana
        The analysis that should be used.
    em_results
        The numpy structured ndarray holding the EM results (from the gamma
        scan).
    rss
        The instance of RandomStateService that should be used to generate
        random numbers from.

    Returns
    -------
    max_TS
        The maximal TS value of all maximized time hypotheses.
    best_em_result
        The row of ``em_results`` that corresponds to the best fit.
    best_fitparam_values
        The instance of numpy ndarray holding the fit parameter values of the
        overall best fit result.
    """
    max_TS = None
    best_em_result = None
    best_fitparam_values = None
    for em_result in em_results:
        change_signal_time_pdf_of_llhratio_function(ana=ana, gauss={'mu': em_result['mu'], 'sigma': em_result['sigma']})

        (log_lambda_max, fitparam_values, _) = ana.llhratio.maximize(rss=rss)

        TS = ana.calculate_test_statistic(log_lambda=log_lambda_max, fitparam_values=fitparam_values)

        if (max_TS is None) or (max_TS < TS):
            max_TS = TS
            best_em_result = em_result
            best_fitparam_values = fitparam_values

    return (max_TS, best_em_result, best_fitparam_values)


def run_gamma_scan_for_single_flare(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    remove_time: float | None = None,
    gamma_min: float = 1,
    gamma_max: float = 5,
    n_gamma: int = 51,
    ppbar: ProgressBar | None = None,
) -> np.ndarray:
    """Runs ``em_fit`` for different gamma values in the signal energy PDF.

    Parameters
    ----------
    ana
        The analysis that should be used.
    remove_time
        Time information of event that should be removed.
    gamma_min
        Lower bound for gamma scan.
    gamma_max
        Upper bound for gamma scan.
    n_gamma
        Number of steps for gamma scan.
    ppbar
        The optional parent instance of ProgressBar.

    Returns
    -------
    em_results
        The numpy structured ndarray with fields

        gamma
            The spectral index value.
        mu
            The determined mean value of the gauss curve.
        sigma
            The determined standard deviation of the gauss curve.
        ns_em
            The scaling factor of the flare.
    """
    em_results_dt = [
        ('gamma', np.float64),
        ('mu', np.float64),
        ('sigma', np.float64),
        ('ns_em', np.float64),
    ]
    em_results = np.empty(n_gamma, dtype=em_results_dt)

    # The EM is performed on the events of all datasets.
    time = np.concatenate([tdm.get_data('time') for tdm in ana.tdm_list])

    gamma_values = np.linspace(gamma_min, gamma_max, n_gamma)

    pbar = ProgressBar(len(gamma_values), parent=ppbar).start()
    for i, gamma in enumerate(gamma_values):
        fitparam_values = np.array([0, gamma], dtype=np.float64)
        ratio = get_energy_spatial_signal_over_background(ana, fitparam_values)
        (mu, sigma, ns) = em_fit(
            time, ratio, n=1, tol=1.0e-200, iter_max=500, weight_thresh=0, initial_width=5000, remove_x=remove_time
        )
        em_results[i] = (
            gamma,
            mu[0],
            sigma[0],
            ns[0],
        )
        pbar.increment()
    pbar.finish()

    return em_results


def unblind_single_flare(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    remove_time: float | None = None,
) -> tuple[float | None, dict | None, np.ndarray | None]:
    """Run EM for a single flare on unblinded data.
    Similar to the original analysis, remove the alert event.

    Parameters
    ----------
    ana
        The analysis that should be used.
    remove_time
        Time of the event that should be removed.
        In the case of the TXS analysis:
        ``remove_time=TXS_0506_PLUS056_ALERT_TIME``.

    Returns
    -------
    max_TS
        The maximal TS value of all maximized time hypotheses.
    best_em_result
        The EM result from the gamma scan corresponding to the best fit.
    best_fitparam_values
        The instance of numpy ndarray holding the fit parameter values of the
        overall best fit result.
    """
    rss = RandomStateService(seed=1)

    ana.unblind(minimizer_rss=rss)

    em_results = run_gamma_scan_for_single_flare(ana=ana, remove_time=remove_time)

    (max_ts, best_em_result, best_fitparam_values) = calculate_TS(ana=ana, em_results=em_results, rss=rss)

    return (max_ts, best_em_result, best_fitparam_values)


def do_trial_with_em(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    rss: RandomStateService,
    mean_n_sig: float = 0,
    gamma_src: float = 2,
    gamma_min: float = 1,
    gamma_max: float = 5,
    n_gamma: int = 21,
    gauss: dict | None = None,
    box: dict | None = None,
    tl: TimeLord | None = None,
    ppbar: ProgressBar | None = None,
) -> np.ndarray:
    """Performs a trial using the expectation maximization algorithm.
    It runs a gamma scan and does the EM for each gamma value.

    Parameters
    ----------
    ana
        The analysis instance that should be used to perform the trial.
    rss
        The instance of RandomStateService that should be used to generate
        random numbers.
    mean_n_sig
        The mean number of signal events that should be generated.
    gamma_src
        The spectral index of the source.
    gamma_min
        Lower bound of the gamma scan.
    gamma_max
        Upper bound of the gamma scan.
    n_gamma
        Number of steps of the gamma scan.
    gauss
        Properties of the Gaussian time PDF.
        None or dictionary with {"mu": float, "sigma": float}.
    box
        Properties of the box time PDF.
        None or dictionary with {"start": float, "stop": float}.
    tl
        The optional instance of TimeLord to measure timing information.
    ppbar
        The optional parent instance of ProgressBar.

    Returns
    -------
    trial
        The numpy structured ndarray of length 1 with the following fields:

        seed
            The seed value used to generate the trial.
        mean_n_sig
            The mean number of signal events of the trial.
        n_sig
            The actual number of signal events in the trial.
        gamma_src
            The spectral index of the source.
        mu_sig
            The mean value of the Gaussian time PDF of the source.
        sigma_sig
            The sigma value of the Gaussian time PDF of the source.
        start_sig
            The start time of the box time PDF of the source.
        stop_sig
            The stop time of the box time PDF of the source.
        ts
            The test-statistic value of the trial.
        ns_fit
            The fitted number of signal events.
        ns_em
            The scaling factor of the flare.
        gamma_fit
            The fitted spectral index of the trial.
        gamma_em
            The spectral index of the best EM trial.
        mu_fit
            The fitted mean value of the Gaussian time PDF.
        sigma_fit
            The fitted sigma value of the Gaussian time PDF.
    """
    trial_dt = [
        ('seed', np.int64),
        ('mean_n_sig', np.float64),
        ('n_sig', np.int64),
        ('gamma_src', np.float64),
        ('mu_sig', np.float64),
        ('sigma_sig', np.float64),
        ('start_sig', np.float64),
        ('stop_sig', np.float64),
        ('ts', np.float64),
        ('ns_fit', np.float64),
        ('ns_em', np.float64),
        ('gamma_fit', np.float64),
        ('gamma_em', np.float64),
        ('mu_fit', np.float64),
        ('sigma_fit', np.float64),
    ]

    trial = np.empty((1,), dtype=trial_dt)

    (n_sig, n_events_list, events_list) = ana.generate_pseudo_data(rss=rss, mean_n_sig=mean_n_sig, tl=tl)
    ana.initialize_trial(
        cast(list[DataFieldRecordArray], events_list),
        cast(list[int | None], n_events_list),
    )

    em_results = run_gamma_scan_for_single_flare(
        ana=ana, gamma_min=gamma_min, gamma_max=gamma_max, n_gamma=n_gamma, ppbar=ppbar
    )

    (max_ts, best_em_result, best_fitparams) = calculate_TS(ana=ana, em_results=em_results, rss=rss)
    assert max_ts is not None
    assert best_em_result is not None
    assert best_fitparams is not None

    trial[0] = (
        rss.seed,
        mean_n_sig,
        n_sig,
        gamma_src,
        gauss['mu'] if gauss is not None else -1,
        gauss['sigma'] if gauss is not None else -1,
        box['start'] if box is not None else -1,
        box['stop'] if box is not None else -1,
        max_ts,
        best_fitparams[0],
        best_em_result['ns_em'],
        best_fitparams[1],
        best_em_result['gamma'],
        best_em_result['mu'],
        best_em_result['sigma'],
    )

    return trial


def do_trials_with_em(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    n: int = 1000,
    ncpu: int | None = None,
    seed: int = 1,
    mean_n_sig: float = 0,
    gamma_src: float = 2,
    gamma_min: float = 1,
    gamma_max: float = 4,
    n_gamma: int = 21,
    gauss: dict | None = None,
    box: dict | None = None,
    tl: TimeLord | None = None,
    ppbar: ProgressBar | None = None,
) -> np.ndarray:
    """Performs ``n_trials`` trials using the expectation maximization
    algorithm. For each trial it runs a gamma scan and does the EM for each
    gamma value.

    Parameters
    ----------
    ana
        The analysis instance that should be used to perform the trials.
    n
        The number of trials to generate.
    ncpu
        The number of CPUs to use to generate the trials. If set to ``None``
        the configured default value will be used.
    mean_n_sig
        The mean number of signal events that should be generated.
    gamma_src
        The spectral index of the source.
    gamma_min
        Lower bound of the gamma scan.
    gamma_max
        Upper bound of the gamma scan.
    n_gamma
        Number of steps of the gamma scan.
    seed
        The seed for the random number generator.
    gauss
        Properties of the Gaussian time PDF.
        None or dictionary with {"mu": float, "sigma": float}.
    box
        Properties of the box time PDF.
        None or dictionary with {"start": float, "stop": float}.
    tl
        The optional instance of TimeLord to measure timing information.
    ppbar
        The optional parent instance of ProgressBar.

    Returns
    -------
    trials
        The numpy structured ndarray of length ``n_trials`` with the results for
        each trial. The array has the following fields:

        seed
            The seed value used to generate the trial.
        mean_n_sig
            The mean number of signal events of the trial.
        n_sig
            The actual number of signal events in the trial.
        gamma_src
            The spectral index of the source.
        mu_sig
            The mean value of the Gaussian time PDF of the source.
        sigma_sig
            The sigma value of the Gaussian time PDF of the source.
        start_sig
            The start time of the box time PDF of the source.
        stop_sig
            The stop time of the box time PDF of the source.
        ts
            The test-statistic value of the trial.
        ns_fit
            The fitted number of signal events.
        ns_em
            The scaling factor of the flare.
        gamma_fit
            The fitted spectral index of the trial.
        gamma_em
            The spectral index of the best EM trial.
        mu_fit
            The fitted mean value of the Gaussian time PDF.
        sigma_fit
            The fitted sigma value of the Gaussian time PDF.
    """
    rss = RandomStateService(seed=seed)

    if mean_n_sig > 0:
        change_fluxmodel_gamma(ana=ana, gamma=gamma_src)

    args_list = [
        (
            (),
            {
                'ana': ana,
                'mean_n_sig': mean_n_sig,
                'gamma_src': gamma_src,
                'gamma_min': gamma_min,
                'gamma_max': gamma_max,
                'n_gamma': n_gamma,
                'gauss': gauss,
                'box': box,
                'ppbar': False,
            },
        )
        for i in range(n)
    ]

    result_list = parallelize(
        func=do_trial_with_em,
        args_list=args_list,
        ncpu=get_ncpu(cfg=ana.cfg, local_ncpu=ncpu),
        rss=rss,
        tl=tl,
        ppbar=ppbar,
    )

    trials = None
    for i, result in enumerate(result_list):
        if trials is None:
            trials = np.empty((n,), dtype=result.dtype)
        trials[i] = result[0]

    assert trials is not None
    return trials


__all__ = (
    'TXS_0506_PLUS056_ALERT_TIME',
    'TXS_0506_PLUS056_SOURCE',
    'calculate_TS',
    'change_fluxmodel_gamma',
    'change_signal_time_pdf_of_llhratio_function',
    'change_time_flux_profile_params',
    'create_analysis',
    'do_trial_with_em',
    'do_trials_with_em',
    'get_energy_spatial_signal_over_background',
    'get_signal_time_pdfs',
    'run_gamma_scan_for_single_flare',
    'unblind_single_flare',
)
