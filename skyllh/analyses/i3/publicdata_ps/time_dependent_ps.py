"""Construction of the time-dependent point-source analysis for the public
data.

The time dependence of the signal is given by a time flux profile of any shape,
e.g. a gaussian, a box, or a tabulated lightcurve model. Several datasets are
supported, where the signal is distributed over the datasets according to their
effective area and their detector on-time overlap with the time flux profile.

The untriggered flare search using the expectation maximization (EM) algorithm
is provided by the
:mod:`skyllh.analyses.i3.publicdata_ps.time_dependent_ps_em` module.
"""

import warnings
from collections.abc import Callable, Iterator
from typing import cast

import numpy as np

import skyllh
from skyllh.analyses.i3.publicdata_ps.backgroundpdf import (
    PDDataBackgroundI3EnergyPDF,
)
from skyllh.analyses.i3.publicdata_ps.detsigyield import (
    PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder,
)
from skyllh.analyses.i3.publicdata_ps.pdfratio import (
    PDSigSetOverBkgPDFRatio,
)
from skyllh.analyses.i3.publicdata_ps.signal_generator import (
    TimeDependentPDDatasetSignalGenerator,
)
from skyllh.analyses.i3.publicdata_ps.signalpdf import (
    PDSignalEnergyPDFSet,
)
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import (
    PDSmearingMatrix,
)
from skyllh.analyses.i3.publicdata_ps.utils import (
    clip_grl_start_times,
    create_energy_cut_spline,
)
from skyllh.core.analysis import (
    SingleSourceMultiDatasetLLHRatioAnalysis,
)
from skyllh.core.analysis import (
    SingleSourceMultiDatasetLLHRatioAnalysis as Analysis,
)
from skyllh.core.background_generator import (
    DatasetBackgroundGenerator,
)
from skyllh.core.backgroundpdf import (
    BackgroundTimePDF,
)
from skyllh.core.config import (
    Config,
)
from skyllh.core.dataset import Dataset
from skyllh.core.detsigyield import (
    TimeDependentDetSigYield,
    TimeDependentDetSigYieldBuilder,
)
from skyllh.core.event_selection import (
    IntersectionEventSelectionMethod,
    SpatialBoxEventSelectionMethod,
    TimeWindowEventSelectionMethod,
)
from skyllh.core.flux_model import (
    BoxTimeFluxProfile,
    GaussianTimeFluxProfile,
    PowerLawEnergyFluxProfile,
    SteadyPointlikeFFM,
    TimeFluxProfile,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.llhratio import MultiDatasetTCLLHRatio, ZeroSigH0SingleDatasetTCLLHRatio
from skyllh.core.logging import (
    get_logger,
)
from skyllh.core.minimizer import (
    LBFGSMinimizerImpl,
    Minimizer,
)
from skyllh.core.minimizers.iminuit import (
    IMinuitMinimizerImpl,
)
from skyllh.core.model import (
    DetectorModel,
)
from skyllh.core.parameters import (
    Parameter,
    ParameterGrid,
    ParameterModelMapper,
)
from skyllh.core.pdf import PDF
from skyllh.core.pdfratio import (
    PDFRatio,
    PDFRatioProduct,
    SigOverBkgPDFRatio,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.scrambling import (
    DataScrambler,
)
from skyllh.core.signal_generator import (
    TimeDependentMultiDatasetSignalGenerator,
)
from skyllh.core.signalpdf import (
    RayleighPSFPointSourceSignalSpatialPDF,
    SignalTimePDF,
)
from skyllh.core.smoothing import (
    BlockSmoothingFilter,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroup,
    SourceHypoGroupManager,
)
from skyllh.core.source_model import (
    PointLikeSource,
)
from skyllh.core.test_statistic import (
    WilksTestStatistic,
)
from skyllh.core.timing import (
    TimeLord,
)
from skyllh.core.trialdata import (
    TrialDataManager,
)
from skyllh.core.utils.analysis import (
    create_trial_data_file,
    pointlikesource_to_data_field_array,
)
from skyllh.core.utils.flux_model import (
    get_ontime_integral_of_TimeFluxProfile,
)
from skyllh.core.utils.tdm import (
    get_tdm_field_func_psi,
)
from skyllh.i3.background_generation import (
    FixedScrambledExpDataI3BkgGenMethod,
)
from skyllh.i3.backgroundpdf import (
    DataBackgroundI3SpatialPDF,
)
from skyllh.i3.config import (
    add_icecube_specific_analysis_required_data_fields,
)
from skyllh.i3.dataset import I3DatasetData
from skyllh.i3.livetime import (
    I3Livetime,
)
from skyllh.i3.scrambling import (
    I3SeasonalVariationTimeScramblingMethod,
)


def create_time_flux_profile(
    cfg: Config,
    gauss: dict | None = None,
    box: dict | None = None,
) -> TimeFluxProfile:
    """Creates a gaussian or box shaped time flux profile.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    gauss
        None or dictionary with {"mu": float, "sigma": float}.
    box
        None or dictionary with {"start": float, "stop": float}.

    Returns
    -------
    time_flux_profile
        The created instance of TimeFluxProfile.
    """
    if (gauss is not None) and (box is not None):
        raise ValueError('Time PDF cannot be both gaussian and box shaped. Please specify only one shape.')

    if gauss is not None:
        return GaussianTimeFluxProfile(t0=gauss['mu'], sigma_t=gauss['sigma'], cfg=cfg)
    if box is not None:
        return BoxTimeFluxProfile.from_start_and_stop_time(start=box['start'], stop=box['stop'], cfg=cfg)

    raise TypeError('Either gauss or box have to be specified as time pdf.')


def create_signal_time_pdf(
    cfg: Config,
    grl: np.ndarray | None = None,
    gauss: dict | None = None,
    box: dict | None = None,
    time_flux_profile: TimeFluxProfile | None = None,
    livetime: Livetime | None = None,
) -> SignalTimePDF:
    """Creates the signal time PDF for a given time flux profile, which can be
    of any shape. For convenience, a gaussian or box shaped time flux profile
    can be specified via the ``gauss`` or ``box`` argument.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    grl
        The structured numpy ndarray holding the good-run-list data. It is used
        to create the Livetime instance if ``livetime`` is ``None``.
    gauss
        None or dictionary with {"mu": float, "sigma": float}.
    box
        None or dictionary with {"start": float, "stop": float}.
    time_flux_profile
        The instance of TimeFluxProfile describing the time dependence of the
        signal. If specified, ``gauss`` and ``box`` must be ``None``.
    livetime
        The instance of Livetime of the dataset. If set to ``None``, it is
        created from ``grl``.

    Returns
    -------
    pdf
        The created time PDF instance.
    """
    if livetime is None:
        if grl is None:
            raise ValueError('Either the grl or the livetime argument must be specified!')
        livetime = I3Livetime.from_grl_data(grl_data=grl)

    if time_flux_profile is None:
        time_flux_profile = create_time_flux_profile(cfg=cfg, gauss=gauss, box=box)
    elif (gauss is not None) or (box is not None):
        raise ValueError('The gauss and box arguments must be None, if a time_flux_profile is specified!')

    pdf = SignalTimePDF(
        cfg=cfg,
        livetime=livetime,
        time_flux_profile=time_flux_profile,
    )

    return pdf


def create_time_pdfratio(
    cfg: Config,
    livetime: Livetime,
    time_flux_profile: TimeFluxProfile,
) -> SigOverBkgPDFRatio:
    """Creates the time PDF ratio for a dataset, i.e. the ratio of the signal
    time PDF for the given time flux profile and the background time PDF,
    which is uniform during the detector on-time of the dataset.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    livetime
        The instance of Livetime of the dataset.
    time_flux_profile
        The instance of TimeFluxProfile describing the time dependence of the
        signal.

    Returns
    -------
    time_pdfratio
        The instance of SigOverBkgPDFRatio for the time PDFs.
    """
    time_bkgpdf = BackgroundTimePDF(
        cfg=cfg,
        livetime=livetime,
        time_flux_profile=BoxTimeFluxProfile.from_start_and_stop_time(
            start=livetime.time_start, stop=livetime.time_stop, cfg=cfg
        ),
    )
    time_sigpdf = create_signal_time_pdf(cfg=cfg, livetime=livetime, time_flux_profile=time_flux_profile)
    time_pdfratio = SigOverBkgPDFRatio(
        cfg=cfg,
        sig_pdf=time_sigpdf,
        bkg_pdf=time_bkgpdf,
        same_axes=False,
    )

    return time_pdfratio


def _iter_pdfratio_pdfs(pdfratio: PDFRatio) -> Iterator[PDF]:
    """Iterates recursively over all PDF instances of the given PDF ratio."""
    if isinstance(pdfratio, PDFRatioProduct):
        yield from _iter_pdfratio_pdfs(pdfratio.pdfratio1)
        yield from _iter_pdfratio_pdfs(pdfratio.pdfratio2)
    elif isinstance(pdfratio, SigOverBkgPDFRatio):
        yield cast(PDF, pdfratio.sig_pdf)
        yield cast(PDF, pdfratio.bkg_pdf)


def get_signal_time_pdfs(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
) -> list[SignalTimePDF]:
    """Returns the signal time PDFs of the log-likelihood ratio functions of
    all datasets of the analysis.

    Parameters
    ----------
    ana
        The analysis instance.

    Returns
    -------
    sig_time_pdfs
        The list of SignalTimePDF instances.
    """
    sig_time_pdfs = []
    for llhratio in cast(MultiDatasetTCLLHRatio, ana.llhratio).llhratio_list:
        pdfratio = cast(ZeroSigH0SingleDatasetTCLLHRatio, llhratio).pdfratio
        sig_time_pdfs.extend(pdf for pdf in _iter_pdfratio_pdfs(pdfratio) if isinstance(pdf, SignalTimePDF))

    return sig_time_pdfs


def change_time_flux_profile(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    time_flux_profile: TimeFluxProfile,
):
    """Changes the time flux profile of the tested signal hypothesis, i.e. the
    time flux profile of the signal time PDFs of all datasets and the time
    flux profile of the time-dependent detector signal yields used by the
    log-likelihood ratio function. The time flux profile of the injected signal
    is not changed. Use :func:`change_injection_time_flux_profile` for that.

    Parameters
    ----------
    ana
        The analysis instance.
    time_flux_profile
        The instance of TimeFluxProfile of the signal hypothesis.

    Raises
    ------
    RuntimeError
        If the analysis uses a time window event selection, because the
        selected events would not match the new time flux profile.
    """
    for evt_sel_method in ana._event_selection_method_list:
        if _uses_time_window_event_selection(evt_sel_method):
            raise RuntimeError(
                'The time flux profile of the signal hypothesis cannot be changed, because the analysis uses a time '
                'window event selection! Create the analysis with evt_sel_time_window=False.'
            )

    for sig_time_pdf in get_signal_time_pdfs(ana):
        sig_time_pdf.time_flux_profile = time_flux_profile

    for shg in ana.shg_mgr.shg_list:
        for builder in shg.detsigyield_builder_list:
            if isinstance(builder, TimeDependentDetSigYieldBuilder):
                builder.time_flux_profile = time_flux_profile

    if ana.detsigyield_service is not None:
        for detsigyield in ana.detsigyield_service.arr.flat:
            if isinstance(detsigyield, TimeDependentDetSigYield):
                detsigyield.time_flux_profile = time_flux_profile


def change_injection_time_flux_profile(
    ana: SingleSourceMultiDatasetLLHRatioAnalysis,
    time_flux_profile: TimeFluxProfile,
):
    """Changes the time flux profile of the injected signal, i.e. the time flux
    profile of the signal generators of all datasets.

    Parameters
    ----------
    ana
        The analysis instance.
    time_flux_profile
        The instance of TimeFluxProfile of the injected signal.
    """
    for sig_generator in ana.sig_generator_list:
        if isinstance(sig_generator, TimeDependentPDDatasetSignalGenerator):
            sig_generator.time_flux_profile = time_flux_profile


def _uses_time_window_event_selection(evt_sel_method) -> bool:
    """Checks if the given event selection method is or contains a time window
    event selection method.
    """
    if isinstance(evt_sel_method, TimeWindowEventSelectionMethod):
        return True
    if isinstance(evt_sel_method, IntersectionEventSelectionMethod):
        return _uses_time_window_event_selection(evt_sel_method.evt_sel_method1) or (
            _uses_time_window_event_selection(evt_sel_method.evt_sel_method2)
        )
    return False


def create_spatial_energy_pdfratio(
    cfg: Config,
    ds: Dataset,
    data: I3DatasetData,
    source: PointLikeSource,
    fluxmodel: SteadyPointlikeFFM,
    param_grid: ParameterGrid,
    sm: PDSmearingMatrix,
    kde_smoothing: bool = False,
    ppbar: ProgressBar | None = None,
) -> PDFRatio:
    """Creates the product of the spatial and energy PDF ratios for a public
    point-source track dataset. This is the default PDF ratio factory of
    :func:`create_analysis`.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    ds
        The instance of Dataset.
    data
        The instance of I3DatasetData holding the data of the dataset.
    source
        The PointLikeSource instance defining the point source position.
    fluxmodel
        The flux model of the source.
    param_grid
        The instance of ParameterGrid for the spectral index.
    sm
        The instance of PDSmearingMatrix of the dataset.
    kde_smoothing
        Apply a KDE-based smoothing to the data-driven background energy pdf.
    ppbar
        The instance of ProgressBar for the optional parent progress bar.

    Returns
    -------
    pdfratio
        The instance of PDFRatio for the product of the spatial and energy PDF
        ratios.
    """
    assert data.exp is not None

    sin_dec_binning = ds.get_binning_definition('sin_dec')
    log_energy_binning = ds.get_binning_definition('log_energy')

    # Create the spatial PDF ratio instance for this dataset.
    spatial_sigpdf = RayleighPSFPointSourceSignalSpatialPDF(cfg=cfg, dec_range=tuple(np.arcsin(sin_dec_binning.range)))
    spatial_bkgpdf = DataBackgroundI3SpatialPDF(cfg=cfg, data_exp=data.exp, sin_dec_binning=sin_dec_binning)
    spatial_pdfratio = SigOverBkgPDFRatio(cfg=cfg, sig_pdf=spatial_sigpdf, bkg_pdf=spatial_bkgpdf)

    # Create the energy PDF ratio instance for this dataset.
    energy_sigpdfset = PDSignalEnergyPDFSet(
        cfg=cfg, ds=ds, src_dec=source.dec, fluxmodel=fluxmodel, param_grid_set=param_grid, ppbar=ppbar, sm=sm
    )
    smoothing_filter = BlockSmoothingFilter(nbins=1)
    energy_bkgpdf = PDDataBackgroundI3EnergyPDF(
        cfg=cfg,
        data_exp=data.exp,
        logE_binning=log_energy_binning,
        sinDec_binning=sin_dec_binning,
        smoothing_filter=smoothing_filter,
        kde_smoothing=kde_smoothing,
    )

    energy_pdfratio = PDSigSetOverBkgPDFRatio(cfg=cfg, sig_pdf_set=energy_sigpdfset, bkg_pdf=energy_bkgpdf)

    return spatial_pdfratio * energy_pdfratio


def create_analysis(
    cfg: Config,
    datasets: list[Dataset],
    source: PointLikeSource,
    box=None,
    gauss=None,
    time_flux_profile: TimeFluxProfile | None = None,
    injection_time_flux_profile: TimeFluxProfile | None = None,
    refplflux_Phi0: float = 1,
    refplflux_E0: float = 1e3,
    refplflux_gamma: float = 2.0,
    ns_seed: float = 10.0,
    ns_min: float = 0.0,
    ns_max: float = 1e3,
    gamma_seed: float | None = 3.0,
    gamma_min: float = 1.0,
    gamma_max: float = 5.0,
    kde_smoothing: bool = False,
    minimizer_impl='LBFGS',
    compress_data: bool = False,
    keep_data_fields: list[str] | None = None,
    evt_sel_delta_angle_deg: float = 10,
    evt_sel_time_window: bool = False,
    pdfratio_factories: dict[str, Callable[..., PDFRatio]] | None = None,
    construct_bkg_generator: bool = True,
    construct_sig_generator: bool = True,
    tl: TimeLord | None = None,
    ppbar: ProgressBar | None = None,
    logger_name: str | None = None,
) -> SingleSourceMultiDatasetLLHRatioAnalysis:
    """Creates the Analysis instance for this particular analysis.

    The time dependence of the signal is given by a time flux profile
    :math:`T(t)`, i.e. the flux of the source is
    :math:`\\Phi(E,t) = \\Phi_0 (E/E_0)^{-\\gamma} T(t)`. The time flux profile
    can be of any shape, e.g. an instance of
    :class:`~skyllh.core.flux_model.TabulatedTimeFluxProfile` for a tabulated
    lightcurve model. For convenience, gaussian and box shaped time flux
    profiles can be specified via the ``gauss`` and ``box`` arguments.

    Several datasets are supported. For each dataset, the signal time PDF is
    normalized over the detector on-time of the dataset, and the expected
    number of signal events of the dataset is given by the integral of the
    time flux profile over the detector on-time of the dataset. Hence, the
    signal is distributed over the datasets according to their effective area
    and their overlap with the time flux profile. Datasets, which do not
    overlap with the time flux profile, do not contribute any signal.

    .. note::

        The datasets must not contain the same events, e.g. a combined dataset
        together with one of its individual seasons. Datasets that were taken
        at the same time, e.g. tracks and cascades, must have their overlap
        removed beforehand, in the data and in the detector response used for
        the signal.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    datasets
        The list of Dataset instances, which should be used in the
        analysis.
    source
        The PointLikeSource instance defining the point source position.
    box
        None if no box shaped time pdf, else dictionary of the format
        ``{'start': float, 'stop': float}``.
    gauss
        None if no gaussian time pdf, else dictionary of the format
        ``{'mu': float, 'sigma': float}``.
    time_flux_profile
        The instance of TimeFluxProfile describing the time dependence of the
        signal hypothesis. The times must be given in MJD. If specified,
        ``box`` and ``gauss`` must be ``None``.
    injection_time_flux_profile
        The optional instance of TimeFluxProfile describing the time dependence
        of the injected signal. If set to ``None``, the time flux profile of
        the signal hypothesis is used.
    refplflux_Phi0
        The flux normalization to use for the reference power law flux model.
    refplflux_E0
        The reference energy to use for the reference power law flux model.
    refplflux_gamma
        The spectral index to use for the reference power law flux model.
    ns_seed
        Value to seed the minimizer with for the ns fit.
    ns_min
        Lower bound for ns fit.
    ns_max
        Upper bound for ns fit.
    gamma_seed
        Value to seed the minimizer with for the gamma fit. If set to None,
        the refplflux_gamma value will be set as gamma_seed.
    gamma_min
        Lower bound for gamma fit.
    gamma_max
        Upper bound for gamma fit.
    kde_smoothing
        Deprecated: use of ``kde_smoothing=True`` is deprecated and will be
        removed in a future version.
        Apply a KDE-based smoothing to the data-driven background pdf.
        Default
    minimizer_impl
        Minimizer implementation to be used. Supported options are "LBFGS"
        (L-BFG-S minimizer used from the :mod:`scipy.optimize` module), or
        "minuit" (Minuit minimizer used by the :mod:`iminuit` module).
        Default
    compress_data
        Flag if the data should get converted from float64 into float32.
    keep_data_fields
        List of additional data field names that should get kept when loading
        the data.
    evt_sel_delta_angle_deg
        The delta angle in degrees for the event selection optimization methods.
    evt_sel_time_window
        Flag if only events within the time range of the time flux profile of
        the signal hypothesis should be selected for the likelihood evaluation.
        This speeds up the analysis significantly for short time flux profiles.
        It can only be used if the time flux profile of the signal hypothesis
        is not changed after the creation of the analysis, i.e. not for the
        expectation maximization based flare search.
    pdfratio_factories
        The optional dictionary mapping dataset names to callables creating
        the product of the spatial and energy PDF ratios for the respective
        dataset. The callables must have the same call signature as
        :func:`create_spatial_energy_pdfratio`, which is used for datasets not
        contained in the dictionary. The time PDF ratio is added automatically.
    construct_bkg_generator
        Flag if the background generator should be constructed (``True``) or not
        (``False``).
    construct_sig_generator
        Flag if the signal generator should be constructed (``True``) or not
        (``False``).
    tl
        The TimeLord instance to use to time the creation of the analysis.
    ppbar
        The instance of ProgressBar for the optional parent progress bar.
    logger_name
        The name of the logger to be used. If set to ``None``, ``__name__`` will
        be used.

    Returns
    -------
    ana
        The Analysis instance for this analysis.
    """
    add_icecube_specific_analysis_required_data_fields(cfg)

    # Remove run number from the dataset data field requirements.
    cfg['datafields'].pop('run', None)

    if logger_name is None:
        logger_name = __name__
    logger = get_logger(logger_name)

    if len(datasets) == 0:
        raise ValueError('At least one dataset must be specified!')

    dataset_names = [ds.name for ds in datasets]
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError(f'The dataset names must be unique! Given dataset names: {dataset_names}')

    if time_flux_profile is None:
        if (gauss is None) and (box is None):
            raise ValueError('No time pdf specified (box, gauss, or time_flux_profile)!')
        time_flux_profile = create_time_flux_profile(cfg=cfg, gauss=gauss, box=box)
    else:
        if (gauss is not None) or (box is not None):
            raise ValueError('The gauss and box arguments must be None, if a time_flux_profile is specified!')
        if not isinstance(time_flux_profile, TimeFluxProfile):
            raise TypeError(
                'The time_flux_profile argument must be an instance of TimeFluxProfile! '
                f'Its current type is {type(time_flux_profile).__name__}.'
            )

    if injection_time_flux_profile is None:
        injection_time_flux_profile = time_flux_profile
    elif not isinstance(injection_time_flux_profile, TimeFluxProfile):
        raise TypeError(
            'The injection_time_flux_profile argument must be an instance of TimeFluxProfile! '
            f'Its current type is {type(injection_time_flux_profile).__name__}.'
        )

    if pdfratio_factories is None:
        pdfratio_factories = {}

    # Create the minimizer instance.
    if minimizer_impl == 'LBFGS':
        minimizer = Minimizer(LBFGSMinimizerImpl(cfg=cfg))
    elif minimizer_impl == 'minuit':
        minimizer = Minimizer(IMinuitMinimizerImpl(cfg=cfg, ftol=1e-8))
    else:
        raise NameError(f'Minimizer implementation `{minimizer_impl}` is not supported Please use `LBFGS` or `minuit`.')

    dtc_dict = None
    dtc_except_fields = None
    if compress_data is True:
        dtc_dict = {np.dtype(np.float64): np.dtype(np.float32)}
        dtc_except_fields = ['mcweight', 'time']

    # Define the flux model.
    fluxmodel = SteadyPointlikeFFM(
        Phi0=refplflux_Phi0,
        energy_profile=PowerLawEnergyFluxProfile(
            E0=refplflux_E0,
            gamma=refplflux_gamma,
            cfg=cfg,
        ),
        cfg=cfg,
    )

    # Define the fit parameter ns.
    param_ns = Parameter(name='ns', initial=ns_seed, valmin=ns_min, valmax=ns_max)

    # Define the fit parameter gamma.
    if gamma_seed is None:
        gamma_seed = refplflux_gamma
    if gamma_max > 4.0:
        logger.warning(
            'You are allowing `gamma` values larger than 4.0. '
            'For such soft spectra, we cannot guarantee the correct '
            'behaviour of the energy PDF.'
        )
    param_gamma = Parameter(name='gamma', initial=gamma_seed, valmin=gamma_min, valmax=gamma_max)

    # Define the detector signal yield builder for the IceCube detector and this
    # source and flux model.
    # The sin(dec) binning will be taken by the builder automatically from the
    # Dataset instance.
    # The time-integrated detector signal yield is scaled by the integral of the
    # time flux profile over the detector on-time of each dataset divided by the
    # live-time of the dataset. The livetimes of the datasets are set below.
    gamma_grid = param_gamma.as_linear_grid(delta=0.1)
    detsigyield_builder = TimeDependentDetSigYieldBuilder(
        builder=PDSingleParamFluxPointLikeSourceI3DetSigYieldBuilder(cfg=cfg, param_grid=gamma_grid),
        time_flux_profile=time_flux_profile,
    )

    # Create a source hypothesis group manager with a single source hypothesis
    # group for the single source.
    shg_mgr = SourceHypoGroupManager(
        SourceHypoGroup(
            sources=source,
            fluxmodel=fluxmodel,
            detsigyield_builders=detsigyield_builder,
        )
    )
    logger.info(str(shg_mgr))

    # Define a detector model for the ns fit parameter.
    detector_model = DetectorModel('IceCube')

    # Define the parameter model mapper for the analysis, which will map global
    # parameters to local source parameters.
    pmm = ParameterModelMapper(models=[detector_model, source])
    pmm.map_param(param_ns, models=detector_model)
    pmm.map_param(param_gamma, models=source)
    logger.info(str(pmm))

    # Define the test statistic.
    test_statistic = WilksTestStatistic()

    # Create the Analysis instance.
    ana = Analysis(
        cfg=cfg,
        shg_mgr=shg_mgr,
        pmm=pmm,
        test_statistic=test_statistic,
        sig_generator_cls=TimeDependentMultiDatasetSignalGenerator,
    )

    # Define the event selection method for pure optimization purposes.
    # We will use the same method for all datasets.
    event_selection_method = SpatialBoxEventSelectionMethod(
        shg_mgr=shg_mgr, delta_angle=np.deg2rad(evt_sel_delta_angle_deg)
    )
    if evt_sel_time_window:
        event_selection_method = event_selection_method & TimeWindowEventSelectionMethod(
            shg_mgr=shg_mgr, time_flux_profile=time_flux_profile
        )

    # Add the data sets to the analysis.
    pbar = ProgressBar(len(datasets), parent=ppbar).start()
    for ds_idx, ds in enumerate(datasets):
        data = cast(
            I3DatasetData,
            ds.load_and_prepare_data(
                keep_fields=keep_data_fields, dtc_dict=dtc_dict, dtc_except_fields=dtc_except_fields, tl=tl
            ),
        )
        assert data.exp is not None

        # Some runs might overlap slightly. So we need to clip those runs.
        clip_grl_start_times(grl_data=cast(np.ndarray, data.grl))

        livetime = I3Livetime.from_grl_data(grl_data=cast(np.ndarray, data.grl))
        detsigyield_builder.set_livetime(ds.name, livetime)

        if get_ontime_integral_of_TimeFluxProfile(profile=time_flux_profile, livetime=livetime) <= 0:
            logger.warning(
                f'The time flux profile of the signal hypothesis does not overlap with the detector on-time of '
                f'dataset "{ds.name}". This dataset will not contribute any signal.'
            )

        sm = PDSmearingMatrix(
            pathfilenames=ds.get_abs_pathfilename_list(ds.get_aux_data_definition('smearing_datafile'))
        )

        # Create the product of the spatial and energy PDF ratios for this
        # dataset.
        pdfratio_factory = pdfratio_factories.get(ds.name, create_spatial_energy_pdfratio)
        pdfratio = pdfratio_factory(
            cfg=cfg,
            ds=ds,
            data=data,
            source=source,
            fluxmodel=fluxmodel,
            param_grid=gamma_grid,
            sm=sm,
            kde_smoothing=kde_smoothing,
            ppbar=ppbar,
        )

        # Create the time PDF ratio instance for this dataset.
        pdfratio = pdfratio * create_time_pdfratio(cfg=cfg, livetime=livetime, time_flux_profile=time_flux_profile)

        # Create a trial data manager and add the required data fields.
        tdm = TrialDataManager()
        tdm.add_source_data_field(name='src_array', func=pointlikesource_to_data_field_array)
        tdm.add_data_field(name='psi', func=get_tdm_field_func_psi(), dt='dec', is_srcevt_data=True)

        data_scrambler = DataScrambler(I3SeasonalVariationTimeScramblingMethod(data=data))
        bkg_gen_method = FixedScrambledExpDataI3BkgGenMethod(cfg=cfg, data_scrambler=data_scrambler)

        bkg_generator = DatasetBackgroundGenerator(
            cfg=cfg,
            dataset=ds,
            data=data,
            bkg_gen_method=bkg_gen_method,
        )

        energy_cut_spline = create_energy_cut_spline(
            ds=ds,
            exp_data=data.exp,
            spl_smooth=ds.get_aux_data('spline_smoothing'),
            cumulative_thr=ds.get_aux_data('cumulative_threshold'),
        )

        sig_generator = TimeDependentPDDatasetSignalGenerator(
            cfg=cfg,
            shg_mgr=shg_mgr,
            ds=ds,
            ds_idx=ds_idx,
            livetime=livetime,
            time_flux_profile=injection_time_flux_profile,
            energy_cut_spline=energy_cut_spline,
            sm=sm,
        )

        ana.add_dataset(
            ds,
            data,
            pdfratio=pdfratio,
            tdm=tdm,
            event_selection_method=event_selection_method,
            bkg_generator=bkg_generator,
            sig_generator=sig_generator,
        )

        pbar.increment()
    pbar.finish()

    ana.construct_services(ppbar=ppbar)

    ana.llhratio = ana.construct_llhratio(minimizer=minimizer, ppbar=ppbar)

    if construct_bkg_generator is True:
        ana.construct_background_generator()

    if construct_sig_generator is True:
        ana.construct_signal_generator()

    return ana


# The names of the expectation maximization based flare search, which were
# provided by this module previously.
_EM_NAMES = (
    'TXS_0506_PLUS056_SOURCE',
    'TXS_0506_PLUS056_ALERT_TIME',
    'change_signal_time_pdf_of_llhratio_function',
    'get_energy_spatial_signal_over_background',
    'change_fluxmodel_gamma',
    'change_time_flux_profile_params',
    'calculate_TS',
    'run_gamma_scan_for_single_flare',
    'unblind_single_flare',
    'do_trial_with_em',
    'do_trials_with_em',
)


def __getattr__(name: str):
    """Provides the names of the expectation maximization based flare search,
    which were moved into the
    :mod:`skyllh.analyses.i3.publicdata_ps.time_dependent_ps_em` module, for
    backward compatibility.
    """
    if name in _EM_NAMES:
        warnings.warn(
            f'Importing "{name}" from {__name__} is deprecated. Import it from '
            'skyllh.analyses.i3.publicdata_ps.time_dependent_ps_em instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from skyllh.analyses.i3.publicdata_ps import time_dependent_ps_em

        return getattr(time_dependent_ps_em, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if __name__ == '__main__':
    from skyllh.core.logging import (
        setup_logging,
    )
    from skyllh.scripting.argparser import (
        create_argparser,
    )

    parser = create_argparser(
        description='Calculates TS for a given source location using the '
        '10-year public point source sample assuming a signal '
        'time PDF.',
    )

    parser.add_argument('--dec', dest='dec', default=5.7, type=float, help='The source declination in degrees.')
    parser.add_argument('--ra', dest='ra', default=77.35, type=float, help='The source right-ascension in degrees.')
    parser.add_argument(
        '--gamma-seed', dest='gamma_seed', default=3, type=float, help='The seed value of the gamma fit parameter.'
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.set_enable_tracing(args.enable_tracing)
    cfg.set_ncpu(args.n_cpu)

    logger = setup_logging(cfg=cfg, name=__name__, log_level='info', log_file=args.debug_logfile)

    datasets = skyllh.create_datasets('IceTracks-DR1', cfg=cfg, base_path=args.data_basepath, names='IC86_II-VII')

    # Define a random state service.
    rss = RandomStateService(args.seed)

    # Define the point source.
    source = PointLikeSource(name='My Point-Like-Source', ra=np.deg2rad(args.ra), dec=np.deg2rad(args.dec))
    logger.info(f'source: {source}')

    tl = TimeLord()

    with tl.task_timer('Creating analysis.'):
        ana = create_analysis(
            cfg=cfg,
            datasets=datasets,
            source=source,
            gamma_seed=args.gamma_seed,
            gauss={'mu': 57000, 'sigma': 62},
            tl=tl,
        )

    with tl.task_timer('Unblinding data.'):
        (TS, param_dict, status) = ana.unblind(minimizer_rss=rss)

    logger.debug(f'TS = {TS:g}')
    logger.debug(f'ns_fit = {param_dict["ns"]:g}')
    logger.debug(f'gamma_fit = {param_dict["gamma"]:g}')
    logger.debug(f'minimizer status = {status}')

    logger.info(f'TimeLord: {tl}')

    tl = TimeLord()
    rss = RandomStateService(seed=1)
    (_, _, _, trials) = create_trial_data_file(
        ana=ana, rss=rss, n_trials=10, mean_n_sig=0, pathfilename=None, ncpu=1, tl=tl
    )
    logger.info(f'trials: {trials}')
    logger.info(f'TimeLord: {tl}')
