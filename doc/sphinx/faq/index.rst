.. _faq_index:

********************************
Frequently Asked Questions (FAQ)
********************************

.. dropdown:: How can I change the flux model of my sources (for my signal events)?
   :icon: question

    When SkyLLH generates pseudo data for an analysis, it will generate background
    and signal events. It might be desired to change the flux model of the sources 
    to generate signal events following a different flux model as originally chosen
    when the analysis was created.

    The most general procedure to change the flux model(s) of the source(s) of 
    (only) the signal generator is to create a new signal generator with a 
    :py:class:`~skyllh.core.source_hypo_grouping.SourceHypoGroupManager` instance 
    that includes the sources with the changed flux model.

    The new signal generator instance should then be set to the
    :py:attr:`~skyllh.core.analysis.Analysis._sig_generator` attribute of the
    :py:class:`~skyllh.core.analysis.Analysis` class. However, if the analysis 
    signal generator relies on signal generators for each individual dataset, such
    dataset signal generators need to be re-created as well and set to the
    :py:attr:`~skyllh.core.analysis.Analysis._sig_generator_list` attribute of the
    :py:class:`~skyllh.core.analysis.Analysis` class.

    .. note::

        If the flux model(s) of the source(s) should be set for the analysis itself,
        i.e. also for the detector signal yield calculation, which is used to weigh
        the datasets of a multi-dataset analysis, the 
        :py:meth:`~skyllh.core.analysis.Analysis.change_shg_mgr` method should be
        called with the new 
        :py:class:`~skyllh.core.source_hypo_grouping.SourceHypoGroupManager` 
        instance. This method will call the ``change_shg_mgr`` method of all 
        analysis components.

        A re-creation of the signal generator instances is then not required.

.. dropdown:: How can I use my own time profile (e.g. a neutrino lightcurve) in a time-dependent analysis?
   :icon: question

    The signal time PDF, the detector signal yield, and the signal injection of
    a time-dependent analysis work with any
    :py:class:`~skyllh.core.flux_model.TimeFluxProfile` instance. A lightcurve
    given as tabulated values can be used via the
    :py:class:`~skyllh.core.flux_model.TabulatedTimeFluxProfile` class. Any other
    shape can be implemented by deriving a class from
    :py:class:`~skyllh.core.flux_model.TimeFluxProfile` and implementing the
    ``__call__``, ``get_integral``, and ``move`` methods. The times must be given
    in MJD.

    For the public data, the time flux profile is passed to the
    :py:func:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps.create_analysis`
    function via the ``time_flux_profile`` argument. For a fixed time profile,
    the analysis is unblinded via
    :py:meth:`~skyllh.core.analysis.Analysis.unblind`, and trials are generated
    via :py:meth:`~skyllh.core.analysis.Analysis.do_trials`. See the
    :doc:`../tutorials/time_dependent_lightcurve` tutorial.

    In a custom analysis, the time dependence is added by multiplying the PDF
    ratio of each dataset with the time PDF ratio created by
    :py:func:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps.create_time_pdfratio`,
    by wrapping the detector signal yield builder into a
    :py:class:`~skyllh.core.detsigyield.TimeDependentDetSigYieldBuilder`, and by
    using the
    :py:class:`~skyllh.core.signal_generator.TimeDependentMultiDatasetSignalGenerator`
    class as signal generator class of the analysis.

.. dropdown:: How is the signal distributed over several datasets in a time-dependent analysis?
   :icon: question

    The expected number of signal events of a dataset is the time-integrated
    detector signal yield of the dataset scaled by the integral of the time flux
    profile over the detector on-time of the dataset divided by the livetime of
    the dataset, see
    :py:class:`~skyllh.core.detsigyield.TimeDependentDetSigYield`. Hence, the
    fitted number of signal events and the injected signal events are distributed
    over the datasets according to their effective area and their detector
    on-time overlap with the time flux profile. Datasets without overlap do not
    contribute any signal.

    The signal time PDF of each dataset is normalized over the detector on-time
    of the dataset.

    The time flux profile of the injected signal can differ from the time flux
    profile of the tested signal hypothesis. It can be specified via the
    ``injection_time_flux_profile`` argument of
    :py:func:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps.create_analysis`,
    or changed via the
    :py:func:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps.change_injection_time_flux_profile`
    function.

.. dropdown:: Can I combine datasets that overlap in time?
   :icon: question

    Yes, as long as the datasets do not contain the same events. Each dataset
    keeps its own good-run-list, background PDFs, background scrambling, and
    signal time PDF, and the likelihood is the product of the likelihoods of the
    datasets. For example, a track and a cascade dataset recorded at the same time
    can be combined, if events contained in both datasets were removed from one of
    them beforehand, in the data and in the detector response used for the signal.

    SkyLLH does not check for common events. A combined dataset, e.g.
    ``IC86_I-XI`` of the IceTracks-DR2 release, must not be used together with its
    individual seasons, e.g. ``IC86_VII``.

.. dropdown:: How can I fit the time lag of a lightcurve, or the parameters of a gaussian or box profile?
   :icon: question

    Any parameter of the time flux profile of the signal hypothesis can be fitted
    together with the global fit parameters, e.g. ns and gamma, via
    :py:func:`~skyllh.core.time_profile_fit.fit_time_flux_profile_params`. It
    maximizes the profile log-likelihood ratio by evaluating a grid of seeds and
    refining the best seeds with a bounded minimization. Trials are generated via
    :py:func:`~skyllh.core.time_profile_fit.do_trials_with_time_flux_profile_fit`.
    Examples of fitted parameters are

    - the ``lag`` of a :py:class:`~skyllh.core.flux_model.TabulatedTimeFluxProfile`,
    - ``t0`` and ``sigma_t`` of a
      :py:class:`~skyllh.core.flux_model.GaussianTimeFluxProfile`, or ``t0`` and
      ``tw`` of a :py:class:`~skyllh.core.flux_model.BoxTimeFluxProfile`,
    - the parameters of the components of a
      :py:class:`~skyllh.core.flux_model.ProductTimeFluxProfile`, e.g. a time
      window applied to a lightcurve, or of a
      :py:class:`~skyllh.core.flux_model.SumTimeFluxProfile`, e.g. the weight of an
      additional flare. The parameters are named ``<component>_<parameter>``.

    For gaussian flares, the expectation maximization of the
    :py:mod:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps_em` module can be
    used instead, or to provide the seeds of the fit via
    :py:func:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps_em.em_results_to_time_flux_profile_seeds`.
    Seeded with the EM results, the fit finds a test-statistic value at least as
    large as the one of the EM, because the EM determines the flare with a
    simplified likelihood for each tested spectral index. For narrow flares a
    grid of seeds needs a fine spacing in time, so the EM seeds should be
    combined with a grid of seeds, e.g. with logarithmically spaced widths. In
    an untriggered search, the fit of the width prefers very narrow flares
    around single events. The ``ts_penalty`` argument created by
    :py:func:`~skyllh.core.time_profile_fit.create_gaussian_flare_ts_penalty`
    adds the usual term ``2 ln(sqrt(2 pi) sigma_t / T)``, which accounts for the
    number of possible flares of the given width.

    The time flux profile passed to the fit must be the instance used by the
    analysis for the signal hypothesis. The injected signal uses an independent
    copy by default, which is not changed by the fit. If the time window event
    selection is used, it must use a fixed time range covering all tested time
    flux profiles, e.g. ``evt_sel_time_window=(t_min, t_max)``.

.. dropdown:: How can I speed up the trials of a time-dependent analysis?
   :icon: question

    - If the time flux profile is short compared to the livetime, select only
      the events within the time range of the profile via the
      ``evt_sel_time_window`` argument of
      :py:func:`~skyllh.analyses.i3.publicdata_ps.time_dependent_ps.create_analysis`.
      The other events have a zero signal time PDF and are treated
      analytically as pure background events, so the result does not change. If
      the time flux profile is fitted, e.g. its lag, a fixed time range covering
      all tested profiles must be given.
    - Set ``bkg_gen_dec_band_only=True`` to scramble only the events within the
      declination band of the event selection when generating background
      events. The other events can never be selected, so the distribution of the
      trials does not change, but the trials for a given random seed differ.
    - Trials can be run in parallel via the ``ncpu`` argument of the trial
      functions.

    The signal energy PDF values of the trial events are cached for the
    recently used grid points of the spectral index, so repeated maximizations
    of the log-likelihood ratio function for different time flux profiles, e.g.
    in a fit of the time flux profile or in the EM gamma scan, are faster.
