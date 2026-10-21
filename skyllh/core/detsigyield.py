import abc
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from skyllh.core.config import (
    Config,
    HasConfig,
)
from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.flux_model import (
    FluxModel,
    TimeFluxProfile,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
    issequence,
    issequenceof,
)
from skyllh.core.source_model import SourceModel
from skyllh.core.types import (
    SourceHypoGroup_t,
)
from skyllh.core.utils.flux_model import (
    get_ontime_integral_of_TimeFluxProfile,
)


class DetSigYield(
    metaclass=abc.ABCMeta,
):
    """This is the abstract base class for a detector signal yield.

    The detector signal yield, Y_s(p_s), is defined as the expected mean
    number of signal events detected by the detector from a given source with
    source parameters p_s.

    To construct a detector signal yield object, four ingredients are
    needed: the dataset holding the monte-carlo data events, a signal flux
    model, the livetime, and a builder instance that knows how to construct
    the actual detector yield in an efficient way.
    In general, the implementation method depends on the detector, the source,
    the flux model with its flux model's signal parameters, and the dataset.
    Hence, for a given detector, source, flux model, and dataset, an appropriate
    implementation method needs to be chosen.
    """

    def __init__(
        self,
        param_names: Sequence[str],
        dataset: Dataset,
        fluxmodel: FluxModel,
        livetime: float | Livetime,
        **kwargs,
    ):
        """Constructs a new detector signal yield object. It takes
        the monte-carlo data events, a flux model of the signal, and the live
        time to compute the detector signal yield.

        Parameters
        ----------
        param_names
            The sequence of parameter names this detector signal yield depends
            on. These are either fixed or floating parameters.
        dataset
            The Dataset instance holding the monte-carlo event data.
        fluxmodel
            The flux model instance. Must be an instance of FluxModel.
        livetime
            The live-time in days to use for the detector signal yield.
        """
        super().__init__(**kwargs)

        self.param_names = param_names
        self.dataset = dataset
        self.fluxmodel = fluxmodel
        self.livetime = livetime

    @property
    def param_names(self):
        """The tuple of parameter names this detector signal yield instance
        is a function of.
        """
        return self._param_names

    @param_names.setter
    def param_names(self, names):
        if not issequence(names):
            names = [names]
        if not issequenceof(names, str):
            raise TypeError(
                f'The param_names property must be a sequence of str instances! Its current type is {classname(names)}.'
            )
        self._param_names = tuple(names)

    @property
    def dataset(self):
        """The Dataset instance, for which this detector signal yield is made
        for.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, ds):
        if not isinstance(ds, Dataset):
            raise TypeError(
                f'The dataset property must be an instance of Dataset! Its current type is {classname(ds)}.'
            )
        self._dataset = ds

    @property
    def fluxmodel(self):
        """The flux model, which should be used to calculate the detector
        signal yield.
        """
        return self._fluxmodel

    @fluxmodel.setter
    def fluxmodel(self, model):
        if not isinstance(model, FluxModel):
            raise TypeError(
                f'The fluxmodel property must be an instance of FluxModel! Its current type is {classname(model)}.'
            )
        self._fluxmodel = model

    @property
    def livetime(self):
        """The live-time in days."""
        return self._livetime

    @livetime.setter
    def livetime(self, lt):
        if not isinstance(lt, (float, Livetime)):
            raise TypeError(
                'The livetime property must be of type float or an instance '
                'of Livetime! '
                f'Its current type is {classname(lt)}.'
            )
        self._livetime = lt

    @abc.abstractmethod
    def sources_to_recarray(
        self,
        sources: SourceModel | Sequence[SourceModel],
    ):
        """This method is supposed to convert a (list of) source model(s) into
        a numpy record array that is understood by the detector signal yield
        class.
        This is for efficiency reasons only. This way the user code can
        pre-convert the list of sources into a numpy record array and cache the
        array.
        The fields of the array are detector signal yield implementation
        dependent, i.e. what kind of sources: point-like source, or extended
        source for instance. Because the sources usually don't change their
        position in the sky, this has to be done only once.

        Parameters
        ----------
        sources
            The source model(s) containing the information of the source(s).

        Returns
        -------
        recarr
            The generated (N_sources,)-shaped 1D numpy record ndarray holding
            the information for each source.
        """

    @abc.abstractmethod
    def __call__(
        self,
        src_recarray,
        src_params_recarray,
    ):
        """Abstract method to retrieve the detector signal yield for the given
        sources and source parameter values.

        Parameters
        ----------
        src_recarray
            The numpy record array containing the information of the sources.
            The required fields of this record array are implementation
            dependent. In the most generic case for a point-like source, it
            must contain the following three fields: ra, dec.
        src_params_recarray
            The numpy record ndarray containing the parameter values of the
            sources. The parameter values can be different for the different
            sources.
            The record array must contain two fields for each source parameter,
            one named <name> with the source's local parameter name
            holding the source's local parameter value, and one named
            <name:gpidx> holding the global parameter index plus one for each
            source value. For values mapping to non-fit parameters, the index
            should be negative.

        Returns
        -------
        detsigyield
            The array with the mean number of signal in the detector for each
            given source.
        grads
            The dictionary holding the gradient values for each global fit
            parameter. The key is the global fit parameter index and the value
            is the (N_sources,)-shaped numpy ndarray holding the gradient value
            dY_k/dp_s.
        """


class DetSigYieldBuilder(
    HasConfig,
    metaclass=abc.ABCMeta,
):
    """Abstract base class for a builder of a detector signal yield. Via the
    ``construct_detsigyield`` method it creates a DetSigYield instance holding
    the internal objects to calculate the detector signal yield.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """Constructor."""
        super().__init__(**kwargs)

    def assert_types_of_construct_detsigyield_arguments(
        self,
        dataset,
        data,
        shgs,
        ppbar,
    ):
        """Checks the types of the arguments for the ``construct_detsigyield``
        method. It raises errors if the arguments have the wrong type.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f'The dataset argument must be an instance of Dataset! Its current type is {classname(dataset)}.'
            )

        if not isinstance(data, DatasetData):
            raise TypeError(
                f'The data argument must be an instance of DatasetData! Its current type is {classname(data)}.'
            )

        if (not isinstance(shgs, SourceHypoGroup_t)) and (not issequenceof(shgs, SourceHypoGroup_t)):
            raise TypeError(
                'The shgs argument must be an instance of SourceHypoGroup '
                'or a sequence of SourceHypoGroup instances!'
                f'Its current type is {classname(shgs)}.'
            )

        if ppbar is not None and not isinstance(ppbar, ProgressBar):
            raise TypeError(
                f'The ppbar argument must be an instance of ProgressBar! Its current type is {classname(ppbar)}.'
            )

    def get_detsigyield_construction_factory(self) -> Callable | None:
        """This method is supposed to return a callable with the call-signature

        .. code::

            __call__(
                dataset,
                data,
                shgs,
                ppbar,
            )


        to construct several DetSigYield instances, one for each provided
        source hypo group (i.e. sources and fluxmodel).
        The return value of this callable must be a sequence of DetSigYield
        instances of the same length as the sequence of ``shgs``.

        Returns
        -------
        factory
            This default implementation returns ``None``, indicating that a
            factory is not supported by this builder.
        """
        return

    @abc.abstractmethod
    def construct_detsigyield(
        self,
        dataset: Dataset,
        data: DatasetData,
        shg: SourceHypoGroup_t,
        ppbar: ProgressBar | None = None,
    ) -> 'DetSigYield':
        """Abstract method to construct the DetSigYield instance.
        This method must be called by the derived class method implementation
        to ensure the compatibility check of the given flux model with the
        supported flux models.

        Parameters
        ----------
        dataset
            The instance of Dataset holding possible dataset specific settings.
        data
            The instance of DatasetData holding the monte-carlo event data.
        shg
            The instance of SourceHypoGroup (i.e. sources and flux model) for
            which the detector signal yield should be constructed.
        ppbar
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield
            An instance derived from DetSigYield.
        """


class NullDetSigYieldBuilder(DetSigYieldBuilder):
    """This class provides a dummy detector signal yield builder, which can
    be used for testing purposes, when an actual builder is not required.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        **kwargs,
    ):
        """Creates a new instance of NullDetSigYieldBuilder.

        Parameters
        ----------
        cfg
            The instance of Config holding the local configuration. Since this
            detector signal yield builder does nothing, this argument is
            optional. If not provided the default configuration is used.
        """
        if cfg is None:
            cfg = Config()

        super().__init__(cfg=cfg, **kwargs)

    def construct_detsigyield(
        self,
        *args,
        **kwargs,
    ):
        """Since this is a dummy detector signal yield builder, calling this
        method will raise a NotImplementedError!
        """
        raise NotImplementedError(
            f'The {classname(self)} detector signal yield builder cannot actually build a DetSigYield instance!'
        )


class TimeDependentDetSigYield(
    DetSigYield,
):
    r"""This class provides a detector signal yield for a source, whose flux
    varies in time according to a time flux profile :math:`T(t)`, i.e.
    :math:`\Phi(E,t) = \Phi_0 \Phi(E) T(t)`. It wraps a time-integrated
    detector signal yield, which is assumed to be proportional to the
    integrated detector live-time of the dataset, and scales it by the factor

    .. math::

        f = \frac{S}{L},

    where :math:`S` is the integral of the time flux profile over the detector
    on-time intervals of the dataset, and :math:`L` is the integrated live-time
    of the dataset. Hence, the detector signal yield is the expected number of
    signal events detected within the dataset given the time profile of the
    flux. For datasets, which do not overlap with the time flux profile, the
    detector signal yield is zero.

    .. note::

        The times of the time flux profile must be given in MJD (days), i.e.
        the same time format as the detector on-time intervals.

    The scaling factor is cached and recalculated automatically whenever the
    time flux profile instance or its parameter values change. The time flux
    profile parameters are not supposed to be fit parameters, hence no
    gradients w.r.t. them are provided.
    """

    def __init__(
        self,
        detsigyield: DetSigYield,
        livetime: Livetime,
        time_flux_profile: TimeFluxProfile,
        **kwargs,
    ):
        """Creates a new time-dependent detector signal yield instance.

        Parameters
        ----------
        detsigyield
            The instance of DetSigYield providing the time-integrated detector
            signal yield, which is proportional to the integrated live-time of
            the dataset.
        livetime
            The instance of Livetime providing the detector on-time intervals of
            the dataset.
        time_flux_profile
            The instance of TimeFluxProfile describing the time dependence of
            the source flux.
        """
        if not isinstance(detsigyield, DetSigYield):
            raise TypeError(
                f'The detsigyield argument must be an instance of DetSigYield! Its current type is '
                f'{classname(detsigyield)}.'
            )
        if not isinstance(livetime, Livetime):
            raise TypeError(
                f'The livetime argument must be an instance of Livetime! Its current type is {classname(livetime)}.'
            )

        super().__init__(
            param_names=detsigyield.param_names,
            dataset=detsigyield.dataset,
            fluxmodel=detsigyield.fluxmodel,
            livetime=livetime,
            **kwargs,
        )

        self._detsigyield = detsigyield
        self.time_flux_profile = time_flux_profile

        self._time_factor = None
        self._time_factor_key = None

    @property
    def detsigyield(self) -> DetSigYield:
        """(read-only) The wrapped time-integrated detector signal yield."""
        return self._detsigyield

    @property
    def time_flux_profile(self) -> TimeFluxProfile:
        """The instance of TimeFluxProfile describing the time dependence of the
        source flux.
        """
        return self._time_flux_profile

    @time_flux_profile.setter
    def time_flux_profile(self, profile):
        if not isinstance(profile, TimeFluxProfile):
            raise TypeError(
                f'The time_flux_profile property must be an instance of TimeFluxProfile! Its current type is '
                f'{classname(profile)}.'
            )
        self._time_flux_profile = profile

    def get_time_factor(self) -> float:
        """Returns the factor, by which the time-integrated detector signal
        yield is scaled, i.e. the integral of the time flux profile over the
        detector on-time intervals divided by the integrated live-time.

        Returns
        -------
        time_factor
            The scaling factor. It is zero if the time flux profile does not
            overlap with any detector on-time interval of the dataset.
        """
        key = (id(self._time_flux_profile), self._time_flux_profile.state)
        if key != self._time_factor_key:
            livetime = cast(Livetime, self._livetime)
            S = get_ontime_integral_of_TimeFluxProfile(profile=self._time_flux_profile, livetime=livetime)
            self._time_factor = S / Livetime.get_integrated_livetime(livetime)
            self._time_factor_key = key

        return cast(float, self._time_factor)

    def sources_to_recarray(
        self,
        sources: SourceModel | Sequence[SourceModel],
    ):
        """Converts the given sources into the source record array of the
        wrapped detector signal yield.
        """
        return self._detsigyield.sources_to_recarray(sources)

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        src_recarray,
        src_params_recarray,
    ) -> tuple[np.ndarray, dict]:
        """Calculates the detector signal yield of the wrapped detector signal
        yield scaled by the time factor.

        Parameters
        ----------
        src_recarray
            The numpy record array containing the information of the sources.
        src_params_recarray
            The numpy record ndarray containing the parameter values of the
            sources.

        Returns
        -------
        detsigyield
            The array with the mean number of signal in the detector for each
            given source.
        grads
            The dictionary holding the gradient values for each global fit
            parameter.
        """
        (values, grads) = cast(
            tuple[np.ndarray, dict],
            self._detsigyield(src_recarray=src_recarray, src_params_recarray=src_params_recarray),
        )

        f = self.get_time_factor()

        values = values * f
        grads = {gpidx: grad * f for (gpidx, grad) in grads.items()}

        return (values, grads)


class TimeDependentDetSigYieldBuilder(
    DetSigYieldBuilder,
):
    """This detector signal yield builder wraps a time-integrated detector
    signal yield builder and constructs instances of
    :class:`~skyllh.core.detsigyield.TimeDependentDetSigYield` for a given time
    flux profile. The detector on-time intervals are taken from the dataset
    data, if its ``livetime`` attribute is an instance of
    :class:`~skyllh.core.livetime.Livetime`, or from the livetimes set via the
    :meth:`set_livetime` method.
    """

    def __init__(
        self,
        builder: DetSigYieldBuilder,
        time_flux_profile: TimeFluxProfile,
        livetimes: dict[str, Livetime] | None = None,
        **kwargs,
    ):
        """Creates a new time-dependent detector signal yield builder.

        Parameters
        ----------
        builder
            The instance of DetSigYieldBuilder for constructing the
            time-integrated detector signal yields.
        time_flux_profile
            The instance of TimeFluxProfile describing the time dependence of
            the source flux.
        livetimes
            The optional dictionary mapping dataset names to Livetime instances.
        """
        if not isinstance(builder, DetSigYieldBuilder):
            raise TypeError(
                f'The builder argument must be an instance of DetSigYieldBuilder! Its current type is '
                f'{classname(builder)}.'
            )

        super().__init__(cfg=builder.cfg, **kwargs)

        self._builder = builder
        self.time_flux_profile = time_flux_profile
        self._livetimes = {} if livetimes is None else dict(livetimes)

    @property
    def builder(self) -> DetSigYieldBuilder:
        """(read-only) The wrapped time-integrated detector signal yield
        builder.
        """
        return self._builder

    @property
    def time_flux_profile(self) -> TimeFluxProfile:
        """The instance of TimeFluxProfile, which is used for the constructed
        detector signal yields.
        """
        return self._time_flux_profile

    @time_flux_profile.setter
    def time_flux_profile(self, profile):
        if not isinstance(profile, TimeFluxProfile):
            raise TypeError(
                f'The time_flux_profile property must be an instance of TimeFluxProfile! Its current type is '
                f'{classname(profile)}.'
            )
        self._time_flux_profile = profile

    def set_livetime(self, dataset_name: str, livetime: Livetime):
        """Sets the Livetime instance for the given dataset.

        Parameters
        ----------
        dataset_name
            The name of the dataset.
        livetime
            The instance of Livetime providing the detector on-time intervals of
            the dataset.
        """
        if not isinstance(livetime, Livetime):
            raise TypeError(
                f'The livetime argument must be an instance of Livetime! Its current type is {classname(livetime)}.'
            )
        self._livetimes[dataset_name] = livetime

    def get_livetime(self, dataset: Dataset, data: DatasetData) -> Livetime:
        """Retrieves the Livetime instance for the given dataset.

        Parameters
        ----------
        dataset
            The instance of Dataset.
        data
            The instance of DatasetData holding the data of the dataset.

        Returns
        -------
        livetime
            The instance of Livetime for the given dataset.
        """
        if dataset.name in self._livetimes:
            return self._livetimes[dataset.name]

        if isinstance(data.livetime, Livetime):
            return data.livetime

        raise KeyError(
            f'No Livetime instance is available for dataset "{dataset.name}"! Set it via the set_livetime method of '
            f'{classname(self)}.'
        )

    def _wrap(self, detsigyield: DetSigYield, dataset: Dataset, data: DatasetData) -> TimeDependentDetSigYield:
        """Wraps the given time-integrated detector signal yield."""
        return TimeDependentDetSigYield(
            detsigyield=detsigyield,
            livetime=self.get_livetime(dataset=dataset, data=data),
            time_flux_profile=self._time_flux_profile,
        )

    def get_detsigyield_construction_factory(self) -> Callable | None:
        """Returns the construction factory of the wrapped builder, which wraps
        the constructed detector signal yields, if the wrapped builder provides
        a factory. Otherwise ``None`` is returned.
        """
        factory = self._builder.get_detsigyield_construction_factory()
        if factory is None:
            return None

        def wrapped_factory(dataset, data, shgs, ppbar):
            detsigyields = factory(dataset=dataset, data=data, shgs=shgs, ppbar=ppbar)
            return [self._wrap(detsigyield=dsy, dataset=dataset, data=data) for dsy in detsigyields]

        return wrapped_factory

    def construct_detsigyield(
        self,
        dataset: Dataset,
        data: DatasetData,
        shg: SourceHypoGroup_t,
        ppbar: ProgressBar | None = None,
    ) -> TimeDependentDetSigYield:
        """Constructs the time-integrated detector signal yield with the wrapped
        builder and wraps it into a TimeDependentDetSigYield instance.

        Parameters
        ----------
        dataset
            The instance of Dataset holding possible dataset specific settings.
        data
            The instance of DatasetData holding the monte-carlo event data.
        shg
            The instance of SourceHypoGroup (i.e. sources and flux model) for
            which the detector signal yield should be constructed.
        ppbar
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield
            The instance of TimeDependentDetSigYield.
        """
        detsigyield = self._builder.construct_detsigyield(dataset=dataset, data=data, shg=shg, ppbar=ppbar)

        return self._wrap(detsigyield=detsigyield, dataset=dataset, data=data)
