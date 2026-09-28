"""This file contains the global configuration dictionary, together with some
convenience utility functions to set different configuration settings.
"""

import copy
import os.path
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import (
    Any,
)

import yaml
from astropy import (
    units,
)

from skyllh.core.datafields import (
    DataFieldStages as DFS,
)
from skyllh.core.py import (
    classname,
)

_BASECONFIG = {
    'multiproc': {
        # The number of CPUs to use for functions that allow multi-processing.
        # If this setting is set to an int value in the range [1, N] this
        # setting will be used if a function's local ncpu setting is not
        # specified.
        'ncpu': None,
    },
    'logging': {
        # The log level of the loggers. The default is 'INFO'
        # (equivalent to logging.INFO).
        # Values are the standard log levels defined in the Python logging
        # module, specified either as strings or their corresponding integer
        # values. E.g.:
        # - 'DEBUG'  (10)
        # - 'INFO'   (20)
        # - 'WARNING' (30)
        'log_level': 'INFO',
        # The default log format.
        'log_format': ('%(asctime)s %(processName)s %(name)s %(levelname)s: %(message)s'),
        # Flag if detailed debug log messages, i.e. trace log messages, should
        # get generated. This is good for debugging but bad for performance.
        'enable_tracing': False,
    },
    'project': {
        # The project's working directory.
        'working_directory': '.',
    },
    'repository': {
        # A base path of repository datasets.
        'base_path': str(Path('~/.cache/skyllh').expanduser()),
        'download_from_origin': True,
    },
    'units': {
        # Definition of the internal units to use. These must match with the
        # units of the monte-carlo data files.
        'internal': {
            'angle': units.radian,
            'energy': units.GeV,
            'length': units.cm,
            'time': units.s,
        },
        'defaults': {
            # Definition of default units used for fluxes.
            'fluxes': {
                'angle': units.radian,
                'energy': units.GeV,
                'length': units.cm,
                'time': units.s,
            }
        },
    },
    'datafields': {
        'run': DFS.ANALYSIS_EXP,
        'ra': DFS.ANALYSIS_EXP,
        'dec': DFS.ANALYSIS_EXP,
        'ang_err': DFS.ANALYSIS_EXP,
        'time': DFS.ANALYSIS_EXP,
        'log_energy': DFS.ANALYSIS_EXP,
        'true_ra': DFS.ANALYSIS_MC,
        'true_dec': DFS.ANALYSIS_MC,
        'true_energy': DFS.ANALYSIS_MC,
        'mcweight': DFS.ANALYSIS_MC,
    },
    # Flag if specific calculations in the core module can be cached.
    'caching': {
        'pdf': {
            'MultiDimGridPDF': False,
        }
    },
}


class Config(
    dict,
):
    """This class, derived from dict, holds the a local configuration state."""

    def __init__(
        self,
    ) -> None:
        """Initializes a new Config instance holding the base configuration."""
        super().__init__(copy.deepcopy(_BASECONFIG))

    @classmethod
    def from_yaml(
        cls,
        pathfilename: str | None,
    ) -> 'Config':
        """Creates a new instance of Config holding the base configuration and
        updated by the configuration items contained in the yaml file using the
        :meth:`dict.update` method.

        Parameters
        ----------
        pathfilename
            Path and filename to the yaml file containing the to-be-updated
            configuration items.
            If set to ``None``, nothing is done.

        Returns
        -------
        cfg
            The instance of Config holding the base configuration and updated by
            the configuration given in the yaml file.
        """
        cfg = cls()

        if pathfilename is None:
            return cfg

        with open(pathfilename) as f:
            user_config_dict = yaml.safe_load(f)
        if user_config_dict is None:
            user_config_dict = {}
        elif not isinstance(user_config_dict, dict):
            raise TypeError(
                f'YAML configuration in "{pathfilename}" must be a mapping, got {type(user_config_dict).__name__}.'
            )

        cfg.update(user_config_dict)

        return cfg

    @classmethod
    def from_dict(
        cls,
        user_dict: dict[str, Any],
    ) -> 'Config':
        """Creates a new instance of Config holding the base configuration and
        updated by the given configuration dictionary using the
        :meth:`dict.update` method.

        Parameters
        ----------
        user_dict
            The dictionary containing the to-be-updated configuration items.

        Returns
        -------
        cfg
            The instance of Config holding the base configuration and updated by
            the given configuration dictionary.
        """
        cfg = cls()

        cfg.update(user_dict)

        return cfg

    @property
    def is_tracing_enabled(self):
        """``True``, if tracing mode is enabled, ``False`` otherwise."""
        return self['logging']['enable_tracing']

    def disable_tracing(
        self,
    ) -> 'Config':
        """Disables the tracing mode of SkyLLH.

        Returns
        -------
        self
            The updated instance of Config.
        """
        self['logging']['enable_tracing'] = False

        return self

    def enable_tracing(
        self,
    ) -> 'Config':
        """Enables the tracing mode of SkyLLH.

        Returns
        -------
        self
            The updated instance of Config.
        """
        self['logging']['enable_tracing'] = True

        return self

    def get_wd(
        self,
    ) -> str:
        """Retrieves the absolute path to the working directory as configured in
        this configuration.

        Returns
        -------
        wd
            The absolute path to the project's working directory.
        """
        wd = os.path.abspath(self['project']['working_directory'])

        return wd

    def set_enable_tracing(
        self,
        flag: bool,
    ) -> 'Config':
        """Sets the setting for tracing.

        Parameters
        ----------
        flag
            The flag if tracing should be enabled (``True``) or disabled
            (``False``).

        Returns
        -------
        self
            The updated instance of Config.
        """
        self['logging']['enable_tracing'] = flag

        return self

    def set_internal_units(
        self,
        angle_unit: units.UnitBase | None = None,
        energy_unit: units.UnitBase | None = None,
        length_unit: units.UnitBase | None = None,
        time_unit: units.UnitBase | None = None,
    ) -> 'Config':
        """Sets the units used internally to compute quantities. These units
        must match the units used in the monte-carlo files.

        Parameters
        ----------
        angle_unit
            The internal unit that should be used for angles.
            If set to ``None``, the unit is not changed.
        energy_unit
            The internal unit that should be used for energy.
            If set to ``None``, the unit is not changed.
        length_unit
            The internal unit that should be used for length.
            If set to ``None``, the unit is not changed.
        time_unit
            The internal unit that should be used for time.
            If set to ``None``, the unit is not changed.

        Returns
        -------
        self
            The updated instance of Config.
        """
        if angle_unit is not None:
            if not isinstance(angle_unit, units.UnitBase):
                raise TypeError('The angle_unit argument must be an instance of astropy.units.UnitBase!')
            self['units']['internal']['angle'] = angle_unit

        if energy_unit is not None:
            if not isinstance(energy_unit, units.UnitBase):
                raise TypeError('The energy_unit argument must be an instance of astropy.units.UnitBase!')
            self['units']['internal']['energy'] = energy_unit

        if length_unit is not None:
            if not isinstance(length_unit, units.UnitBase):
                raise TypeError('The length_unit argument must be an instance of astropy.units.UnitBase!')
            self['units']['internal']['length'] = length_unit

        if time_unit is not None:
            if not isinstance(time_unit, units.UnitBase):
                raise TypeError('The time_unit argument must be an instance of astropy.units.UnitBase!')
            self['units']['internal']['time'] = time_unit

        return self

    def set_ncpu(
        self,
        ncpu: int,
    ) -> 'Config':
        """Sets the global setting for the number of CPUs to use, when
        parallelization is available.

        Parameters
        ----------
        ncpu
            The number of CPUs.

        Returns
        -------
        self
            The updated instance of Config.
        """
        self['multiproc']['ncpu'] = ncpu

        return self

    def set_wd(
        self,
        path: str | None = None,
    ) -> str:
        """Sets the project's working directory configuration variable and adds
        it to the Python path variable.

        Parameters
        ----------
        path
            The path of the project's working directory. This can be a path
            relative to the path given by ``os.path.getcwd``, the current
            working directory of the program.
            If set to ``None``, the path is taken from the working directory
            setting of the given configuration.

        Returns
        -------
        wd
            The absolute path to the project's working directory.
        """
        if path is None:
            path = self['project']['working_directory']

        if self['project']['working_directory'] in sys.path:
            sys.path.remove(self['project']['working_directory'])

        wd = os.path.abspath(str(path))
        self['project']['working_directory'] = wd
        sys.path.insert(0, wd)

        return wd

    def to_internal_time_unit(
        self,
        time_unit: units.UnitBase,
    ):
        """Calculates the conversion factor from the given time unit to the
        internal time unit specified by this local configuration.

        Parameters
        ----------
        time_unit
            The time unit from which to convert to the internal time unit.
        """
        internal_time_unit = self['units']['internal']['time']
        factor = time_unit.to(internal_time_unit)

        return factor

    def wd_filename(self, filename: str) -> str:
        """Generates the fully qualified file name under the project's working
        directory of the given file.

        Parameters
        ----------
        filename
            The name of the file for which to generate the working directory
            path file name.

        Returns
        -------
        pathfilename
            The generated fully qualified path file name of ``filename`` with
            the project's working directory prefixed.
        """
        pathfilename = os.path.join(self.get_wd(), filename)

        return pathfilename


# The Config instance, which is used when no Config instance is passed
# explicitly. It is set via the ``use_config`` context manager.
_CURRENT_CONFIG: ContextVar[Config | None] = ContextVar('skyllh_current_config', default=None)

# The process-wide default Config instance, which is used when no Config
# instance is passed explicitly and no Config instance is set as the current
# one. It is created on first use.
_DEFAULT_CONFIG: Config | None = None


def get_default_config() -> Config:
    """Returns the process-wide default Config instance, which is used when no
    Config instance is passed explicitly and no Config instance is set via the
    :func:`use_config` context manager. It is created on first use.

    Returns
    -------
    cfg
        The default instance of Config.
    """
    global _DEFAULT_CONFIG

    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = Config()

    return _DEFAULT_CONFIG


def set_default_config(cfg: Config) -> None:
    """Sets the process-wide default Config instance, which is used when no
    Config instance is passed explicitly and no Config instance is set via the
    :func:`use_config` context manager. This is convenient for scripts and
    notebooks, where the configuration should be defined only once.

    Parameters
    ----------
    cfg
        The instance of Config that should be used as default.
    """
    global _DEFAULT_CONFIG

    if not isinstance(cfg, Config):
        raise TypeError(f'The cfg argument must be an instance of Config! Currently its type is {classname(cfg)}!')

    _DEFAULT_CONFIG = cfg


@contextmanager
def use_config(cfg: Config) -> Iterator[Config]:
    """Context manager, which sets the given Config instance as the current
    one. All SkyLLH objects and functions that are created or called within the
    context without an explicit Config instance, will use this Config instance.

    Worker processes of :func:`skyllh.core.multiproc.parallelize` inherit the
    current Config instance of the main process.

    Parameters
    ----------
    cfg
        The instance of Config that should be used within the context.

    Yields
    ------
    cfg
        The given instance of Config.
    """
    if not isinstance(cfg, Config):
        raise TypeError(f'The cfg argument must be an instance of Config! Currently its type is {classname(cfg)}!')

    token = _CURRENT_CONFIG.set(cfg)
    try:
        yield cfg
    finally:
        _CURRENT_CONFIG.reset(token)


def get_current_config() -> Config:
    """Returns the Config instance set via the :func:`use_config` context
    manager, or the process-wide default Config instance, if no Config instance
    is set.

    Returns
    -------
    cfg
        The current instance of Config.
    """
    cfg = _CURRENT_CONFIG.get()
    if cfg is None:
        cfg = get_default_config()

    return cfg


def resolve_config(
    cfg: Config | None = None,
    objs: Iterable[Any] | None = None,
) -> Config:
    """Determines the Config instance to use. The following order of precedence
    applies:

        1. The given ``cfg`` instance, if not ``None``.
        2. The Config instance of the first object in ``objs`` that holds a
           Config instance, i.e. is an instance of :class:`HasConfig`.
        3. The Config instance set via the :func:`use_config` context manager.
        4. The process-wide default Config instance, see
           :func:`get_default_config`.

    Parameters
    ----------
    cfg
        The explicitly given instance of Config, or ``None``.
    objs
        The optional iterable of objects, e.g. Dataset instances, whose Config
        instance should be used if ``cfg`` is ``None``.

    Returns
    -------
    cfg
        The instance of Config to use.
    """
    if cfg is not None:
        if not isinstance(cfg, Config):
            raise TypeError(f'The cfg argument must be an instance of Config! Currently its type is {classname(cfg)}!')
        return cfg

    if objs is not None:
        for obj in objs:
            if isinstance(obj, HasConfig):
                return obj.cfg

    return get_current_config()


class HasConfig:
    """Classifier class defining the cfg property. Classes that derive from
    this class indicate, that they hold an instance of Config.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        *args,
        **kwargs,
    ):
        """Creates a new instance having the property ``cfg``.

        Parameters
        ----------
        cfg
            The instance of Config holding the local configuration. If set to
            ``None``, the current Config instance is used, see
            :func:`resolve_config`.
        """
        super().__init__(*args, **kwargs)

        self.cfg = resolve_config(cfg)

    @property
    def cfg(self) -> Config:
        """The instance of Config holding the local configuration."""
        return self._cfg

    @cfg.setter
    def cfg(self, c):
        if not isinstance(c, Config):
            raise TypeError(f'The cfg property must be an instance of Config! Currently its type is {classname(c)}!')
        self._cfg = c
