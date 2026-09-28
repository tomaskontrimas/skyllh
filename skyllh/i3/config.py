"""This file defines IceCube specific global configuration."""

from skyllh.core.config import Config, resolve_config
from skyllh.core.datafields import (
    DataFieldStages as DFS,
)


def add_icecube_specific_analysis_required_data_fields(cfg: Config | None = None):
    """Adds IceCube specific data fields required by an IceCube analysis to
    the given local configuration.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration. If set to
        ``None``, the current Config instance is used, see
        :func:`~skyllh.core.config.resolve_config`.
    """
    cfg = resolve_config(cfg)

    cfg['datafields']['azi'] = DFS.ANALYSIS_EXP
    cfg['datafields']['zen'] = DFS.ANALYSIS_EXP
    cfg['datafields']['sin_dec'] = DFS.ANALYSIS_EXP
    cfg['datafields']['sin_true_dec'] = DFS.ANALYSIS_MC
