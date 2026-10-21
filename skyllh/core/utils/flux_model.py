import numpy as np
from scipy.stats import (
    rv_continuous,
)

from skyllh.core.flux_model import (
    TimeFluxProfile,
)
from skyllh.core.livetime import (
    Livetime,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.random import (
    RandomStateService,
)


def create_scipy_stats_rv_continuous_from_TimeFluxProfile(
    profile: TimeFluxProfile,
):
    """This function builds a scipy.stats.rv_continuous instance for a given
    :class:`~skyllh.core.flux_model.TimeFluxProfile` instance.

    It can be used to generate random numbers according to the given time flux
    profile function.

    Parameters
    ----------
    profile
        The instance of TimeFluxProfile providing the function of the time flux
        profile.

    Returns
    -------
    rv
        The instance of rv_continuous_frozen representing the time flux profile
        as a continuous random variate instance.
    """
    if not isinstance(profile, TimeFluxProfile):
        raise TypeError(
            f'The profile argument must be an instance of TimeFluxProfile! Its current type is {classname(profile)}!'
        )

    norm = 0
    tot_integral = profile.get_total_integral()
    if tot_integral != 0:
        norm = 1 / tot_integral

    class rv_continuous_from_TimeFluxProfile(rv_continuous):
        """This class provides a scipy ``rv_continuous`` random variable whose
        probability density is given by a time flux profile instance.
        """

        def __init__(self, *args, **kwargs):
            """Creates a new instance of the subclass of rv_continuous using
            the time flux profile.
            """
            self._profile = profile
            self._norm = norm

            super().__init__(*args, **kwargs)

        def _pdf(self, x, *args):
            """Calculates the probability density of the time flux profile
            function for given time values.
            """
            pd = self._profile(t=x) * self._norm

            return pd

        def _cdf(self, x, *args):
            """Calculates the cumulative distribution function values for rhe
            given time values. If the time flux profile instance provides a
            ``cdf`` method, it will be used. Otherwise the generic ``_cdf``
            method of the ``rv_continuous`` class will be used.
            """
            profile_cdf = getattr(self._profile, 'cdf', None)
            if callable(profile_cdf):
                return profile_cdf(t=x)

            return super()._cdf(x)

    rv = rv_continuous_from_TimeFluxProfile(
        a=profile.t_start,
        b=profile.t_stop,
    ).freeze(loc=0, scale=1)

    return rv


def get_ontime_intervals_of_TimeFluxProfile(
    profile: TimeFluxProfile,
    livetime: Livetime,
) -> np.ndarray:
    """Returns the detector on-time intervals that overlap with the time range
    of the given time flux profile, clipped to the time range of the profile.

    Parameters
    ----------
    profile
        The instance of TimeFluxProfile.
    livetime
        The instance of Livetime providing the detector on-time intervals.

    Returns
    -------
    intervals
        The (N_intervals,2)-shaped numpy ndarray holding the start and stop
        times of the on-time intervals. N_intervals can be zero.
    """
    return livetime.get_uptime_intervals_between(profile.t_start, profile.t_stop)


def get_ontime_integral_of_TimeFluxProfile(
    profile: TimeFluxProfile,
    livetime: Livetime,
) -> float:
    r"""Calculates the integral of the given time flux profile over the
    detector on-time, i.e.

    .. math::

        S = \sum_i \int_{t_{i,\mathrm{start}}}^{t_{i,\mathrm{stop}}} T(t) dt,

    where the sum runs over the detector on-time intervals.

    Parameters
    ----------
    profile
        The instance of TimeFluxProfile.
    livetime
        The instance of Livetime providing the detector on-time intervals.

    Returns
    -------
    S
        The integral of the time flux profile over the detector on-time. It is
        zero if the profile does not overlap with any detector on-time interval.
    """
    intervals = get_ontime_intervals_of_TimeFluxProfile(profile=profile, livetime=livetime)
    if len(intervals) == 0:
        return 0.0

    return float(np.sum(profile.get_integral(intervals[:, 0], intervals[:, 1])))


def draw_ontimes_from_TimeFluxProfile(
    profile: TimeFluxProfile,
    livetime: Livetime,
    rss: RandomStateService,
    size: int,
    n_bisect_iter: int = 64,
) -> np.ndarray:
    """Draws random times distributed according to the given time flux profile
    restricted to the detector on-time.

    The time flux profile can be of any shape. First, the on-time interval is
    drawn according to the profile integral within each on-time interval.
    Second, the time within the chosen interval is determined by inverting the
    cumulative profile integral numerically via bisection. Hence, the times are
    drawn exactly, without rejection sampling.

    Parameters
    ----------
    profile
        The instance of TimeFluxProfile describing the time distribution.
    livetime
        The instance of Livetime providing the detector on-time intervals.
    rss
        The instance of RandomStateService to draw random numbers from.
    size
        The number of times to draw.
    n_bisect_iter
        The number of bisection iterations to determine the time within an
        on-time interval. Each iteration halves the uncertainty of the time.

    Returns
    -------
    times
        The (size,)-shaped numpy ndarray holding the drawn times.

    Raises
    ------
    ValueError
        If times are requested, but the time flux profile does not overlap with
        any detector on-time interval.
    """
    if size == 0:
        return np.array([], dtype=np.float64)

    intervals = get_ontime_intervals_of_TimeFluxProfile(profile=profile, livetime=livetime)

    w = np.zeros((len(intervals),), dtype=np.float64)
    if len(intervals) > 0:
        w = np.atleast_1d(profile.get_integral(intervals[:, 0], intervals[:, 1])).astype(np.float64)
    W = np.sum(w)
    if not (W > 0):
        raise ValueError(
            f'Cannot draw {size} times from the time flux profile {profile}, because it does not overlap with any '
            f'detector on-time interval of {livetime}!'
        )

    # Draw the on-time interval for each time.
    cum_w = np.cumsum(w)
    u = rss.random.uniform(0, W, size)
    idxs = np.clip(np.searchsorted(cum_w, u, side='right'), 0, len(w) - 1)

    t_lo = intervals[idxs, 0].copy()
    t_hi = intervals[idxs, 1].copy()
    t_a = intervals[idxs, 0]

    # The target value of the profile integral from the start of the chosen
    # interval.
    target = rss.random.uniform(0, 1, size) * w[idxs]

    for _ in range(n_bisect_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        F = np.atleast_1d(profile.get_integral(t_a, t_mid))
        m = target > F
        t_lo = np.where(m, t_mid, t_lo)
        t_hi = np.where(m, t_hi, t_mid)

    return 0.5 * (t_lo + t_hi)
