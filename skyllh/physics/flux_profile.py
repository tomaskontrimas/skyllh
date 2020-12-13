# -*- coding: utf-8 -*-

r"""The `flux_profile` module contains classes describing different flux profile math functions. Those flux profiles can be used in a flux model class.
"""

import abc
import numpy as np
import scipy.stats

from astropy import units

from skyllh.core.config import CFG
from skyllh.core.math import MathFunction
from skyllh.core.py import (
    float_cast
)


class FluxProfile(MathFunction, metaclass=abc.ABCMeta):
    """The abstract base class for a flux profile math function.
    """

    def __init__(self):
        super(FluxProfile, self).__init__()


class SpatialFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for a spatial flux profile function.
    """

    def __init__(
            self, angle_unit=None):
        """Creates a new SpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super(SpatialFluxProfile, self).__init__()

        self.angle_unit = angle_unit

    @property
    def angle_unit(self):
        """The set unit of angle used for this spatial flux profile.
        If set to ``Ǹone`` the configured default angle unit for fluxes is used.
        """
        return self._angle_unit
    @angle_unit.setter
    def angle_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['angle']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property angle_unit must be of type '
                'astropy.units.UnitBase!')
        self._angle_unit = unit

    @abc.abstractmethod
    def __call__(self, alpha, delta, angle_unit=None):
        """This method is supposed to return the spatial profile value for the
        given celestrial coordinates.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            If ``None``, the set angle unit of this SpatialFluxProfile is
            assumed.

        Returns
        -------
        values : 1D numpy ndarray
            The spatial profile values.
        """
        pass


class UnitySpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for the constant profile function 1 for any spatial
    coordinates.
    """
    def __init__(self, angle_unit=None):
        """Creates a new UnitySpatialFluxProfile instance.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super(UnitySpatialFluxProfile, self).__init__(
            angle_unit=angle_unit)

    @property
    def math_function_str(self):
        return '1'

    def __call__(self, alpha, delta, angle_unit=None):
        """Returns 1 as numpy ndarray in same shape as alpha and delta.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate.
        delta : float | 1d numpy ndarray of float
            The declination coordinate.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles.
            By the definition of this class this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray
            1 in same shape as alpha and delta.
        """
        (alpha, delta) = np.atleast_1d(alpha, delta)
        if(alpha.shape != delta.shape):
            raise ValueError('The alpha and delta arguments must be of the '
                'same shape!')

        return np.ones_like(alpha)


class PointSpatialFluxProfile(SpatialFluxProfile):
    """Spatial flux profile for a delta function at the celestrical coordinate
    (alpha_s, delta_s).
    """
    def __init__(self, alpha_s, delta_s, angle_unit=None):
        """Creates a new spatial flux profile for a point.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``Ǹone``, the configured default angle unit for fluxes is
            used.
        """
        super(PointSpatialFluxProfile, self).__init__(
            angle_unit=angle_unit)

        self.alpha_s = alpha_s
        self.delta_s = delta_s

        # Define the names of the parameters, which can be updated.
        self.param_names = ('alpha_s', 'delta_s')

    @property
    def alpha_s(self):
        """The right-ascention of the point-like source.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._alpha_s
    @alpha_s.setter
    def alpha_s(self, v):
        v = float_cast(v,
            'The alpha_s property must be castable to type float!')
        self._alpha_s = v

    @property
    def delta_s(self):
        """The declination of the point-like source.
        The unit is the set angle unit of this SpatialFluxProfile instance.
        """
        return self._delta_s
    @delta_s.setter
    def delta_s(self, v):
        v = float_cast(v,
            'The delta_s property must be castable to type float!')
        self._delta_s = v

    @property
    def math_function_str(self):
        """(read-only) The string representation of the mathematical function of
        this spatial flux profile instance.
        """
        return 'delta(alpha-%g%s)*delta(delta-%g%s)'%(
            self._alpha_s, self._angle_unit.to_string(), self._delta_s,
            self._angle_unit.to_string())

    def __call__(self, alpha, delta, angle_unit=None):
        """Returns a numpy ndarray in same shape as alpha and delta with 1 if
        `alpha` equals `alpha_s` and `delta` equals `delta_s`, and 0 otherwise.

        Parameters
        ----------
        alpha : float | 1d numpy ndarray of float
            The right-ascention coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        delta : float | 1d numpy ndarray of float
            The declination coordinate at which to evaluate the spatial flux
            profile. The unit must be the internally used angle unit.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given celestrial angles. The values will be
            converted to the set angle unit of this SpatialFluxProfile instance.
            If set to ``None``, the set angle unit of this SpatialFluxProfile
            instance is assumed.

        Returns
        -------
        value : 1D numpy ndarray of int8
            A numpy ndarray in same shape as alpha and delta with 1 if `alpha`
            equals `alpha_s` and `delta` equals `delta_s`, and 0 otherwise.
        """
        (alpha, delta) = np.atleast_1d(alpha, delta)
        if(alpha.shape != delta.shape):
            raise ValueError('The alpha and delta arguments must be of the '
                'same shape!')

        # Convert the angle values to the set angle unit of this
        # SpatialFluxProfile instance if a angle unit is specified and does not
        # match the set angle unit of this SpatialFluxProfile instance.
        if((angle_unit is not None) and (angle_unit != self._angle_unit)):
            angle_unit_conv_factor = angle_unit.to(self._angle_unit)
            # Convert by creating a copy to not alter the input arrays.
            alpha = alpha * angle_unit_conv_factor
            delta = delta * angle_unit_conv_factor

        value = ((alpha == self._alpha_s) &
                 (delta == self._delta_s)).astype(np.int8, copy=False)

        return value


class EnergyFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for an energy flux profile function.
    """

    def __init__(self, energy_unit=None):
        """Creates a new energy flux profile with a given energy unit to be used
        for flux calculation.

        Parameters
        ----------
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super(EnergyFluxProfile, self).__init__()

        # Set the energy unit.
        self.energy_unit = energy_unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux profile calculation.
        """
        return self._energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['energy']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property energy_unit must be of type '
                'astropy.units.UnitBase!')
        self._energy_unit = unit

    @abc.abstractmethod
    def __call__(self, E, energy_unit=None):
        """This method is supposed to return the energy profile value for the
        given energy value.

        Parameters
        ----------
        E : float | 1d numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energy value(s). The energy values are
            converted to the set energy unit of this EnergyFluxProfile instance.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The energy profile values for the given energies.
        """
        pass


class UnityEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for the constant function 1.
    """
    def __init__(self, energy_unit=None):
        """Creates a new UnityEnergyFluxProfile instance.

        Parameters
        ----------
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super(UnityEnergyFluxProfile, self).__init__(
            energy_unit=energy_unit)

    @property
    def math_function_str(self):
        """The string representation of the mathematical function of this energy
        flux profile.
        """
        return '1'

    def __call__(self, E, energy_unit=None):
        """Returns 1 as numpy ndarray in some shape as E.

        Parameters
        ----------
        E : float | 1D numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            By definition of this specific class, this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray of int8
            1 in same shape as E.
        """
        E = np.atleast_1d(E)

        values = np.ones_like(E, dtype=np.int8)

        return values


class PowerLawEnergyFluxProfile(EnergyFluxProfile):
    """Energy flux profile for a power law profile with a reference energy
    ``E0`` and a spectral index ``gamma``.

    .. math::
        (E / E_0)^{-\gamma}
    """
    def __init__(self, E0, gamma, energy_unit=None):
        """Creates a new power law flux profile with the reference energy ``E0``
        and spectral index ``gamma``.

        Parameters
        ----------
        E0 : castable to float
            The reference energy.
        gamma : castable to float
            The spectral index.
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super(PowerLawEnergyFluxProfile, self).__init__(
            energy_unit=energy_unit)

        self.E0 = E0
        self.gamma = gamma

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('E0', 'gamma')

    @property
    def E0(self):
        """The reference energy in the set energy unit of this EnergyFluxProfile
        instance.
        """
        return self._E0
    @E0.setter
    def E0(self, v):
        v = float_cast(v,
            'Property E0 must be castable to type float!')
        self._E0 = v

    @property
    def gamma(self):
        """The spectral index.
        """
        return self._gamma
    @gamma.setter
    def gamma(self, v):
        v = float_cast(v,
            'Property gamma must be castable to type float!')
        self._gamma = v

    @property
    def math_function_str(self):
        """The string representation of this EnergyFluxProfile instance.
        """
        return '(E / (%g %s))^-%g'%(self._E0, self._energy_unit, self._gamma)

    def __call__(self, E, energy_unit=None):
        """Returns the power law values for the given energies as numpy ndarray
        in same shape as E.

        Parameters
        ----------
        E : float | 1D numpy ndarray of float
            The energy value for which to retrieve the energy profile value.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies. The energy values will be converted
            to the set energy unit of this PowerLawEnergyFluxProfile instance.
            If set to ``None``, the set energy unit of this EnergyFluxProfile
            instance is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The energy profile values for the given energies.
        """
        E = np.atleast_1d(E)

        if((energy_unit is not None) and (energy_unit != self._energy_unit)):
            energy_unit_conv_factor = energy_unit.to(self._energy_unit)
            # Covert by creating a copy to not alter the input array.
            E = E * energy_unit_conv_factor

        value = np.power(E / self._E0, -self._gamma)

        return value


class TimeFluxProfile(FluxProfile, metaclass=abc.ABCMeta):
    """The abstract base class for a time flux profile function.
    """

    def __init__(self, t_start=-np.inf, t_end=np.inf, time_unit=None):
        """Creates a new time flux profile instance.

        Parameters
        ----------
        t_start : float
            The start time of the time profile.
            If set to -inf, it means, that the profile starts at the beginning
            of the entire time-span of the dataset.
        t_end : float
            The end time of the time profile.
            If set to +inf, it means, that the profile ends at the end of the
            entire time-span of the dataset.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super(TimeFluxProfile, self).__init__()

        self.time_unit = time_unit

        self.t_start = t_start
        self.t_end = t_end

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('t_start', 't_end')

    @property
    def t_start(self):
        """The start time of the time profile. Can be -inf which means, that
        the profile starts at the beginning of the entire dataset.
        """
        return self._t_start
    @t_start.setter
    def t_start(self, t):
        t = float_cast(t,
            'The t_start property must be castable to type float!')
        self._t_start = t

    @property
    def t_end(self):
        """The end time of the time profile. Can be +inf which means, that
        the profile ends at the end of the entire dataset.
        """
        return self._t_end
    @t_end.setter
    def t_end(self, t):
        t = float_cast(t,
            'The t_end property must be castable to type float!')
        self._t_end = t

    @property
    def duration(self):
        """(read-only) The duration of the time profile.
        """
        return self._t_end - self._t_start

    @property
    def time_unit(self):
        """The unit of time used for the flux profile calculation.
        """
        return self._time_unit
    @time_unit.setter
    def time_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['time']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property time_unit must be of type '
                'astropy.units.UnitBase!')
        self._time_unit = unit

    def get_total_integral(self):
        """Calculates the total integral of the time profile from t_start to
        t_end.

        Returns
        -------
        integral : float
            The integral value of the entire time profile.
            The value is in the set time unit of this TimeFluxProfile instance.
        """
        integral = self.get_integral(self._t_start, self._t_end)

        return integral

    @abc.abstractmethod
    def __call__(self, t, time_unit=None):
        """This method is supposed to return the time profile value for the
        given times.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values will be converted to
            the set time unit of this TimeFluxProfile instance.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The time profile values.
        """
        pass

    @abc.abstractmethod
    def move(self, dt, time_unit=None):
        """Abstract method to move the time profile by the given amount of time.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given time difference. The time difference value
            will be converted to the set time unit of this TimeFluxProfile
            instance.
            If set to ``Ǹone``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """
        pass

    @abc.abstractmethod
    def get_integral(self, t1, t2, time_unit=None):
        """This method is supposed to calculate the integral of the time profile
        from time ``t1`` to time ``t2``.

        Parameters
        ----------
        t1 : float | array of float
            The start time of the integration.
        t2 : float | array of float
            The end time of the integration.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values will be converted to
            the set time unit of this TimeFluxProfile instance.
            If set to ``Ǹone``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s) of the time profile. The values are in the
            set time unit of this TimeFluxProfile instance.
        """
        pass


class UnityTimeFluxProfile(TimeFluxProfile):
    """Time flux profile for the constant profile function ``1``.
    """
    def __init__(self, time_unit=None):
        super(UnityTimeFluxProfile, self).__init__(
            time_unit=time_unit)

    @property
    def math_function_str(self):
        return '1'

    def __call__(self, t, time_unit=None):
        """Returns 1 as numpy ndarray in same shape as t.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            By definition of this specific class, this argument is ignored.

        Returns
        -------
        values : 1D numpy ndarray of int8
            1 in same shape as ``t``.
        """
        t = np.atleast_1d(t)

        values = np.ones_like(t, dtype=np.int8)

        return values

    def move(self, dt, time_unit=None):
        """Moves the time profile by the given amount of time. By definition
        this method does nothing, because the profile is 1 over the entire
        dataset time range.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given time difference.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.
        """
        pass

    def get_integral(self, t1, t2, time_unit=None):
        """Calculates the integral of the time profile from time t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The start time of the integration.
        t2 : float | array of float
            The end time of the integration.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values are converted to the
            set time unit of this UnityTimeFluxProfile instance.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s) of the time profile. The values are in the
            set time unit of this TimeFluxProfile instance.
        """
        if((time_unit is not None) and (time_unit != self._time_unit)):
            time_unit_conv_factor = time_unit.to(self._time_unit)
            # Convert by creating a copy to not alter the input arrays.
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        integral = t2 - t1

        return integral


class BoxTimeFluxProfile(TimeFluxProfile):
    """This class describes a box-shaped time flux profile.
    It has the following parameters:

        t0 : float
            The mid time of the box profile.
        tw : float
            The width of the box profile.

    The box is centered at ``t0`` and extends to +/-``tw``/2 around ``t0``.
    """
    def __init__(self, t0, tw, time_unit=None):
        """Creates a new box-shaped time profile instance.

        Parameters
        ----------
        t0 : float
            The mid time of the box profile.
        tw : float
            The width of the box profile.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        t_start = t0 - tw/2.
        t_end = t0 + tw/2.

        super(BoxTimeFluxProfile, self).__init__(
            t_start=t_start, t_end=t_end, time_unit=time_unit)

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('t0', 'tw')

    @property
    def t0(self):
        """The time of the mid point of the box.
        The value is in the set time unit of this TimeFluxProfile instance.
        """
        return 0.5*(self._t_start + self._t_end)
    @t0.setter
    def t0(self, t):
        old_t0 = self.t0
        dt = t - old_t0
        self.move(dt)

    @property
    def tw(self):
        """The time width of the box.
        The value is in the set time unit of this TimeFluxProfile instance.
        """
        return self._t_end - self._t_start
    @tw.setter
    def tw(self, w):
        t0 = self.t0
        self._t_start = t0 - 0.5*w
        self._t_end = t0 + 0.5*w

    @property
    def math_function_str(self):
        t0 = self.t0
        tw = self.tw
        return '1 for t in [%g-%g/2; %g+%g/2], 0 otherwise'%(
            t0, tw, t0, tw)

    def __call__(self, t, time_unit=None):
        """Returns 1 for all t within the interval [t0-tw/2; t0+tw/2], and 0
        otherwise.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values are converted to the
            set time unit of this BoxTimeFluxProfile instance.
            If set to ``None``, the set time unit of this BoxTimeFluxProfile
            instance is assumed.

        Returns
        -------
        values : 1D numpy ndarray of int8
            The value(s) of the time flux profile for the given time(s).
            The time profile value is 1 of the time value is within in the time
            box, and 0 otherwise.
        """
        t = np.atleast_1d(t)

        if((time_unit is not None) and (time_unit != self._time_unit)):
            time_unit_conv_factor = time_unit.to(self._time_unit)
            t = t * time_unit_conv_factor

        values = np.zeros((t.shape[0],), dtype=np.int8)
        m = (t >= self._t_start) & (t <= self._t_end)
        values[m] = 1

        return values

    def move(self, dt, time_unit=None):
        """Moves the box-shaped time profile by the time difference dt.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of ``dt``. The time difference is converted to the set time
            unit of this BoxTimeFluxProfile instance.
            If set to ``None``, the set time unit of this BoxTimeFluxProfile
            instance is assumed.
        """
        if((time_unit is not None) and (time_unit != self._time_unit)):
            dt = dt * time_unit.to(self._time_unit)

        self._t_start += dt
        self._t_end += dt

    def get_integral(self, t1, t2, time_unit=None):
        """Calculates the integral of the box-shaped time flux profile from
        time t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The start time(s) of the integration.
        t2 : float | array of float
            The end time(s) of the integration.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values are converted to the
            set time unit of this BoxTimeFluxProfile instance.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s). The values are in the set time unit of this
            BoxTimeFluxProfile instance.
        """
        t1 = np.atleast_1d(t1)
        t2 = np.atleast_1d(t2)

        if((time_unit is not None) and (time_unit != self._time_unit)):
            time_unit_conv_factor = time_unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        integral = np.zeros((t1.shape[0],), dtype=np.float)

        m = (t2 >= self._t_start) & (t1 <= self._t_end)
        N = np.count_nonzero(m)

        t1 = np.max(np.vstack((t1[m], np.repeat(self._t_start, N))).T, axis=1)
        t2 = np.min(np.vstack((t2[m], np.repeat(self._t_end, N))).T, axis=1)

        integral[m] = t2 - t1

        return integral


class GaussianTimeFluxProfile(TimeFluxProfile):
    """This class describes a gaussian-shaped time flux profile.
    It has the following parameters:

        t0 : float
            The mid time of the gaussian profile.
        sigma_t : float
            The one-sigma width of the gaussian profile.
    """
    def __init__(self, t0, sigma_t, tol=1e-12, time_unit=None):
        """Creates a new gaussian-shaped time profile instance.

        Parameters
        ----------
        t0 : float
            The mid time of the gaussian profile.
        sigma_t : float
            The one-sigma width of the gaussian profile.
        tol : float
            The tolerance of the gaussian value. This defines the start and end
            time of the gaussian profile.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        # Calculate the start and end time of the gaussian profile, such that
        # at those times the gaussian values obey the given tolerance.
        dt = np.sqrt(-2*sigma_t*sigma_t*np.log(np.sqrt(2*np.pi)*sigma_t*tol))
        t_start = t0 - dt
        t_end = t0 + dt

        # A Gaussian profile extends to +/- infinity by definition.
        super(GaussianTimeFluxProfile, self).__init__(
            t_start=t_start, t_end=t_end, time_unit=time_unit)

        self.t0 = t0
        self.sigma_t = sigma_t

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('t0', 'sigma_t')

    @property
    def t0(self):
        """The time of the mid point of the gaussian profile.
        The unit of the value is the set time unit of this TimeFluxProfile
        instance.
        """
        return 0.5*(self._t_start + self._t_end)
    @t0.setter
    def t0(self, t):
        t = float_cast(t,
            'The t0 property must be castable to type float!')
        old_t0 = self.t0
        dt = t - old_t0
        self.move(dt)

    @property
    def sigma_t(self):
        """The one-sigma width of the gaussian profile.
        The unit of the value is the set time unit of this TimeFluxProfile
        instance.
        """
        return self._sigma_t
    @sigma_t.setter
    def sigma_t(self, sigma):
        sigma = float_cast(sigma,
            'The sigma_t property must be castable to type float!')
        self._sigma_t = sigma

    def __call__(self, t, time_unit=None):
        """Returns the gaussian profile value for the given time ``t``.

        Parameters
        ----------
        t : float | 1D numpy ndarray of float
            The time(s) for which to get the time flux profile values.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values are converted to the
            set time unit of this GaussianTimeFluxProfile instance.
            If set to ``None``, the set time unit of this
            GaussianTimeFluxProfile is assumed.

        Returns
        -------
        values : 1D numpy ndarray of float
            The value(s) of the time flux profile for the given time(s).
        """
        t = np.atleast_1d(t)

        if((time_unit is not None) and (time_unit != self._time_unit)):
            time_unit_conv_factor = time_unit.to(self._time_unit)
            t = t * time_unit_conv_factor

        s = self._sigma_t
        twossq = 2*s*s
        t0 = 0.5*(self._t_end + self._t_start)
        dt = t - t0

        values = 1/(np.sqrt(np.pi*twossq)) * np.exp(-dt*dt/twossq)

        return values

    def move(self, dt, time_unit=None):
        """Moves the gaussian time profile by the given amount of time.

        Parameters
        ----------
        dt : float
            The time difference of how far to move the time profile in time.
            This can be a positive or negative time shift value.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given time difference. The time difference is
            converted to the set unit of this GaussianTimeFluxProfile instance.
            If set to ``None``, the set time unit of this
            GaussianTimeFluxProfile instance is assumed.
        """
        if((unit is not None) and (unit != self._time_unit)):
            dt = dt * unit.to(self._time_unit)

        self._t_start += dt
        self._t_end += dt

    def get_integral(self, t1, t2, time_unit=None):
        """Calculates the integral of the gaussian time profile from time ``t1``
        to time ``t2``.

        Parameters
        ----------
        t1 : float | array of float
            The start time(s) of the integration.
        t2 : float | array of float
            The end time(s) of the integration.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times. The time values are converted to the
            set time unit of this GaussianTimeFluxProfile instance.
            If set to ``None``, the set time unit of this TimeFluxProfile
            instance is assumed.

        Returns
        -------
        integral : array of float
            The integral value(s). The values are in the set time unit of
            this TimeFluxProfile instance.
        """
        if((time_unit is not None) and (time_unit != self._time_unit)):
            time_unit_conv_factor = time_unit.to(self._time_unit)
            t1 = t1 * time_unit_conv_factor
            t2 = t2 * time_unit_conv_factor

        t0 = 0.5*(self._t_end + self._t_start)
        sigma_t = self._sigma_t

        c1 = scipy.stats.norm.cdf(t1, loc=t0, scale=sigma_t)
        c2 = scipy.stats.norm.cdf(t2, loc=t0, scale=sigma_t)

        integral = c2 - c1

        return integral
