# -*- coding: utf-8 -*-

r"""The `flux_model` module contains classes for different flux models. The
class for the most generic flux model is `FluxModel`, which is an abstract base
class. It describes a mathematical function for the differential flux:

.. math::

    d^4\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) / (dA d\Omega dE dt)
"""

import abc
import numpy as np
import scipy.stats

from astropy import units
from copy import deepcopy

from skyllh.core.config import CFG
from skyllh.core.math import MathFunction
from skyllh.core.model import Model
from skyllh.core.py import (
    classname,
    isproperty,
    issequence,
    issequenceof,
    float_cast
)
from skyllh.physics.flux_profile import (
    FluxProfile,
    SpatialFluxProfile
    UnitySpatialFluxProfile,
    UnityEnergyFluxProfile,
    UnityTimeFluxProfile
)


class FluxModel(MathFunction, Model, metaclass=abc.ABCMeta):
    r"""Abstract base class for all flux models
    :math:`\Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s)`.

    This base class defines the units used for the flux calculation. The unit
    of the flux is ([angle]^{-2} [energy]^{-1} [length]^{-2} [time]^{-1}).

    At this point the functional form of the flux model is not yet defined.
    """

    def __init__(
            self, angle_unit=None, energy_unit=None, length_unit=None,
            time_unit=None, **kwargs):
        """Creates a new FluxModel instance and defines the user-defined units.

        Parameters
        ----------
        angle_unit : instance of astropy.units.UnitBase | None
            The used unit for angles.
            If set to ``None``, the configured default angle unit for fluxes is
            used.
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        length_unit : instance of astropy.units.UnitBase | None
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        time_unit : instance of astropy.units.UnitBase | None
            The used unit for time.
            If set to ``None``, the configured default time unit for fluxes is
            used.
        """
        super(FluxModel, self).__init__(**kwargs)

        # Define the units.
        self.angle_unit = angle_unit
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.time_unit = time_unit

    @property
    def angle_unit(self):
        """The unit of angle used for the flux calculation.
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

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation.
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

    @property
    def length_unit(self):
        """The unit of length used for the flux calculation.
        """
        return self._length_unit
    @length_unit.setter
    def length_unit(self, unit):
        if(unit is None):
            unit = CFG['units']['defaults']['fluxes']['length']
        if(not isinstance(unit, units.UnitBase)):
            raise TypeError('The property length_unit must be of type '
                'astropy.units.UnitBase!')
        self._length_unit = unit

    @property
    def time_unit(self):
        """The unit of time used for the flux calculation.
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

    @property
    def unit_str(self):
        """The string representation of the flux unit.
        """
        return '1/(%s %s %s^2 %s)'%(
            self.energy_unit.to_string(), (self.angle_unit**2).to_string(),
            self.length_unit.to_string(), self.time_unit.to_string())

    @property
    def unit_latex_str(self):
        """The latex string representation of the flux unit.
        """
        return r'%s$^{-1}$ %s$^{-1}$ %s$^{-2}$ %s$^{-1}$'%(
            self.energy_unit.to_string(), (self.angle_unit**2).to_string(),
            self.length_unit.to_string(), self.time_unit.to_string())

    def __str__(self):
        """Pretty string representation of this class.
        """
        return self.math_function_str + ' ' + self.unit_str

    @abc.abstractmethod
    def __call__(
            self, alpha, delta, E, t,
            angle_unit=None, energy_unit=None, time_unit=None):
        """The call operator to retrieve a flux value for a given celestrial
        position, energy, and observation time.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1D numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1D numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1D numpy ndarray of float
            The energy for which to retrieve the flux value.
        t : float | (Ntime,)-shaped 1D numpy ndarray of float
            The observation time for which to retrieve the flux value.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given angles.
            If ``None``, the set angle unit of the flux model is assumed.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If ``None``, the set energy unit of the flux model is assumed.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If ``None``, the set time unit of the flux model is assumed.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values are in unit of the set flux model units
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        pass


class FactorizedFluxModel(FluxModel):
    r"""This class describes a flux model where the spatial, energy, and time
    profiles of the source factorize. That means the flux can be written as:

    .. math::

        \Phi_S(\alpha,\delta,E,t | \vec{x}_s,\vec{p}_s) =
            \Phi_0
            \Psi_{\mathrm{S}}(\alpha,\delta|\vec{p}_s)
            \epsilon_{\mathrm{S}}(E|\vec{p}_s)
            T_{\mathrm{S}}(t|\vec{p}_s)

    where, :math:`\Phi_0` is the normalization constant of the flux, and
    :math:`\Psi_{\mathrm{S}}`, :math:`\epsilon_{\mathrm{S}}`, and
    :math:`T_{\mathrm{S}}` are the spatial, energy, and time profiles of the
    flux, respectively.
    """
    def __init__(
            self, Phi0, spatial_profile=None, energy_profile=None,
            time_profile=None, length_unit=None, **kwargs):
        """Creates a new factorized flux model.

        Parameters
        ----------
        Phi0 : float
            The flux normalization constant. It must be given in the unit
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}, as defined
            by the spatial, energy, time profiles, and the length unit of this
            FactorizedFluxModel instance.
        spatial_profile : SpatialFluxProfile instance | None
            The SpatialFluxProfile instance providing the spatial profile
            function of the flux.
            If set to None, an instance of UnitySpatialFluxProfile will be used,
            which represents the constant function 1.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : TimeFluxProfile instance | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        length_unit : instance of astropy.units.UnitBase | None
            The used unit for length.
            If set to ``None``, the configured default length unit for fluxes is
            used.
        """
        self.Phi0 = Phi0
        self.spatial_profile = spatial_profile
        self.energy_profile = energy_profile
        self.time_profile = time_profile

        # The base class will set the default (internally used) flux unit, which
        # will be set automatically to the particular profile.
        super(FactorizedFluxModel, self).__init__(
            angle_unit=spatial_profile.angle_unit,
            energy_unit=energy_profile.energy_unit,
            time_unit=time_profile.time_unit,
            length_unit=length_unit,
            **kwargs
        )

        # Define the parameters which can be set via the `set_parameters`
        # method.
        self.parameter_names = ('Phi0',)

    @property
    def Phi0(self):
        """The flux normalization constant.
        The unit of this normalization constant is
        ([angle]^{-2} [energy]^{-1} [length]^{-2} [time]^{-1}).
        """
        return self._Phi0
    @Phi0.setter
    def Phi0(self, v):
        v = float_cast(v,
            'The Phi0 property must be castable to type float!')
        self._Phi0 = v

    @property
    def spatial_profile(self):
        """Instance of SpatialFluxProfile describing the spatial profile of the
        flux.
        """
        return self._spatial_profile
    @spatial_profile.setter
    def spatial_profile(self, profile):
        if(profile is None):
            profile = UnitySpatialFluxProfile()
        if(not isinstance(profile, SpatialFluxProfile)):
            raise TypeError('The spatial_profile property must be None, or an '
                'instance of SpatialFluxProfile!')
        self._spatial_profile = profile

    @property
    def energy_profile(self):
        """Instance of EnergyFluxProfile describing the energy profile of the
        flux.
        """
        return self._energy_profile
    @energy_profile.setter
    def energy_profile(self, profile):
        if(profile is None):
            profile = UnityEnergyFluxProfile()
        if(not isinstance(profile, EnergyFluxProfile)):
            raise TypeError('The energy_profile property must be None, or an '
                'instance of EnergyFluxProfile!')
        self._energy_profile = profile

    @property
    def time_profile(self):
        """Instance of TimeFluxProfile describing the time profile of the flux.
        """
        return self._time_profile
    @time_profile.setter
    def time_profile(self, profile):
        if(profile is None):
            profile = UnityTimeFluxProfile()
        if(not isinstance(profile, TimeFluxProfile)):
            raise TypeError('The time_profile property must be None, or an '
                'instance of TimeFluxProfile!')
        self._time_profile = profile

    @property
    def math_function_str(self):
        """The string showing the mathematical function of the flux.
        """
        return '%.3e * %s * %s * %s * %s'%(
            self._Phi0,
            self.unit_str,
            self._spatial_profile.math_function_str,
            self._energy_profile.math_function_str,
            self._time_profile.math_function_str
        )

    @property
    def angle_unit(self):
        """The unit of angle used for the flux calculation. The unit is
        taken and set from and to the set spatial flux profile, respectively.
        """
        return self._spatial_profile.angle_unit
    @angle_unit.setter
    def angle_unit(self, unit):
        self._spatial_profile.angle_unit = unit

    @property
    def energy_unit(self):
        """The unit of energy used for the flux calculation. The unit is
        taken and set from and to the set energy flux profile, respectively.
        """
        return self._energy_profile.energy_unit
    @energy_unit.setter
    def energy_unit(self, unit):
        self._energy_profile.energy_unit = unit

    @property
    def time_unit(self):
        """The unit of time used for the flux calculation. The unit is
        taken and set from and to the set time flux profile, respectively.
        """
        return self._time_profile.time_unit
    @time_unit.setter
    def time_unit(self, unit):
        self._time_profile.time_unit = unit

    @property
    def parameter_names(self):
        """The tuple holding the names of the math function's parameters. This
        is the total set of parameter names for all flux profiles of this
        FactorizedFluxModel instance.
        """
        pnames = list(self._parameter_names)
        pnames += self._spatial_profile.parameter_names
        pnames += self._energy_profile.parameter_names
        pnames += self._time_profile.parameter_names

        return tuple(pnames)
    @parameter_names.setter
    def parameter_names(self, names):
        super(FactorizedFluxModel, self.__class__).parameter_names.fset(
            self, names)

    def __call__(
            self, alpha, delta, E, t,
            angle_unit=None, energy_unit=None, time_unit=None):
        """Calculates the flux values for the given celestrial positions,
        energies, and observation times.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float
            The energy for which to retrieve the flux value.
        t : float | (Ntime,)-shaped 1d numpy ndarray of float
            The observation time for which to retrieve the flux value.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given angles.
            If ``None``, the set angle unit of the spatial flux profile is
            assumed.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If ``None``, the set energy unit of the energy flux profile is
            assumed.
        time_unit : instance of astropy.units.UnitBase | None
            The unit of the given times.
            If ``None``, the set time unit of the time flux profile is
            assumed.

        Returns
        -------
        flux : (Ncoord,Nenergy,Ntime)-shaped ndarray of float
            The flux values are in unit
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        spatial_profile_values = self._spatial_profile(
            alpha, delta, angle_unit=angle_unit)
        energy_profile_values = self._energy_profile(
            E, energy_unit=energy_unit)
        time_profile_values = self._time_profile(
            t, time_unit=time_unit)

        flux = (
            self._Phi0 *
            spatial_profile_values[:,np.newaxis,np.newaxis] *
            energy_profile_values[np.newaxis,:,np.newaxis] *
            time_profile_values[np.newaxis,np.newaxis,:]
        )

        return flux

    def set_parameters(self, pdict):
        """Sets the parameters of the flux model. For this factorized flux model
        it means that it sets the parameters of the spatial, energy, and time
        profiles.

        Parameters
        ----------
        pdict : dict
            The flux parameter dictionary.

        Returns
        -------
        updated : bool
            Flag if parameter values were actually updated.
        """
        updated = False

        updated |= super(FactorizedFluxModel, self).set_parameters(pdict)

        updated |= self._spatial_profile.set_parameters(pdict)
        updated |= self._energy_profile.set_parameters(pdict)
        updated |= self._time_profile.set_parameters(pdict)

        return updated


class IsPointlikeSource(object):
    """This is a classifier class that can be used by other classes to indicate
    that the specific class describes a point-like source.
    """
    def __init__(
            self, ra_func_instance=None, get_ra_func=None, set_ra_func=None,
            dec_func_instance=None, get_dec_func=None, set_dec_func=None,
            **kwargs):
        """Constructor method. Gets called when an instance of a class is
        created which derives from this IsPointlikeSource class.

        Parameters
        ----------

        """
        super(IsPointlikeSource, self).__init__(**kwargs)

        self._ra_func_instance = ra_func_instance
        self._get_ra_func = get_ra_func
        self._set_ra_func = set_ra_func

        self._dec_func_instance = dec_func_instance
        self._get_dec_func = get_dec_func
        self._set_dec_func = set_dec_func

    @property
    def ra(self):
        """The right-ascention coordinate of the point-like source.
        """
        return self._get_ra_func(self._ra_func_instance)
    @ra.setter
    def ra(self, v):
        self._set_ra_func(self._ra_func_instance, v)

    @property
    def dec(self):
        """The declination coordinate of the point-like source.
        """
        return self._get_dec_func(self._dec_func_instance)
    @dec.setter
    def dec(self, v):
        self._set_dec_func(self._dec_func_instance, v)


class PointlikeSourceFFM(FactorizedFluxModel, IsPointlikeSource):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point. This class provides the base class for a flux
    model of a point-like source.
    """
    def __init__(
            self, alpha_s, delta_s, Phi0, energy_profile, time_profile,
            angle_unit=None, length_unit=None):
        """Creates a new factorized flux model for a point-like source.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        Phi0 : float
            The flux normalization constant in unit of flux.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        time_profile : TimeFluxProfile instance | None
            The TimeFluxProfile instance providing the time profile function
            of the flux.
            If set to None, an instance of UnityTimeFluxProfile will be used,
            which represents the constant function 1.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit for angles used for the flux unit.
            If set to ``None``, the configured internal angle unit is used.
        length_unit : instance of astropy.units.UnitBase | None
            The unit for length used for the flux unit.
            If set to ``None``, the configured internal length unit is used.
        """
        spatial_profile=PointSpatialFluxProfile(
            alpha_s, delta_s, angle_unit=angle_unit)

        super(PointlikeSourceFFM, self).__init__(
            Phi0=Phi0,
            spatial_profile=spatial_profile,
            energy_profile=energy_profile,
            time_profile=time_profile,
            length_unit=length_unit,
            ra_func_instance=spatial_profile,
            get_ra_func=spatial_profile.__class__.alpha_s.fget,
            set_ra_func=spatial_profile.__class__.alpha_s.fset,
            dec_func_instance=spatial_profile,
            get_dec_func=spatial_profile.__class__.delta_s.fget,
            set_dec_func=spatial_profile.__class__.delta_s.fset
        )


class SteadyPointlikeSourceFFM(PointlikeSourceFFM):
    """This class describes a factorized flux model (FFM), where the spatial
    profile is modeled as a point and the time profile as constant 1. It is
    derived from the ``PointlikeSourceFFM`` class.
    """
    def __init__(
            self, alpha_s, delta_s, Phi0, energy_profile,
            angle_unit=None, length_unit=None, time_unit=None):
        """Creates a new factorized flux model for a point-like source with no
        time dependance.

        Parameters
        ----------
        alpha_s : float
            The right-ascention of the point-like source.
        delta_s : float
            The declination of the point-like source.
        Phi0 : float
            The flux normalization constant.
        energy_profile : EnergyFluxProfile instance | None
            The EnergyFluxProfile instance providing the energy profile
            function of the flux.
            If set to None, an instance of UnityEnergyFluxProfile will be used,
            which represents the constant function 1.
        """
        super(SteadyPointlikeSourceFFM, self).__init__(
            alpha_s=alpha_s,
            delta_s=delta_s,
            Phi0=Phi0,
            energy_profile=energy_profile,
            time_profile=UnityTimeFluxProfile(time_unit=time_unit),
            angle_unit=angle_unit,
            length_unit=length_unit
        )

    def __call__(
            self, alpha, delta, E,
            angle_unit=None, energy_unit=None):
        """Calculates the flux values for the given celestrial positions, and
        energies.

        Parameters
        ----------
        alpha : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The right-ascention coordinate for which to retrieve the flux value.
        delta : float | (Ncoord,)-shaped 1d numpy ndarray of float
            The declination coordinate for which to retrieve the flux value.
        E : float | (Nenergy,)-shaped 1d numpy ndarray of float
            The energy for which to retrieve the flux value.
        angle_unit : instance of astropy.units.UnitBase | None
            The unit of the given angles.
            If ``None``, the set angle unit of the spatial flux profile is
            assumed.
        energy_unit : instance of astropy.units.UnitBase | None
            The unit of the given energies.
            If ``None``, the set energy unit of the energy flux profile is
            assumed.

        Returns
        -------
        flux : (Ncoord,Nenergy)-shaped ndarray of float
            The flux values are in unit
            [energy]^{-1} [angle]^{-2} [length]^{-2} [time]^{-1}.
        """
        spatial_profile_values = self._spatial_profile(
            alpha, delta, unit=angle_unit)
        energy_profile_values = self._energy_profile(
            E, unit=energy_unit)

        flux = (
            self._Phi0 *
            spatial_profile_values[:,np.newaxis] *
            energy_profile_values[np.newaxis,:]
        )

        return flux
