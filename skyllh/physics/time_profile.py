# -*- coding: utf-8 -*-

from __future__ import division

import abc
import numpy as np

from skyllh.core.math import MathFunction
from skyllh.core.py import float_cast, classname

class TimeProfile(MathFunction, metaclass=abc.ABCMeta):
    """Abstract base class for an emission time profile of a source.
    """

    def __init__(self, t_start, t_end):
        """Creates a new time profile instance.

        Parameters
        ----------
        t_start : float
            The MJD start time of the box profile.
        t_end : float
            The MJD end time of the box profile.
        """
        super(TimeProfile, self).__init__()

        self.t_start = t_start
        self.t_end = t_end

    @property
    def t_start(self):
        """The MJD start time of the box profile.
        """
        return self._t_start
    @t_start.setter
    def t_start(self, t):
        t = float_cast(t,
            'The t_start property must be castable to type float!'
        )
        self._t_start = t

    @property
    def t_end(self):
        """The MJD end time of the box profile.
        """
        return self._t_end
    @t_end.setter
    def t_end(self, t):
        t = float_cast(t,
            'The t_end property must be castable to type float!'
        )
        self._t_end = t

    @property
    def duration(self):
        """The duration (in days) of the time profile.
        """
        return self._t_end - self._t_start

    @abc.abstractmethod
    def move(self, dt):
        """Abstract method to move the time profile by the given amount of time.

        Parameters
        ----------
        dt : float
            The MJD time difference of how far to move the time profile in time.
            This can be a positive or negative time shift.
        """
        pass

    @abc.abstractmethod
    def get_integral(self, t1, t2):
        """This method is supposed to calculate the integral of the time profile
        from time t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The MJD start time of the integration.
        t2 : float | array of float
            The MJD end time of the integration.

        Returns
        -------
        integral : array of float
            The integral value(s) of the time profile.
        """
        pass

    @abc.abstractmethod
    def get_total_integral(self):
        """This method is supposed to calculate the total integral of the time
        profile from t_start to t_end.

        Returns
        -------
        integral : float
            The integral value of the entire time profile.
        """
        pass

    @abc.abstractmethod
    def get_value(self, t):
        """Retrieves the value of the time profile at time t.

        Parameters
        ----------
        t : float
            The MJD time for which the time profile value should get retrieved.

        Returns
        -------
        value : float
            The time profile value at the given time.
        """
        pass


class BoxTimeProfile(TimeProfile):
    """The BoxTimeProfile describes a box-shaped emission time profile of a
    source. It has the following fit parameters:

        t_0 : float
            The mid MJD time of the box profile.
        t_w : float
            The width (days) of the box profile.
    """
    def __init__(self, t_0, t_w):
        """Creates a new box-shaped time profile instance.

        Parameters
        ----------
        t_0 : float
            The mid MJD time of the box profile.
        t_w : float
            The width (days) of the box profile.
        """
        t_start = t_0 - t_w/2.
        t_end = t_0 + t_w/2.

        super(BoxTimeProfile, self).__init__(t_start, t_end)

        self.param_names = ('t_0', 't_w')

    def move(self, dt):
        """Moves the box-shaped time profile by the time difference dt.

        Parameters
        ----------
        dt : float
            The MJD time difference of how far to move the time profile in time.
            This can be a positive or negative time shift.
        """
        self._t_start += dt
        self._t_end += dt

    @property
    def t_0(self):
        """The time of the mid point of the box.
        """
        return 0.5*(self._t_start + self._t_end)
    @t_0.setter
    def t_0(self, t):
        old_t_0 = self.t_0
        dt = t - old_t_0
        self.move(dt)

    @property
    def t_w(self):
        """The time width (in days) of the box.
        """
        return self._t_end - self._t_start
    @t_w.setter
    def t_w(self, w):
        t_0 = self.t_0
        self._t_start = t_0 - 0.5*w
        self._t_end = t_0 + 0.5*w

    def math_function_str(self):
        """The string showing the mathematical function of this box time
        profile.
        """
        return '1/%g'%(self.t_w)

    def get_integral(self, t1, t2):
        """Calculates the integral of the box-shaped time profile from MJD time
        t1 to time t2.

        Parameters
        ----------
        t1 : float | array of float
            The MJD start time(s) of the integration.
        t2 : float | array of float
            The MJD end time(s) of the integration.

        Returns
        -------
        integral : array of float
            The integral value(s).
        """
        t1 = np.atleast_1d(t1)
        t2 = np.atleast_1d(t2)

        integrals = np.zeros((t1.shape[0],), dtype=np.float)

        m = (t2 > self._t_start) & (t1 < self._t_end)
        N = np.count_nonzero(m)

        t1 = np.max(np.vstack((t1[m], np.repeat(self._t_start, N))).T, axis=1)
        t2 = np.min(np.vstack((t2[m], np.repeat(self._t_end, N))).T, axis=1)

        f = 1./self.duration

        integrals[m] = f*(t2-t1)

        return integrals

    def get_total_integral(self):
        """Calculates the the total integral of the box-shaped time profile
        from t_start to t_end. By definition it is 1.

        Returns
        -------
        integral : float
            The integral value of the entire time profile.
        """
        return 1.

    def get_value(self, t):
        """Retrieves the value of the box-shaped time profile at time t.
        For a box-shaped time profile the values are all equal to 1/duration
        for times within the time duration and zero otherwise.

        Parameters
        ----------
        t : float | array of float
            The MJD time(s) for which the time profile value(s) should get
            retrieved.

        Returns
        -------
        values : array of float
            The time profile value(s) at the given time(s).
        """
        t = np.atleast_1d(t)

        values = np.zeros((t.shape[0],), dtype=np.float)
        m = (t >= self._t_start) & (t < self._t_end)
        values[m] = 1./self.duration

        return values
