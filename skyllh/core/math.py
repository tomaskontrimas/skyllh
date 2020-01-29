# -*- coding: utf-8 -*-

"""The `math` module contains classes for pure mathematical objects.
"""

from __future__ import division

import abc

from skyllh.core.py import (
    classname,
    isproperty,
    issequence,
    issequenceof
)


class MathFunction(object):
    """This abstract base class provides an implementation for a mathematical
    function. Such a function has defined parameters, which are implemented as
    class properties. The tuple of parameter names is defined through the
    `parameter_names` property.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(MathFunction, self).__init__(**kwargs)

        self.parameter_names = ()

    @property
    def parameter_names(self):
        """The tuple holding the names of the math function's parameters.
        """
        return self._parameter_names
    @parameter_names.setter
    def parameter_names(self, names):
        if(not issequence(names)):
            names = (names,)
        if(not issequenceof(names, str)):
            raise TypeError('The parameter_names property must be a sequence '
                'of str!')
        names = tuple(names)
        # Check if all the given names are actual properties of this
        # MathFunction class.
        for name in names:
            if(not hasattr(self, name)):
                raise KeyError('The "%s" class does not have an attribute '
                    'named "%s"!'%(classname(self), name))
            if(not isproperty(self, name)):
                raise TypeError('The attribute "%s" of "%s" is not a '
                    'property!'%(classname(self), name))
        self._parameter_names = names

    @property
    @abc.abstractmethod
    def math_function_str(self):
        """The string showing the mathematical function of this MathFunction.
        """
        pass

    def __str__(self):
        """Pretty string representation of this MathFunction instance.
        """
        return self.math_function_str

    def copy(self, newparams=None):
        """Copies this MathFunction object by calling the copy.deepcopy
        function, and sets new parameters if requested.

        Parameters
        ----------
        newparams : dict | None
            The dictionary with the new parameter values to set, where the
            dictionary key is the parameter name and the dictionary value is the
            new value of the parameter.
        """
        f = deepcopy(self)

        # Set the new parameter values.
        if(newparams is not None):
            f.set_parameters(newparams)

        return f

    def set_parameters(self, pdict):
        """Sets the parameters of the math function to the given parameter
        values.

        Parameters
        ----------
        pdict : dict (name: value)
            The dictionary holding the names of the parameters and their new
            values.

        Returns
        -------
        updated : bool
            Flag if parameter values were actually updated.
        """
        if(not isinstance(pdict, dict)):
            raise TypeError('The pdict argument must be of type dict!')

        updated = False

        for pname in self._parameter_names:
            current_value = getattr(self, pname)
            pvalue = pdict.get(pname, current_value)
            if(pvalue != current_value):
                setattr(self, pname, pvalue)
                updated = True

        return updated