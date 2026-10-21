from collections.abc import Callable

import numpy as np

from skyllh.core.background_generation import (
    BackgroundGenerationMethod,
)
from skyllh.core.dataset import Dataset, DatasetData
from skyllh.core.py import (
    classname,
)
from skyllh.core.random import RandomStateService
from skyllh.core.scrambling import (
    DataScrambler,
)
from skyllh.core.storage import DataFieldRecordArray


class FixedScrambledExpDataI3BkgGenMethod(
    BackgroundGenerationMethod,
):
    """This class implements the background event generation method for the
    IceCube detector using scrambled experimental data as background hypothesis
    with a fixed number of background events equal to the number of events in
    the dataset. This background generation method is the one used in SkyLab.
    """

    def __init__(
        self,
        data_scrambler: DataScrambler,
        keep_mask_func: Callable[[DataFieldRecordArray], np.ndarray] | None = None,
        **kwargs,
    ):
        """Creates a new background generation method instance to generate
        background events from scrambled experimental data with a fixed number
        of events equal to the number of events in the dataset.

        Parameters
        ----------
        data_scrambler
            The DataScrambler instance to use to generate scrambled experimental
            data.
        keep_mask_func
            The optional function, which takes the experimental data and
            returns the mask of the events, which should be scrambled and
            returned as background events. The other events are only counted
            as generated background events. This speeds up the background
            generation, but it must be used only if the masked events cannot
            be selected by the event selection of the analysis for any
            scrambling, e.g. events outside the declination band of the
            sources, when only the time and right-ascension are scrambled.
        """
        super().__init__(**kwargs)

        self.data_scrambler = data_scrambler
        self.keep_mask_func = keep_mask_func

        # The cache of the experimental data, the hash of the mask, and the
        # kept experimental data events for the last used experimental data.
        self._kept_exp_cache: tuple | None = None

    @property
    def data_scrambler(self):
        """The DataScrambler instance that implements the data scrambling."""
        return self._data_scrambler

    @data_scrambler.setter
    def data_scrambler(self, scrambler):
        if not isinstance(scrambler, DataScrambler):
            raise TypeError(
                'The data_scrambler property must be an instance of '
                'DataScrambler! '
                f'Its current type is {classname(scrambler)}!'
            )
        self._data_scrambler = scrambler

    def generate_events(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        rss: RandomStateService,
        dataset: Dataset,
        data: DatasetData,
        **kwargs,
    ) -> tuple[int, DataFieldRecordArray]:
        """Generates background events from the given data, by scrambling the
        experimental data. The number of events is equal to the size of the
        given dataset.

        Parameters
        ----------
        rss
            The instance of RandomStateService that should be used to generate
            random numbers from. It is used to scramble the experimental data.
        dataset
            The Dataset instance describing the dataset for which background
            events should get generated.
        data
            The DatasetData instance holding the data of the dataset for which
            background events should get generated.

        Returns
        -------
        n_bkg
            The number of generated background events.
        bkg_events
            The instance of DataFieldRecordArray holding the generated
            background events.
        """
        assert data.exp is not None

        if self.keep_mask_func is None:
            bkg_events = self._data_scrambler.scramble_data(rss=rss, dataset=dataset, data=data.exp, copy=True)
            return (len(bkg_events), bkg_events)

        # Scramble only the events, which can be selected by the analysis.
        mask = np.asarray(self.keep_mask_func(data.exp), dtype=np.bool_)
        mask_hash = hash(mask.tobytes())
        cache = self._kept_exp_cache
        if (cache is None) or (cache[0] is not data.exp) or (cache[1] != mask_hash):
            cache = (data.exp, mask_hash, data.exp[mask])
            self._kept_exp_cache = cache
        bkg_events = self._data_scrambler.scramble_data(rss=rss, dataset=dataset, data=cache[2], copy=True)

        return (len(data.exp), bkg_events)
