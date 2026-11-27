from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Encoding(ABC):
    """
    Abstract base class for encoding data into qubit coordinates.

    An encoding holds the data of one dataset and transforms it into an array
    of qubit coordinates of shape (n_samples, n_qubits, 2).

    Attributes
    ----------
    labels : np.ndarray or None
        Labels of the encoded samples, for encodings which derive them from
        the data (set by `encode`). None if the encoding keeps one sample per
        input row, in which case the labels must be supplied separately.
    """

    labels: Optional[np.ndarray] = None

    @abstractmethod
    def encode(self, **kwargs) -> np.ndarray:
        """
        Encode the data into qubit coordinates.

        Parameters
        ----------
        **kwargs
            Parameters specific to each encoding.

        Returns
        -------
        np.ndarray
            Qubit coordinates of shape (n_samples, n_qubits, 2).
        """
        pass
