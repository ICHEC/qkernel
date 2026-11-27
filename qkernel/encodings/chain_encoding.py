import numpy as np

from ..utils import normalise_array
from .base import Encoding


class ChainEncoding(Encoding):
    """
    Class for implementing the chain encoding.

    This encoding maps input band features into a 2D spatial representation
    suitable for quantum device constraints.

    Parameters
    ----------
    bands_array : np.ndarray
        Numpy array containing band values to be encoded.
    """

    def __init__(self, bands_array: np.ndarray) -> None:
        """
        Initialise chain encoding.

        Parameters
        ----------
        bands_array : np.ndarray
            Input feature array representing bands.
        """
        self.bands_array_normalised = normalise_array(bands_array)

    def encode(self, c1: float = 36, c2: float = 10) -> np.ndarray:
        """
        Perform chain encoding of normalised band data.

        Parameters
        ----------
        c1 : float
            Offset constant. Must satisfy |c1| ≤ 38.
        c2 : float
            Scaling constant. Must satisfy |-c1 + n_bands * c2| ≤ 38.

        Returns
        -------
        np.ndarray
            Encoded array of shape (n_samples, n_bands, 2),
            where each point corresponds to (x_position, band_value).
        """
        x = -c1 + c2 * np.arange(self.bands_array_normalised.shape[1])

        return np.stack(
            (
                np.broadcast_to(x, self.bands_array_normalised.shape),
                self.bands_array_normalised,
            ),
            axis=2,
        )
