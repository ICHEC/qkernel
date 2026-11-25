import numpy as np

from .base import Encoding


class RadialEncoding(Encoding):
    """
    Encoder for transforming feature vectors into qubit coordinates
    using radial encoding.

    Notes
    -----
    If the feature vector is `(x1, x2, ..., xn)` (where n is the number of features),
    then each feature xi is mapped to a point on a circle:

        (xi -> ri cos(2π i/n), ri sin(2π i/n))

    where ri = (xi / max(x) + shift) * scaling.

    The encoding ensures geometric separation of qubits by distributing
    features uniformly on the unit circle with varying radii.
    """

    def __init__(self, bands_array: np.ndarray) -> None:
        """
        Initialise radial encoding.

        Parameters
        ----------
        bands_array : np.ndarray
            Input feature array (bands) of shape (n_samples, n_features).
        """
        self.bands_array = bands_array
        self.n_features = self.bands_array.shape[1]
        angles = np.linspace(0, 2 * np.pi, self.n_features, endpoint=False)
        self.unit_circle = np.array([np.cos(angles), np.sin(angles)]).T

    def encode(self, shift: float = 1.0, scaling: float = 5.4) -> np.ndarray:
        """
        Generate radial encoding coordinates.

        Parameters
        ----------
        shift : float
            Offset applied to feature values before scaling.
        scaling : float
            Scaling factor controlling radial spread of points.

        Returns
        -------
        np.ndarray
            Encoded coordinates of shape (n_samples, n_features, 2),
            where each feature is mapped to (x, y) on the unit circle.
        """
        max_feature = np.max(self.bands_array)
        radius = (self.bands_array / max_feature + shift) * scaling
        coords = radius[:, :, None] * self.unit_circle[None, :, :]
        return coords
