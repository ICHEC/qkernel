import numpy as np
import pandas as pd
import skimage as ski
from scipy.stats import rankdata

from ..utils import majority_vote, normalise_array
from .base import Encoding


class SpatialPatchEncoding(Encoding):
    """
    Spatial patch encoding for transforming spatial + spectral data into
    graph-like structures for quantum analog processing.

    Neighbouring points are grouped into non-overlapping square patches, and each
    patch forms one graph.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialise spatial patch encoding.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing at least:
            ['label', 'latitude', 'longitude', 'b02', 'b03', 'b04'].
            The points must lie on a square grid. The input is not modified.
        """
        self.df = df.copy()
        self.colour_transformation()
        self.rank_sort_coordinates()

    def colour_transformation(self) -> None:
        """
        Convert RGB spectral bands into HSV color space representation.
        """
        rgb_coordinates = normalise_array(self.df[["b04", "b03", "b02"]].to_numpy())
        ## Colour transformation. We add a new dimension to fit in the skimage input.
        hsv_coordinates = ski.color.rgb2hsv(rgb_coordinates[np.newaxis, :, :])[0]
        self.df["hsv_coordinates"] = list(hsv_coordinates)

    def rank_sort_coordinates(self) -> None:
        """
        Rank and sort spatial coordinates into a discrete grid. The
        elements on the dataset will be ordered with values depending
        on latitude and longitude.
        """
        self.df["lat_rank"] = rankdata(self.df["latitude"], method="dense") - 1
        self.df["lon_rank"] = rankdata(self.df["longitude"], method="dense") - 1
        self.df.sort_values(by=["lat_rank", "lon_rank"], inplace=True)

    def split_into_patches(self, patch_side: int = 2) -> None:
        """
        Splits the data points into square patches.

        Parameters
        ----------
        patch_side : int
            Number of points on each side of the patch (default is 2).
        """
        n_grid_side = int(max(self.df["lat_rank"]) + 1)
        patch_coordinates = []
        patch_labels = []

        for i in range(0, n_grid_side, patch_side):
            for j in range(0, n_grid_side, patch_side):
                mask = (
                    (self.df["lat_rank"] >= i)
                    & (self.df["lat_rank"] < i + patch_side)
                    & (self.df["lon_rank"] >= j)
                    & (self.df["lon_rank"] < j + patch_side)
                )

                patch_coordinates.append(
                    np.stack(self.df["hsv_coordinates"][mask].to_numpy())
                )

                patch_labels.append(
                    majority_vote(np.stack(self.df["label"][mask].to_numpy()))
                )

        self.patch_coordinates = np.stack(patch_coordinates)
        self.labels = np.array(patch_labels)

    def encode(
        self,
        patch_side: int = 2,
        scaling: float = 37.0,
    ) -> np.ndarray:
        """
        Computes the coordinates of the nodes of each of the graphs in the dataset.
        It does so by assigning one angle to each of the nodes. This angle
        will be the HUE angle plus an offset factor so that our graphs are embeddable
        in the analog device. Then we use those angles to change from polar
        to cartesian coordinates.

        The label of each graph, decided by majority vote among its points,
        is stored in the `labels` attribute.

        Parameters
        ----------
        patch_side : int
            Number of points on each side of the patch (default is 2).
        scaling : float
            Radial coordinate to apply to our coordinates. (default is 37).

        Returns
        -------
        np.ndarray
            Graph coordinates of shape (n_graphs, patch_side**2, 2).
        """
        self.split_into_patches(patch_side=patch_side)
        _, M, _ = self.patch_coordinates.shape

        unit_circle_division = (2 * np.pi) / (patch_side**2 * 2)

        angles_normalised = np.mod(
            self.patch_coordinates[:, :, 0], unit_circle_division
        )

        offsets = 2 * np.arange(patch_side**2) * unit_circle_division
        offsets = offsets[:M]

        angles_divided = angles_normalised + offsets

        x = scaling * np.cos(angles_divided)
        y = scaling * np.sin(angles_divided)

        return np.stack((x, y), axis=2)
