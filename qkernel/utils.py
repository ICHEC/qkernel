import collections as col
import random as rd
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist


def compute_excitation_count(probs: np.ndarray) -> np.ndarray:
    """
    Convert probability distributions into excitation-count representations.

    Each probability vector is transformed by grouping probabilities according
    to the number of excitations (i.e., number of '1' bits in the binary
    representation of basis states.

    Parameters
    ----------
    probs : np.ndarray
        Array of shape (n_samples, 2^n_features), where each row is a
        probability distribution over computational basis states.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_features + 1), where each column
        corresponds to the total probability mass of states with a given
        number of excitations.
    """
    num_features = int(np.log2(probs.shape[1]))
    excitations = np.zeros((probs.shape[0], num_features + 1))

    for i in range(2**num_features):
        n_excit = np.binary_repr(i, num_features).count("1")
        excitations[:, n_excit] += probs[:, i]
    return excitations


def majority_vote(x: Union[np.ndarray, List[int]], seed: int = 24041916) -> int:
    """
    Finds the most common value of an array | list. If there is a tie,
    the value will be decided randomly.

    Parameters
    ----------
    x : np.ndarray | list
        An array of integers.
    seed : int
        Random seed used to randomly break ties.

    Returns
    -------
    int
        The most common value of the list
    """
    rd.seed(seed)
    counts = col.Counter(x)
    max_count = max(counts.values())
    ties = [k for k, v in counts.items() if v == max_count]
    most_common_value = rd.choice(ties)
    return int(most_common_value)


def normalise_array(x: np.ndarray, n: int = 1, axis: int = 0) -> np.ndarray:
    """
    Normalises array between 0 and n along a given dimension.

    Parameters
    ----------
    x : np.ndarray
        An array of floats.
    n : int
        The maximum of the normalised array.
    axis : int
        The axis along which do the normalisation.

    Returns
    -------
    np.ndarray
        The normalised array
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_min = np.min(x, axis=axis, keepdims=True)
    x_normalised = ((x - x_min) / (x_max - x_min)) * n
    return x_normalised


def plot_label_grid_with_points(
    df: pd.DataFrame,
    label_col: str = "label",
    lat_col: str = "lat_rank",
    lon_col: str = "lon_rank",
    label_names: Optional[List[str]] = None,
    cmap_name: str = "tab10",
    point_color: str = "black",
    point_size: int = 20,
    figsize: Tuple[float, float] = (8, 8),
    legend_loc: str = "upper left",
    draw_squares: bool = False,
    n_convoluted_side: int = 2,
) -> None:
    """
    Plots a categorical label grid with overlaid scatter points.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing label, latitude rank and longitude rank columns,
        sorted by latitude rank and then longitude rank. The points must lie
        on a square grid, as in the `df` attribute of `ConvolutionalEncoding`.
    label_col : str
        Name of the column containing label values, which must start at 1.
    lat_col : str
        Name of the column containing latitude ranks.
    lon_col : str
        Name of the column containing longitude ranks.
    label_names : list of str
        Names of the categories, in the order of numeric labels.
    cmap_name : str
        Name of the matplotlib colormap to use for categories.
    point_color : str
        Color of the scatter points.
    point_size : int
        Size of the scatter points.
    figsize : tuple
        Figure size.
    legend_loc : str
        Location of the legend.
    draw_squares : bool
        Whether to outline the blocks of points used by the convolution.
    n_convoluted_side : int
        Number of points on each side of the outlined blocks.
    """
    # Prepare the label grid
    num_points_per_side = int(np.sqrt(len(df)))  # assumes square grid
    label_grid = (
        df[label_col].to_numpy().reshape(num_points_per_side, num_points_per_side)
    )

    # Map labels to colors
    cmap = plt.get_cmap(cmap_name)
    num_labels = int(df[label_col].max())  # assumes labels start at 1
    colors = [cmap(i) for i in range(num_labels)]
    color_indices = label_grid - 1  # map label 1->0, 2->1, etc.

    # Plot the grid
    plt.figure(figsize=figsize)
    plt.imshow(
        color_indices,
        origin="lower",
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=num_labels - 0.5,
    )
    plt.xlabel("Longitude Rank")
    plt.ylabel("Latitude Rank")
    plt.title("Label Grid")

    # Add legend
    if label_names is None:
        label_names = [f"Label {i + 1}" for i in range(num_labels)]
    legend_handles = [
        mpatches.Patch(color=colors[i], label=label_names[i]) for i in range(num_labels)
    ]
    plt.legend(handles=legend_handles, loc=legend_loc)

    # Scatter points
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
    plt.scatter(lon, lat, color=point_color, s=point_size, edgecolor="white")

    if draw_squares:
        # Outline each block of the grid
        for i in range(0, num_points_per_side, n_convoluted_side):
            for j in range(0, num_points_per_side, n_convoluted_side):
                rect = mpatches.Rectangle(
                    (j, i),  # bottom-left corner
                    n_convoluted_side - 1,
                    n_convoluted_side - 1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                )
                plt.gca().add_patch(rect)

    plt.show()


def check_embeddability(
    graph_list: Union[np.ndarray, List[np.ndarray]],
    max_atoms: int = 80,
    max_distance_from_origin: float = 38.0,
    min_distance_between_atoms: float = 5.0,
) -> Dict[int, List[int]]:
    """
    Check whether a list of atom position graphs are embeddable on the AnalogQPU.

    Parameters
    ----------
    graph_list : np.ndarray or list of np.ndarray
        List of N graphs, each of shape (n_atoms, 2) representing atom
        coordinates in µm.
    max_atoms : int
        Maximum number of atoms allowed in a single register.
    max_distance_from_origin : float
        Maximum allowed distance (µm) of any atom from the origin.
    min_distance_between_atoms : float
        Minimum allowed distance (µm) between any pair of atoms.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping graph index to its violated constraints:
        0 (too many atoms), 1 (too far from the origin) and 2 (atoms too close).
        An empty dict means all graphs are embeddable.
    """
    failed_clauses = {}
    for i, graph in enumerate(graph_list):
        violations = []
        # 1. Maximum atom number
        if len(graph) > max_atoms:
            violations.append(0)
        # 2. Distance from origin
        if np.any(np.linalg.norm(graph, axis=1) > max_distance_from_origin):
            violations.append(1)
        # 3. Distance between atoms
        if np.any(pdist(graph, metric="euclidean") < min_distance_between_atoms):
            violations.append(2)
        if violations:
            failed_clauses[i] = violations
    return failed_clauses
