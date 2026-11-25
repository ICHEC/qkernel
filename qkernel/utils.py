import collections as col
import random as rd
from typing import List, Union

import numpy as np


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
