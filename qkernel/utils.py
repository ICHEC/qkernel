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
