from typing import Dict, Optional

import numpy as np
from qkernel.utils import compute_excitation_count
from scipy.spatial import distance


def exponential_jensen_shannon_distance(
    pi: np.ndarray,
    pj: np.ndarray,
    mu: float = 1,
) -> float:
    """
    Exponential Jensen–Shannon divergence (square of JS distance)
    converted into a similarity score.
    """
    return np.exp(-mu * distance.jensenshannon(pi, pj) ** 2)


def exponential_distance(
    pi: np.ndarray,
    pj: np.ndarray,
    mu: float = 1,
) -> float:
    """
    Exponential L1-based similarity between two probability distributions.
    """
    return np.exp(-mu * np.abs(pi - pj).sum())


DISTANCES_DICT = {
    "exp_js": exponential_jensen_shannon_distance,
    "exp": exponential_distance,
}


class Kernel:
    """
    Kernel matrix computation based on a distance-induced similarity function.

    This class constructs Gram (kernel) matrices between training and test
    probability distributions. Optionally, the input probabilities are
    transformed into excitation-count representations before computing the kernel.
    """

    def __init__(
        self,
        p_train: np.ndarray,
        p_test: np.ndarray,
        excitations: bool = True,
        distance_fn: str = DISTANCES_DICT["exp_js"],
        distance_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialise kernel computation module.

        Parameters
        ----------
        p_train : np.ndarray
            Training probability distributions of shape (n_train, 2^n_features).
        p_test : np.ndarray
            Test probability distributions of shape (n_test, 2^n_features).
        excitations : bool, optional
            If True, transforms probability vectors into excitation-count
            representations before computing kernels.
        distance_fn : callable, optional
            Function that computes similarity between two probability vectors.
            Expected signature: distance_fn(pi, pj, **kwargs) -> float.
        distance_kwargs : dict or None, optional
            Additional keyword arguments passed to the distance function.
        """
        self.distance_fn = DISTANCES_DICT[distance_fn]
        self.distance_kwargs = distance_kwargs or {}

        if excitations:
            self.p_train = compute_excitation_count(p_train)
            self.p_test = compute_excitation_count(p_test)
        else:
            self.p_train = p_train
            self.p_test = p_test

        self.n_train = len(self.p_train)
        self.n_test = len(self.p_test)

    def compute_gram_train(self) -> np.ndarray:
        """
        Compute the Gram (kernel) matrix for the training set.

        Returns
        -------
        np.ndarray
            Symmetric Gram matrix of shape (n_train, n_train) where
            entry (i, j) corresponds to the kernel value between
            training samples i and j.
        """
        gram_train = np.ones((self.n_train, self.n_train))

        for i in range(self.n_train - 1):
            for j in range(i + 1, self.n_train):
                gram_train[i, j] = gram_train[j, i] = self.distance_fn(
                    self.p_train[i],
                    self.p_train[j],
                    **self.distance_kwargs,
                )

        return gram_train

    def compute_gram_test(self) -> np.ndarray:
        """
        Compute the Gram (kernel) matrix between test and training sets.


        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_test, n_train) where entry (i, j)
            corresponds to the kernel value between test sample i and
            training sample j.
        """
        gram_test: np.ndarray = np.ones((self.n_test, self.n_train))

        for i in range(self.n_test):
            for j in range(self.n_train):
                gram_test[i, j] = self.distance_fn(
                    self.p_test[i],
                    self.p_train[j],
                    **self.distance_kwargs,
                )

        return gram_test
