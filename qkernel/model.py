from abc import ABC, abstractmethod

import numpy as np
from sklearn.svm import SVC


class Model(ABC):
    """
    Abstract base class for machine learning models.

    Defines the required interface for training and prediction
    using kernel (Gram) matrices.
    """

    def __init__(self) -> None:
        """Initialise base model."""
        pass

    @abstractmethod
    def train(self, gram_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model using a training Gram matrix.

        Parameters
        ----------
        gram_train : np.ndarray
            Precomputed kernel (Gram) matrix for training.
        y_train : np.ndarray
            Training labels.
        """
        pass

    @abstractmethod
    def predict(self, gram_test: np.ndarray) -> np.ndarray:
        """
        Predict labels using a test Gram matrix.

        Parameters
        ----------
        gram_test : np.ndarray
            Precomputed kernel (Gram) matrix for test data.
        """
        pass


class SVM(Model):
    """
    Quantum-kernel-compatible Support Vector Machine wrapper.

    This class uses scikit-learn's SVC with a precomputed kernel,
    making it suitable for quantum kernel methods (QSVM-style pipelines).

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to sklearn.svm.SVC.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the SVM model with a precomputed kernel.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to sklearn's SVC.
        """
        self.model = SVC(kernel="precomputed", **kwargs)

    def train(self, gram_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model.

        Parameters
        ----------
        gram_train : np.ndarray
            Training Gram matrix.
        y_train : np.ndarray
            Training labels.
        """
        self.model.fit(gram_train, y_train)

    def get_accuracy(self, gram: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        gram : np.ndarray
            Gram matrix (train or test).
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        return self.model.score(gram, y)

    def predict(self, gram: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        gram : np.ndarray
            Gram matrix for prediction.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return self.model.predict(gram)
