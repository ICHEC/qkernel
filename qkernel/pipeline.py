import json
import os
import time
from typing import Optional, Type, Union

import numpy as np
import pandas as pd

from .encodings import Encoding
from .hamiltonian import MyQLMHamiltonian, QuTiPHamiltonian
from .kernel import Kernel
from .model import SVM
from .simulator import MyQLMSimulator, QuTiPSimulator

HAMILTONIANS_DICT = {
    "myqlm": MyQLMHamiltonian,
    "qutip": QuTiPHamiltonian,
}

SIMULATORS_DICT = {
    "myqlm": MyQLMSimulator,
    "qutip": QuTiPSimulator,
}

MODELS_DICT = {"svc": SVM}


class Pipeline:
    """
    End-to-end quantum machine learning pipeline.

    This class orchestrates the full workflow:
    Data encoding → Hamiltonian construction → quantum simulation →
    kernel computation → classical model training → evaluation and saving results.

    A single encoding is supplied, which the pipeline applies to both the train
    and test data to turn them into qubit coordinates.
    """

    def __init__(
        self,
        encoding: Type[Encoding],
        data_train: Union[np.ndarray, pd.DataFrame],
        data_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        args: dict,
    ):
        """
        Initialize pipeline with an encoding, data and configuration arguments.

        Parameters
        ----------
        encoding : type
            Encoding class (a subclass of `Encoding`) applied to both datasets.
        data_train : np.ndarray or pd.DataFrame
            Training data, of the type expected by the encoding.
        data_test : np.ndarray or pd.DataFrame
            Test data, of the type expected by the encoding.
        y_train : np.ndarray or None
            Training labels. If None, the labels produced by the encoding
            are used.
        y_test : np.ndarray or None
            Test labels. If None, the labels produced by the encoding
            are used.
        args : dict
            Dictionary containing all pipeline configuration parameters.
        """
        self.encoding_cls = encoding
        self.data_train = data_train
        self.data_test = data_test
        self.y_train = y_train
        self.y_test = y_test
        self.args = args

    def run(self):
        """
        Execute the full pipeline end-to-end.

        Steps:
        1. Encode data into qubit coordinates
        2. Construct Hamiltonians
        3. Run quantum/classical simulation
        4. Compute kernel matrices
        5. Train and evaluate model
        6. Save results
        """
        t1 = time.time()
        self.compute_encoding()
        self.hamiltonian_cls = HAMILTONIANS_DICT[self.args["hamiltonian"]["backend"]]
        self.compute_hamiltonian()
        self.simulator_cls = SIMULATORS_DICT[self.args["simulator"]["backend"]]
        self.run_simulation()
        self.compute_kernel()
        self.model_cls = MODELS_DICT[self.args["model"]["type"]]
        self.run_model()
        t2 = time.time()
        self.time = t2 - t1
        self.save_results(self.args["results"]["path"], self.args["results"]["name"])

    def compute_encoding(self):
        """
        Encode train and test data into qubit coordinates.

        Labels which were not supplied are taken from the encodings.

        Raises
        ------
        ValueError
            If there are no labels for a dataset, or their number differs from
            the number of encoded samples.
        """
        kwargs = (self.args.get("encoding") or {}).get("kwargs") or {}

        for dataset_type in ["train", "test"]:
            encoding = self.encoding_cls(getattr(self, f"data_{dataset_type}"))
            qbits = encoding.encode(**kwargs)

            labels = getattr(self, f"y_{dataset_type}")
            if labels is None:
                labels = encoding.labels
            if labels is None:
                raise ValueError(
                    f"y_{dataset_type} is required: "
                    f"{self.encoding_cls.__name__} does not produce labels."
                )
            if len(labels) != len(qbits):
                raise ValueError(
                    f"Got {len(labels)} {dataset_type} labels for "
                    f"{len(qbits)} encoded samples."
                )

            setattr(self, f"x_{dataset_type}", qbits)
            setattr(self, f"y_{dataset_type}", labels)

    def compute_hamiltonian(self):
        """
        Construct Hamiltonians from encoded qubit coordinates.

        Converts qbit representations into Hamiltonian objects
        for both train and test datasets.
        """
        for dataset_type in ["train", "test"]:
            qbits = getattr(self, f"x_{dataset_type}")
            hamiltonian = self.hamiltonian_cls(
                qbits, **self.args["hamiltonian"].get("kwargs") or {}
            )
            setattr(
                self,
                f"hamiltonians_{dataset_type}",
                hamiltonian.generate_hamiltonians_list(),
            )

    def run_simulation(self):
        """
        Run quantum or classical simulations on Hamiltonians.

        Produces probability distributions for each Hamiltonian
        in both train and test sets.
        """
        for dataset_type in ["train", "test"]:
            hamiltonian_list = getattr(self, f"hamiltonians_{dataset_type}")
            simulator = self.simulator_cls(
                hamiltonian_list, **self.args["simulator"].get("kwargs") or {}
            )
            setattr(
                self,
                f"probabilities_{dataset_type}",
                simulator.get_probabilities_list(),
            )

    def compute_kernel(self):
        """
        Compute kernel (Gram) matrices from probability distributions.

        Generates:
        - Training Gram matrix
        - Test Gram matrix
        """
        kernel = Kernel(
            self.probabilities_train,
            self.probabilities_test,
            **self.args["kernel"].get("kwargs") or {},
        )
        self.gram_train = kernel.compute_gram_train()
        self.gram_test = kernel.compute_gram_test()

    def run_model(self):
        """
        Train classical model and evaluate performance.

        Computes:
        - Training accuracy
        - Test accuracy
        - Predicted labels on test set
        """
        model = self.model_cls(**self.args["model"].get("kwargs"))
        model.train(self.gram_train, self.y_train)
        self.train_accuracy = model.get_accuracy(self.gram_train, self.y_train)
        self.test_accuracy = model.get_accuracy(self.gram_test, self.y_test)
        self.predicted_labels = model.predict(self.gram_test)

    def save_results(self, path: str, name: str) -> None:
        """
        Save pipeline results to a JSON file.

        Parameters
        ----------
        path : str
            Path to the output JSON file.
        name : str
            Name to give to the results file.
        """
        output = {
            "args": self.args,
            "results": {
                "train_accuracy": self.train_accuracy,
                "test_accuracy": self.test_accuracy,
                "predicted_labels": self.predicted_labels.tolist(),
                "time": self.time,
            },
        }

        filename = f"{name}.json"
        full_path = os.path.join(path, filename)
        with open(full_path, "w") as f:
            json.dump(output, f, indent=4)
