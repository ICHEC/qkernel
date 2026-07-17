import json
import os
import time

import numpy as np

from .kernel import Kernel
from .hamiltonian import MyQLMHamiltonian, QuTiPHamiltonian
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
    Hamiltonian construction → quantum simulation → kernel computation →
    classical model training → evaluation and saving results.

    Qbit arrays and labels are supplied directly rather than loaded/embedded
    from a dataset.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        args: dict
    ):
        """
        Initialize pipeline with configuration arguments and qbit data.

        Parameters
        ----------
        x_train : np.ndarray
            Training qbit array.
        x_test : np.ndarray
            Test qbit array.
        y_train : np.ndarray
            Training labels.
        y_test : np.ndarray
            Test labels.
        args : dict
            Dictionary containing all pipeline configuration parameters.
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.args = args
        
    def run(self):
        """
        Execute the full pipeline end-to-end.

        Steps:
        1. Construct Hamiltonians
        2. Run quantum/classical simulation
        3. Compute kernel matrices
        4. Train and evaluate model
        5. Save results
        """
        t1 = time.time()
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

    def compute_hamiltonian(self):
        """
        Construct Hamiltonians from embedded quantum states.

        Converts qbit representations into Hamiltonian objects
        for both train and test datasets.
        """
        for dataset_type in ["train", "test"]:
            qbits = getattr(self, f"qbits_{dataset_type}")
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
