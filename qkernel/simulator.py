from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import qkernel
import qutip
from qat.core import Batch, Schedule
from qlmaas.qpus import AnalogQPU


class Simulator(ABC):
    """
    Abstract base class for quantum simulators.

    This class defines a common interface for running a list of Hamiltonians
    and extracting probability distributions from their evolution.
    """

    def __init__(
        self,
        hamiltonians_list: List[qkernel.hamiltonian.Hamiltonian],
        duration: float = 0.66,
    ) -> None:
        """
        Initialise simulator.

        Parameters
        ----------
        hamiltonians_list : list
            List of Hamiltonian objects.
        duration : float, optional
            Evolution time for simulation (default is 0.66).
        """
        self.hamiltonians_list = hamiltonians_list
        self.duration = duration

    @abstractmethod
    def get_probabilities_list(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run simulation and return probability distributions.

        Returns
        -------
        list or np.ndarray
            Probability distributions for each Hamiltonian.
        """
        pass


class MyQLMSimulator(Simulator):
    """
    Simulator using Qaptiva / QLM Analog Quantum Processing Unit.

    This class converts Hamiltonians into QLM schedules and executes them
    on an AnalogQPU backend.
    """

    def __init__(
        self,
        hamiltonians_list: List[qkernel.hamiltonian.Hamiltonian],
        duration: float = 0.66,
    ) -> None:
        """
        Initialize QLM simulator.

        Parameters
        ----------
        hamiltonians_list : list
            List of QLM-compatible Hamiltonians (Observable-based).
        duration : float, optional
            Evolution time for analog simulation.
        """
        self.hamiltonians_list = hamiltonians_list
        self.my_qpu = AnalogQPU()
        self.duration = duration

    def _create_jobs(self) -> None:
        """
        Convert Hamiltonians into QLM jobs (Schedules → Jobs).
        """
        self.jobs = [
            Schedule(drive=h, tmax=self.duration).to_job()
            for h in self.hamiltonians_list
        ]

    def _run_jobs(self) -> None:
        """
        Submit jobs to the QPU and collect results.
        """
        async_result = self.my_qpu.submit(Batch(self.jobs))
        self.results = async_result.join()

    def get_probabilities_list(self) -> np.ndarray:
        """
        Execute all Hamiltonians and return probability distributions.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_states) containing probabilities.
        """
        self._create_jobs()
        self._run_jobs()

        return np.array([[r.probability for r in result] for result in self.results])


class QuTiPSimulator:
    """
    Classical simulator using QuTiP for time evolution.

    This class evolves quantum states under a given Hamiltonian
    using Schrödinger equation integration.
    """

    def __init__(
        self,
        hamiltonians_list: List[qutip.Qobj],
        duration: float = 0.66,
    ) -> None:
        """
        Initialize QuTiP simulator.

        Parameters
        ----------
        hamiltonians_list : list
            List of QuTiP Qobj Hamiltonians.
        duration : float
            Evolution time.
        """
        self.hamiltonians_list = hamiltonians_list
        self.duration = duration

    def _evolve_hamiltonian(self, hamiltonian: qutip.Qobj) -> np.ndarray:
        """
        Evolve a single Hamiltonian using QuTiP solver.

        Parameters
        ----------
        hamiltonian : qutip.Qobj
            Hamiltonian operator.

        Returns
        -------
        np.ndarray
            Final state vector as dense array.
        """
        initial_state = qutip.tensor([qutip.basis(2)] * len(hamiltonian.dims[0]))

        evolved_state = qutip.sesolve(
            hamiltonian,
            initial_state,
            [0.0, self.duration],
        )

        return evolved_state.final_state.full()

    def get_probabilities_list(self) -> List[np.ndarray]:
        """
        Compute probability distributions for all Hamiltonians.

        Returns
        -------
        list of np.ndarray
            List of probability distributions per Hamiltonian.
        """
        probabilities_list = []

        for hamiltonian in self.hamiltonians_list:
            final_state = self._evolve_hamiltonian(hamiltonian)
            probabilities_list.append(np.real(np.conj(final_state) * final_state))

        return probabilities_list
