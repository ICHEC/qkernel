from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import qutip
from qat.core import Observable, Term


class Hamiltonian(ABC):
    """
    Abstract base class for Hamiltonian construction.

    This class defines the interface for generating Hamiltonians
    from a list of qbits coordinates (qbits_list).
    """

    def __init__(
        self,
        qbits_list: List[np.ndarray],
        omega: float = 2 * np.pi,
        delta: float = 0,
        c6: float = 865723.02,
    ) -> None:
        """

        Initialize Hamiltonian base class.

        Parameters
        ----------
        qbits_list : list
            List of qbit configurations.
        omega : float, optional
            Driving term strength (default is 2π).
        delta : float, optional
            Detuning parameter (default is 0).
        c6 : float, optional
            Interaction coefficient (default is 865723.02).
        """
        self.qbits_list = qbits_list
        self.nqbits = len(qbits_list[0])
        self.omega = omega
        self.delta = delta
        self.c6 = c6

    @abstractmethod
    def generate_single_hamiltonian(self, qbits: np.ndarray) -> Any:
        """
        Generate Hamiltonian for a single qbit configuration.

        Parameters
        ----------
        qbits : array-like
            Coordinates or features of the qubits.

        Returns
        -------
        object
            Depends on subclass implementation.
        """
        pass

    def generate_hamiltonians_list(self) -> List[Any]:
        """
        Generate Hamiltonians for all qbit configurations.

        Returns
        -------
        list
            List of Hamiltonians, one per qbit configuration.
        """
        return [self.generate_single_hamiltonian(q) for q in self.qbits_list]


class MyQLMHamiltonian(Hamiltonian):
    """
    Quantum Hamiltonian implemented using QAT Observables.

    This class constructs a Hamiltonian composed of:

    - Amplitude (X-field) term
    - Detuning (Z-field occupation) term
    - Interaction term with 1/r^6 decay
    """

    def _occupation_operator(self, qi: int) -> Observable:
        """
        Construct occupation operator (n_i = (1 - Z_i)/2).

        Parameters
        ----------
        qi : int
            Qubit index.

        Returns
        -------
        Observable
            Occupation number operator for qubit i.
        """
        return (
            1
            - Observable(
                self.nqbits,
                pauli_terms=[Term(1.0, "Z", [qi])],
            )
        ) / 2

    def generate_single_hamiltonian(
        self, qbits: np.ndarray
    ) -> List[Tuple[float, Observable]]:
        """
        Build Hamiltonian for a single configuration using QAT Observables.

        Parameters
        ----------
        qbits : np.ndarray
            Spatial positions or feature vectors for qubits.

        Returns
        -------
        list of tuples
            Each tuple is (coefficient, Observable).
        """
        amplitude_term = Observable(
            self.nqbits,
            pauli_terms=[Term(0.5, "X", [i]) for i in range(self.nqbits)],
        )

        detuning_term = sum(self._occupation_operator(i) for i in range(self.nqbits))

        interaction_term = sum(
            (1 / np.linalg.norm(qbits[i] - qbits[j]) ** 6)
            * self._occupation_operator(i)
            * self._occupation_operator(j)
            for i in range(self.nqbits - 1)
            for j in range(i + 1, self.nqbits)
        )

        return [
            (self.omega, amplitude_term),
            (-self.delta, detuning_term),
            (self.c6, interaction_term),
        ]


class QuTiPHamiltonian(Hamiltonian):
    """
    Quantum Hamiltonian implemented using QuTiP operators.
    This version builds the Hamiltonian as a full tensor-product
    operator using QuTiP (for simulation purposes).
    """

    def _qutip_op(
        self,
        i: int,
        op_i: qutip.Qobj,
        j: int = None,
        op_j: qutip.Qobj = None,
    ) -> qutip.Qobj:
        """
        Construct a full system operator using tensor products.

        Parameters
        ----------
        i : int
            First qubit index.
        op_i : qutip.Qobj
            Operator acting on qubit i.
        j : int, optional
            Second qubit index (for interaction terms).
        op_j : qutip.Qobj, optional
            Operator acting on qubit j.
        ops = [qutip.qeye(2)] * self.nqbits

        Returns
        -------
        qutip.Qobj
            Tensor product operator on full Hilbert space.
        """
        ops = [qutip.qeye(2)] * self.nqbits
        ops[i] = op_i
        if j is not None:
            ops[j] = op_j
        return qutip.tensor(ops)


    def generate_single_hamiltonian(self, qbits: np.ndarray) -> qutip.Qobj:
        """
        Construct full QuTiP Hamiltonian for a single configuration.

        Parameters
        ----------
        qbits : array-like
            Spatial coordinates or features of qubits.

        Returns
        -------
        qutip.Qobj
            Full Hamiltonian operator.
        """
        H = sum(
            self.omega * 0.5 * self._qutip_op(i, qutip.sigmax())
            for i in range(self.nqbits)
        )

        H -= sum(
            self.delta * self._qutip_op(i, qutip.num(2)) for i in range(self.nqbits)
        )

        H += sum(
            (self.c6 / np.linalg.norm(qbits[i] - qbits[j]) ** 6)
            * self._qutip_op(i, qutip.num(2), j, qutip.num(2))
            for i in range(self.nqbits - 1)
            for j in range(i + 1, self.nqbits)
        )

        return H
