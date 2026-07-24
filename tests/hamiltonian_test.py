import numpy as np
import pytest
import qutip
from qat.core import Observable
from qkernel4eo.embeddings.chain_embedding import ChainEmbedding
from qkernel4eo.embeddings.radial_embedding import RadialEmbedding

from qkernel.hamiltonian import Hamiltonian, MyQLMHamiltonian, QuTiPHamiltonian

RNG = np.random.default_rng(42)
# 3 features: satisfies chain embedding constraint |-c_1 + n*c_2| = |-36+30| = 6 <= 38
# 2 samples: normalise_array needs at least 2 rows along axis=0 to avoid 0/0 = NaN
N_QBITS = 3
BANDS = RNG.random((2, N_QBITS))

QBITS_LIST = [
    RadialEmbedding(BANDS).embed()[0],  # shape (N_QBITS, 2)
    ChainEmbedding(BANDS).embed()[0],  # shape (N_QBITS, 2)
]


@pytest.fixture
def myqlm():
    return MyQLMHamiltonian(QBITS_LIST)


@pytest.fixture
def qutip_h():
    return QuTiPHamiltonian(QBITS_LIST)


# --- Hamiltonian (abstract base) ---


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        Hamiltonian(QBITS_LIST)


def test_nqbits_set_correctly(myqlm):
    assert myqlm.nqbits == N_QBITS


def test_default_parameters(myqlm):
    assert myqlm.omega == 2 * np.pi
    assert myqlm.delta == 0
    assert myqlm.c6 == 865723.02


def test_generate_hamiltonians_list_length(myqlm):
    result = myqlm.generate_hamiltonians_list()
    assert len(result) == len(QBITS_LIST)


# --- MyQLMHamiltonian ---


def test_myqlm_returns_three_terms(myqlm):
    result = myqlm.generate_hamiltonian(QBITS_LIST[0])
    assert len(result) == 3


def test_myqlm_coefficients(myqlm):
    result = myqlm.generate_hamiltonian(QBITS_LIST[0])
    assert result[0][0] == myqlm.omega
    assert result[1][0] == -myqlm.delta
    assert result[2][0] == myqlm.c6


def test_myqlm_terms_are_observables(myqlm):
    result = myqlm.generate_hamiltonian(QBITS_LIST[0])
    for _, obs in result:
        assert isinstance(obs, Observable)


def test_myqlm_custom_parameters():
    h = MyQLMHamiltonian(QBITS_LIST, omega=1.0, delta=0.5, c6=100.0)
    result = h.generate_hamiltonian(QBITS_LIST[0])
    assert result[0][0] == 1.0
    assert result[1][0] == -0.5
    assert result[2][0] == 100.0


# --- QuTiPHamiltonian ---


def test_qutip_returns_qobj(qutip_h):
    result = qutip_h.generate_hamiltonian(QBITS_LIST[0])
    assert isinstance(result, qutip.Qobj)


def test_qutip_hamiltonian_shape(qutip_h):
    result = qutip_h.generate_hamiltonian(QBITS_LIST[0])
    dim = 2**N_QBITS
    assert result.shape == (dim, dim)


def test_qutip_hamiltonian_is_hermitian(qutip_h):
    result = qutip_h.generate_hamiltonian(QBITS_LIST[0])
    assert result.isherm


def test_qutip_op_single_qubit_shape(qutip_h):
    op = qutip_h._qutip_op(0, qutip.sigmax())
    assert op.shape == (2**N_QBITS, 2**N_QBITS)


def test_qutip_op_two_qubit_shape(qutip_h):
    op = qutip_h._qutip_op(0, qutip.num(2), 1, qutip.num(2))
    assert op.shape == (2**N_QBITS, 2**N_QBITS)


def test_qutip_generate_hamiltonians_list_length(qutip_h):
    result = qutip_h.generate_hamiltonians_list()
    assert len(result) == len(QBITS_LIST)
