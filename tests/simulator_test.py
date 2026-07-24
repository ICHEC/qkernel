import numpy as np
import pytest
from qkernel4eo.embeddings.chain_embedding import ChainEmbedding
from qkernel4eo.embeddings.radial_embedding import RadialEmbedding

from qkernel.hamiltonian import QuTiPHamiltonian
from qkernel.simulator import QuTiPSimulator, Simulator

RNG = np.random.default_rng(42)
# 3 features: satisfies chain embedding constraint |-c_1 + n*c_2| = |-36+30| = 6 <= 38
# 2 samples: normalise_array needs at least 2 rows along axis=0 to avoid 0/0 = NaN
N_QBITS = 3
BANDS = RNG.random((2, N_QBITS))

QBITS_LIST = np.array(
    [
        RadialEmbedding(BANDS).embed()[0],  # shape (N_QBITS, 2)
        ChainEmbedding(BANDS).embed()[0],  # shape (N_QBITS, 2)
    ]
)  # stacked to shape (N, N_QBITS, 2) since Hamiltonian.__init__ reads qbits.shape[1]

HAMILTONIANS_LIST = QuTiPHamiltonian(QBITS_LIST).generate_hamiltonians_list()


@pytest.fixture
def simulator():
    return QuTiPSimulator(HAMILTONIANS_LIST)


# --- Simulator (abstract base) ---


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        Simulator(HAMILTONIANS_LIST)


# --- QuTiPSimulator ---


def test_evolve_hamiltonian_returns_array(simulator):
    result = simulator._evolve_hamiltonian(HAMILTONIANS_LIST[0])
    assert isinstance(result, np.ndarray)


def test_evolve_hamiltonian_shape(simulator):
    result = simulator._evolve_hamiltonian(HAMILTONIANS_LIST[0])
    assert result.flatten().shape == (2**N_QBITS,)


def test_get_probabilities_list_length(simulator):
    result = simulator.get_probabilities_list()
    assert len(result) == len(HAMILTONIANS_LIST)


def test_probabilities_are_nonnegative(simulator):
    for probs in simulator.get_probabilities_list():
        assert np.all(probs >= 0)


def test_probabilities_sum_to_one(simulator):
    for probs in simulator.get_probabilities_list():
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)
