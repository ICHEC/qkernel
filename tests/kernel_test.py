import numpy as np
import pytest

from qkernel.kernel import (
    Kernel,
    exponential_distance,
    exponential_jensen_shannon_distance,
)

RNG = np.random.default_rng(42)
N_TRAIN = 4
N_TEST = 2
N_STATES = 8  # 3 qubits → 2^3 states
N_QUBITS = 3


def make_probs(n, rng):
    x = rng.random((n, N_STATES))
    return x / x.sum(axis=1, keepdims=True)


P_TRAIN = make_probs(N_TRAIN, RNG)
P_TEST = make_probs(N_TEST, RNG)


@pytest.fixture
def kernel():
    return Kernel(P_TRAIN, P_TEST, excitations=False, distance_fn="exp_js")


# --- exponential_jensen_shannon_distance ---


def test_exp_js_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(exponential_jensen_shannon_distance(p, p), 1.0)


def test_exp_js_different_distributions():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    result = exponential_jensen_shannon_distance(p, q)
    assert 0 < result < 1


def test_exp_js_mu_scaling():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    r1 = exponential_jensen_shannon_distance(p, q, mu=1)
    r2 = exponential_jensen_shannon_distance(p, q, mu=2)
    assert r2 < r1


# --- exponential_distance ---


def test_exp_dist_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(exponential_distance(p, p), 1.0)


def test_exp_dist_known_value():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    np.testing.assert_allclose(exponential_distance(p, q), np.exp(-2.0))


def test_exp_dist_mu_scaling():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    r1 = exponential_distance(p, q, mu=1)
    r2 = exponential_distance(p, q, mu=2)
    assert r2 < r1


# --- Kernel.__init__ ---


def test_kernel_stores_distributions_without_excitations():
    k = Kernel(P_TRAIN, P_TEST, excitations=False, distance_fn="exp_js")
    np.testing.assert_array_equal(k.p_train, P_TRAIN)
    np.testing.assert_array_equal(k.p_test, P_TEST)


def test_kernel_transforms_with_excitations():
    k = Kernel(P_TRAIN, P_TEST, excitations=True, distance_fn="exp_js")
    assert k.p_train.shape == (N_TRAIN, N_QUBITS + 1)
    assert k.p_test.shape == (N_TEST, N_QUBITS + 1)


def test_kernel_distance_kwargs_defaults_to_empty(kernel):
    assert kernel.distance_kwargs == {}


def test_kernel_n_train_n_test(kernel):
    assert kernel.n_train == N_TRAIN
    assert kernel.n_test == N_TEST


def test_kernel_exp_distance_fn():
    k = Kernel(P_TRAIN, P_TEST, excitations=False, distance_fn="exp")
    gram = k.compute_gram_train()
    assert gram.shape == (N_TRAIN, N_TRAIN)


# --- compute_gram_train ---


def test_gram_train_shape(kernel):
    assert kernel.compute_gram_train().shape == (N_TRAIN, N_TRAIN)


def test_gram_train_diagonal_is_ones(kernel):
    gram = kernel.compute_gram_train()
    np.testing.assert_allclose(np.diag(gram), 1.0)


def test_gram_train_is_symmetric(kernel):
    gram = kernel.compute_gram_train()
    np.testing.assert_allclose(gram, gram.T)


def test_gram_train_values_in_range(kernel):
    gram = kernel.compute_gram_train()
    assert np.all(gram > 0) and np.all(gram <= 1)


# --- compute_gram_test ---


def test_gram_test_shape(kernel):
    assert kernel.compute_gram_test().shape == (N_TEST, N_TRAIN)


def test_gram_test_values_in_range(kernel):
    gram = kernel.compute_gram_test()
    assert np.all(gram > 0) and np.all(gram <= 1)


def test_gram_test_custom_mu():
    k1 = Kernel(
        P_TRAIN,
        P_TEST,
        excitations=False,
        distance_fn="exp_js",
        distance_kwargs={"mu": 1},
    )
    k2 = Kernel(
        P_TRAIN,
        P_TEST,
        excitations=False,
        distance_fn="exp_js",
        distance_kwargs={"mu": 10},
    )
    gram1 = k1.compute_gram_test()
    gram2 = k2.compute_gram_test()
    assert gram2.mean() < gram1.mean()
