import sys
import time
import numpy as np
from nose.tools import ok_, eq_
from numpy.testing import assert_almost_equal
from . import alexander_yusov as model


def test_python3():
    ok_(sys.version_info > (3, 3))


def check_shape(**kwargs):
    np.random.seed(42)
    H, W, K = 10, 12, 20
    h, w = 4, 5
    max_iter = 1
    X = np.random.rand(H, W, K)
    F, B, s, A, LL = model.run_EM(X, h, w, max_iter=max_iter, **kwargs)
    eq_(F.shape, (h, w))
    eq_(B.shape, (H, W))
    eq_(A.shape, (H-h+1, W-w+1))
    eq_(np.array(LL).shape, (max_iter,))


def generate_data(H, W, K, h, w, use_MAP=False, seed=42):
    # explicit data with no noise
    np.random.seed(seed)
    X = np.zeros((H, W, K))
    F = np.zeros((h, w))
    B = np.random.rand(H, W)

    coords = []
    q = np.zeros((H-h+1, W-w+1, K))
    for k in range(K):
        x = np.random.randint(0, H-h+1)
        y = np.random.randint(0, W-w+1)
        coords.append((x, y))
        X[:, :, k] = np.copy(B)
        X[x:x+h, y:y+w, k] = F
        q[x, y, k] = 1.

    A = np.random.rand(H - h + 1, W - w + 1)
    A /= A.sum()

    if use_MAP:
        q = np.array(coords).T

    return X, F, B, A, q


def check_e_step(use_MAP=False):
    H, W, K = 4, 5, 2
    h, w = 2, 3
    s = 1e-1
    X, F, B, A, q = \
        generate_data(H, W, K, h, w, use_MAP=use_MAP)

    pred_q = model.run_e_step(X, F, B, s, A, use_MAP=use_MAP)
    if use_MAP:
        assert_almost_equal(q, pred_q)
    else:
        assert_almost_equal(q, pred_q, 5)


def check_m_step(use_MAP=False):
    H, W, K = 7, 8, 2
    h, w = 2, 3
    X, F, B, A, q = generate_data(H, W, K, h, w, use_MAP=use_MAP)

    pred_F, pred_B, pred_s, pred_A = \
        model.run_m_step(X, q, h, w, use_MAP=use_MAP)

    assert_almost_equal(F, pred_F)
    assert_almost_equal(B, pred_B)


def check_e_step_time(use_MAP=False):
    H, W, K = 50, 100, 50
    h, w = 40, 50
    s = 0.1
    X, F, B, A, q = generate_data(H, W, K, h, w)
    t_start = time.perf_counter()
    model.run_e_step(X, F, B, s, A, use_MAP=use_MAP)
    computation_time = time.perf_counter() - t_start
    assert computation_time < 1


def check_m_step_time(use_MAP=False):
    H, W, K = 50, 100, 50
    h, w = 40, 50
    X, F, B, A, q = generate_data(H, W, K, h, w, use_MAP=use_MAP)

    t_start = time.perf_counter()
    model.run_m_step(X, q, h, w, use_MAP=use_MAP)
    computation_time = time.perf_counter() - t_start
    assert computation_time < 1


def test_output_shape():
    check_shape()
    check_shape(use_MAP=True)


def test_e_step():
    check_e_step()
    check_e_step(use_MAP=True)


def test_m_stap():
    check_m_step()
    check_m_step(use_MAP=True)


def test_e_step_time():
    check_e_step_time()
    check_e_step_time(use_MAP=True)


def test_m_step_time():
    check_m_step_time()
    check_m_step_time(use_MAP=True)
