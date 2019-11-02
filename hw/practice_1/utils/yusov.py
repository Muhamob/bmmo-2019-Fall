import math

import numpy as np
from scipy.signal import fftconvolve, correlate


def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, K = X.shape
    h, w = F.shape
    dh = H - h + 1
    dw = W - w + 1

    ll = np.empty([dh, dw, K])

    for i in range(dh):
        for j in range(dw):
            B_ = np.copy(B)  # copy is necessary
            B_[i:i + h, j:j + w] = F
            under_exponent = + (-1. / (2 * s ** 2)) * (X - np.expand_dims(B_, axis=-1)) ** 2
            ll[i, j] = np.sum(under_exponent, axis=(0, 1))

    ll -= H*W*(0.5 * math.log(2 * math.pi) + math.log(s))
    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : np.array, shape (H, W, K)
        K images of size H x W.
    F : np.array, shape (h, w)
        Estimate of villain's face.
    B : np.array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : np.array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : np.array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    elbo = np.sum(q*(calculate_log_probability(X, F, B, s)+np.log(A)))

    if not use_MAP:
        return elbo
    else:
        raise NotImplementedError()


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    log_likelihood = calculate_log_probability(X, F, B, s)
    log_prior = np.expand_dims(np.log(A), axis=-1)

    # shape (H-h+1, W-w+1, K)
    log_aposteriory_estimation = log_likelihood + log_prior
    max_across_images = np.max(log_aposteriory_estimation, axis=(0, 1), keepdims=True)
    log_aposteriory_exp = np.exp(log_aposteriory_estimation - max_across_images)

    q = log_aposteriory_exp / np.sum(log_aposteriory_exp, axis=(0, 1), keepdims=True)

    if not use_MAP:
        return q
    else:
        raise NotImplementedError()


def update_A(q, use_map=False):
    if not use_map:
        num = np.sum(q, axis=-1)
        return num / np.sum(num)
    else:
        raise NotImplementedError("Update A with use_map set True still no implemented")


def update_F(q, X, use_map=False):
    den = np.sum(q)

    f_k_list = []
    for k in range(X.shape[-1]):
        # q_k(d_ij)
        x_k = X[:, :, k]
        q_k = q[:, :, k]
        f_k_list.append(correlate(x_k, q_k, mode="valid", method="fft"))

    f_ = sum(f_k_list)

    if not use_map:
        return f_ / den
    else:
        raise NotImplementedError("Update face matrix with use_map set True still no implemented")


def update_B(q, X, h, w, use_map=False):
    b_num = 0
    den = 0

    for k in range(q.shape[2]):
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                q_kij = q[i, j, k]
                # update numerator
                X_ = np.copy(X)
                x_k = X_[:, :, k]
                x_k[i:i+h, j:j+w] = 0
                b_num += q_kij * x_k

                # update denominator
                mask = np.ones_like(x_k)
                mask[i:i+h, j:j+w] = 0
                den += q_kij*mask

    if not use_map:
        return b_num / den
    else:
        raise NotImplementedError("Update background matrix with use_map set True still no implemented")


def update_s(q, X, F, B, h, w, use_map=False):
    den = X.size
    num = 0

    for k in range(q.shape[2]):
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                q_kij = q[i, j, k]
                x_k = X[:, :, k]
                means = np.copy(B)
                means[i:i+h, j:j+w] = np.copy(F)
                num += q_kij * np.sum((x_k - means) ** 2)

    if not use_map:
        return math.sqrt(num / den)
    else:
        raise NotImplementedError("Update s^2 matrix with use_map set True still no implemented")


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F, B, s, A given estimate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    A = update_A(q, use_MAP)
    F = update_F(q, X, use_MAP)
    B = update_B(q, X, h, w, use_MAP)
    s = update_s(q, X, F, B, h, w, use_MAP)

    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    x_max = np.max(X)
    H, W = X.shape[0], X.shape[1]

    F = x_max*np.abs(np.random.randn(h, w)) if F is None else F
    B = x_max*np.abs(np.random.randn(H, W)) if B is None else B
    s = np.std(X[:, :, 0]) if s is None else s
    A = np.random.rand(H-h+1, W-w+1) if A is None else A
    A = A / np.sum(A)
    q = run_e_step(X, F, B, s, A, use_MAP)

    i = 0

    elbo = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
    elbo_prev = -np.inf
    LL = []

    while elbo - elbo_prev > tolerance and i < max_iter:
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        LL.append((q, F, B, s, A))
        elbo_prev = elbo
        elbo = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        print("Iteration number " + i + ", ELBO: " + elbo)

    print("Training ended")
    return LL


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    pass
