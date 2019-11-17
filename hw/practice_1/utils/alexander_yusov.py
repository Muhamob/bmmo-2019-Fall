import numpy as np
from scipy import signal
from scipy.signal import fftconvolve


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

    ll -= H*W*(0.5 * np.log(2 * np.pi) + np.log(s))
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

    # [dh, dw, k]
    log_p = calculate_log_probability(X, F, B, s) + np.expand_dims(np.log(A+1e-10), axis=-1)

    if not use_MAP:
        elbo = np.sum(q*log_p) - np.sum(np.log(q+1e-10) * q)
        return elbo
    else:
        K = X.shape[-1]
        elbo = 0

        for k in range(K):
            i, j = list(map(lambda x: int(x), q[:, k]))
            elbo += log_p[i, j, k]

        return elbo


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
    log_prior = np.expand_dims(np.log(A+1e-10), axis=-1)

    # shape (H-h+1, W-w+1, K)
    log_aposteriory_estimation = log_likelihood + log_prior
    max_across_images = np.max(log_aposteriory_estimation, axis=(0, 1), keepdims=True)
    log_aposteriory_exp = np.exp(log_aposteriory_estimation - max_across_images)

    q = log_aposteriory_exp / np.sum(log_aposteriory_exp, axis=(0, 1), keepdims=True)

    if not use_MAP:
        return q
    else:
        K = X.shape[-1]
        q_ = np.empty((2, K))
        for k in range(K):
            indices = np.unravel_index(np.argmax(q[:, :, k], axis=None), q.shape[:2])
            q_[:, k] = indices
        return q_


def update_A(q, use_MAP=False, dh=None, dw=None):
    """
    Make one step of updating matrix of priors on face location
    :param dh: H-h+1, used only with MAP-EM
    :param dw: W-w+1, used only with MAP-EM
    :param q: array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    :param use_map:
    :return: array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    if not use_MAP:
        num = np.sum(q, axis=-1)
        return num / np.sum(num)
    else:
        assert dh is not None and dw is not None, "Specify dh and dw"
        K = q.shape[-1]
        q_ = make_q_from_MAP(q, dh, dw, K)
        return np.mean(q_, axis=-1)


def correlate_across_axes(x1, x2, mode="valid", axes=(0, 1)):
    q_ = np.copy(x2)
    q_ = np.flip(q_, axis=axes[0])
    q_ = np.flip(q_, axis=axes[1])

    convolved = []

    for k in range(q_.shape[-1]):
        convolved.append(signal.fftconvolve(x1[:, :, k], q_[:, :, k], mode=mode))

    return np.asarray(convolved)

    # turns out, that older version of scipy doesn't have axes parameter in fftconvole
    # return signal.fftconvolve(x1, q_, mode=mode, axes=axes)


def update_F(q, X, use_map=False, h=None, w=None):
    den = X.shape[-1]

    if not use_map:
        f_ = correlate_across_axes(X, q, mode="valid", axes=(0, 1))
        f_ = np.sum(f_, axis=-1)

        return f_ / den
    else:
        assert h is not None and w is not None, "Specify h and w"

        face = 0
        for k in range(den):
            i, j = list(map(lambda x: int(x), q[:, k]))
            face += X[i:i+h, j:j+w, k]

        return face / den


def make_q_from_MAP(q, dh, dw, K):
    q_ = np.zeros((dh, dw, K))
    for k in range(K):
        i, j = list(map(lambda x: int(x), q[:, k]))
        q_[i, j, k] = 1

    return q_


def update_B(q, X, h, w, use_map=False):
    H, W, K = X.shape

    if use_map:
        dh = H - h + 1
        dw = W - w + 1
        q_ = make_q_from_MAP(q, dh, dw, K)
    else:
        q_ = q

    nq = 1 - fftconvolve(np.ones((h, w, 1)), q_, mode="full", axes=(0, 1))
    num_ = np.sum(X * nq, axis=-1)
    den = np.sum(nq, axis=-1)

    indices = np.where(den != 0)
    B = np.zeros_like(X[:, :, 0])
    B[indices] = num_[indices] / den[indices]
    return B


def update_s(q, X, F, B, h, w, use_map=False):
    H, W, K = X.shape
    den = X.size
    num = 0

    F_ = np.expand_dims(np.copy(F), axis=-1)

    if use_map:
        dh = H - h + 1
        dw = W - w + 1
        q_ = make_q_from_MAP(q, dh, dw, K)
    else:
        q_ = q

    for i in range(q_.shape[0]):
        for j in range(q_.shape[1]):
            q_kij = q_[i, j, :]

            B_ = np.expand_dims(np.copy(B), axis=-1)
            means = B_
            means[i:i+h, j:j+w, :] = F_

            num += np.sum(q_kij * np.sum((X - means) ** 2, axis=(0, 1)))

    return np.sqrt(num / den)


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
    H, W, K = X.shape
    dh = H - h + 1
    dw = W - w + 1

    A = update_A(q, use_MAP, dh, dw)
    F = update_F(q, X, use_MAP, h, w)
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
    H, W, K = X.shape

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

        # run every 2 times
        if i % 2 == 0:
            elbo_prev = elbo
            elbo = calculate_lower_bound(X, F, B, s, A, q, use_MAP)

        LL.append(elbo)
        i += 1
        print("iter", i, "out of", max_iter, "elbo normalized", elbo / K, "elbo diff", elbo - elbo_prev)

    return F, B, s, A, LL


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
    elbo_max = -np.inf
    params_max = None

    for i in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP)
        # q = run_e_step(X, F, B, s, A, use_MAP)
        # elbo = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        elbo = LL[-1]

        if elbo > elbo_max:
            elbo_max = elbo
            params_max = (F, B, s, A, LL)

    return params_max