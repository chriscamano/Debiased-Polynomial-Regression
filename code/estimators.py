import numpy as np
from numpy.random import default_rng
from math import factorial
from numpy.linalg import lstsq

from numpy.polynomial.legendre import legvander, Legendre
from numpy.polynomial.hermite_e import hermevander, HermiteE
from numpy.polynomial import Polynomial

from scipy.special import eval_hermitenorm
from scipy.stats import unitary_group

_rng = default_rng()

from random_eigs import jacobi_eigs, gue_eigs
from util import *
#=========== Polynomial Regression ===========

def sample_lev_gauss(n, d):
    out = np.empty(n, dtype=float)
    for i in range(n):
        eigs   = gue_eigs(d)
        out[i] = eigs[_rng.integers(0, d + 1)]
    return out
    
def sample_lev_unif(n, d,interval):
    out = np.empty(n)
    for i in range(n):
        vals = jacobi_eigs(d,interval)
        out[i] = vals[_rng.integers(0, d+1)]
    return out
    
def gauss_leverage_function(ts, d):
    ts = np.asarray(ts, dtype=float).ravel()
    H  = np.vstack([eval_hermitenorm(k, ts) for k in range(d + 1)])
    norms = np.array([factorial(k) for k in range(d + 1)], dtype=float)
    levs  = (H**2 / norms[:, None]).sum(axis=0)
    return levs
 
def unif_leverage_function(times, d):
    ks   = np.arange(d+1)
    V    = legvander(times, d)            
    norms = (2*ks + 1) / 2              
    return (norms * V**2).sum(axis=1)

def debiased_regression(f, d, n, interval, measure="uniform", x_eval=None):
    """
    [Algorithm 1] Debiased active polynomial regression 
    Computes an unbiased polynomial approximation of the optimal polynomial approximation for a given function
    using a collection of d+1 (real) eigenvalues from a complex hermitian random matrix and n-d+1 leverage score
    samples from a target measure.
    """
    m = n - (d + 1)
    if measure == "uniform":
        lam        = jacobi_eigs(d, interval)
        tail       = sample_lev_unif(m, d, interval) if m > 0 else np.empty(0)
        lev_func   = unif_leverage_function
        basis_matrix = legendre_orthonormal_matrix
        basis      = "legendre"

    elif measure == "gaussian":
        lam        = gue_eigs(d)
        tail       = sample_lev_gauss(m, d) if m > 0 else np.empty(0)
        lev_func   = gauss_leverage_function
        basis_matrix = hermite_orthonormal_matrix
        basis      = "hermite"

    else:
        raise ValueError("measure must be 'uniform' or 'gaussian'")
        
    t   = np.concatenate([lam, tail])
    y   = np.asarray([f(tt) for tt in t], dtype=float)
    lev = lev_func(t, d)
    S   = np.diag(1 / np.sqrt(lev))

    M, scale_vec = basis_matrix(t, d)
    coeffs       = lstsq(S @ M, S @ y, rcond=None)[0]
    poly = build_polynomial(coeffs, basis, scale_vec)
    p_eval = None if x_eval is None else poly(x_eval)

    return {
        "poly":    poly,
        "coeffs":  coeffs,
        "basis":   basis,
        "times":   t,
        "p_eval":  p_eval
    }


def leverage_score_regression(f, d, n, interval, measure="uniform", x_eval=None):
    """
    **Biased** active polynomial regression 
    Computes an biased polynomial approximation of the optimal polynomial approximation for a given function
    using a collection of n leverage score samples from a target measure.
    """
    if measure == "uniform":
        t       = sample_lev_unif(n, d, interval) 
        lev_func   = unif_leverage_function
        basis_matrix = legendre_orthonormal_matrix
        basis      = "legendre"

    elif measure == "gaussian":
        t       = sample_lev_gauss(n, d) 
        lev_func   = gauss_leverage_function
        basis_matrix = hermite_orthonormal_matrix
        basis      = "hermite"

    else:
        raise ValueError("measure must be 'uniform' or 'gaussian'")
    
    y   = np.asarray([f(tt) for tt in t], dtype=float)
    lev = lev_func(t, d)
    S   = np.diag(1 / np.sqrt(lev))

    M, scale_vec = basis_matrix(t, d)
    coeffs       = lstsq(S @ M, S @ y, rcond=None)[0]
    poly = build_polynomial(coeffs, basis, scale_vec)
    p_eval = None if x_eval is None else poly(x_eval)

    return {
        "poly":    poly,
        "coeffs":  coeffs,
        "basis":   basis,
        "times":   t,
        "p_eval":  p_eval
    }
    

#=========== Fourier Regression ===========

def debiased_fourier_regression(f, d, n,z_grid=None,rng=None):
    """
    [Algorithm 4] Debiased active Fourier regression 
    Computes an unbiased approximation of the optimal polynomial approximation for a given periodic function
    using a collection of d+1 (real) eigenvalues from from random (Harr) orthogonal matrices and n-d+1 leverage score
    (uniform) samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    X  = unitary_group.rvs(d+1)
    z0 = np.linalg.eigvals(X)

    m = n - (d + 1)
    if m > 0:
        θ      = rng.uniform(0, 2*np.pi, size=m)
        z_tail = np.exp(1j * θ)
    else:
        z_tail = np.empty(0, dtype=complex)

    z = np.concatenate([z0, z_tail])
    y = np.asarray([f(zi) for zi in z], dtype=complex)

    V      = np.vander(z, N=d+1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)
    poly   = np.poly1d(coeffs[::-1])
    p_eval = None if z_grid is None else poly(z_grid)

    return {
        "poly":    poly,
        "coeffs":  coeffs,
        "basis":   "fourier",
        "times":   z,
        "p_eval":  p_eval
    }


def leverage_score_fourier_regression(f, d, n,z_grid=None, rng=None):
    """
    [Algorithm 4] Biased active Fourier regression 
    Computes an biased approximation of the optimal polynomial approximation for a given periodic function
    using a collection of n leverage score (uniform) samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    θ = rng.uniform(0, 2*np.pi, size=n)
    z = np.exp(1j * θ)
    y = np.asarray([f(zi) for zi in z], dtype=complex)
    V      = np.vander(z, N=d+1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)

    poly   = np.poly1d(coeffs[::-1])
    p_eval = None if z_grid is None else poly(z_grid)

    return {
        "poly":    poly,
        "coeffs":  coeffs,
        "basis":   "fourier",
        "times":   z,
        "p_eval":  p_eval
    }