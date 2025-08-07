import numpy as np 
from scipy.linalg import eigh_tridiagonal
from numpy.random import default_rng
_rng = default_rng()

def jacobi_eigs(n,interval,rng=None):
    """
    Efficient eigenvalue computation for the tridiagonal matrix model (TMM) corresponding to the Jacobi ensemble.
    This algorithm  Build the explicit construction for the TMM and calls an O(dlog(d)) eigensolve via
    eigh_tridiagonal(diag, off, eigvals_only=True)
    
    See: 
    https://arxiv.org/abs/1601.01146 [pg 15]
    https://arxiv.org/abs/math/0410034 [pg 4]
    """
    if rng is None:
        rng = _rng
    a=interval[0]
    b=interval[1]
    m = n + 1
    p = np.zeros(2*m)
    for k in range(1, 2*m):
        if k % 2 == 0:
            alpha = (2*m - k) / 2
            beta = (2*m - k + 2) / 2
        else:
            alpha = beta = (2*m - k + 1) / 2
        p[k] = rng.beta(alpha, beta)
    diag = np.empty(m)
    off = np.empty(m-1)
    for k in range(1, m+1):
        p2k2 = p[2*k - 2]
        p2k3 = p[2*k - 3] if 2*k-3 >= 0 else 0.0
        p2k1 = p[2*k - 1]
        diag[k-1] = p2k2*(1 - p2k3) + p2k1*(1 - p2k2)
        if k < m:
            p2k = p[2*k]
            off[k-1] = np.sqrt(p2k1*(1 - p2k2) * p2k*(1 - p2k1))
    return (b-a)*eigh_tridiagonal(diag, off, eigvals_only=True) +a #Interval might be busted 


def gue_eigs(d):
    """
    Efficient eigenvalue computation for the tridiagonal matrix model (TMM) corresponding to the Gaussian unitary ensemble.
    This algorithm  Build the explicit construction for the TMM and calls an O(dlog(d)) eigensolve via
    eigh_tridiagonal(diag, off, eigvals_only=True)
    
    See: 
    https://arxiv.org/abs/1601.01146 [pg 10]
    https://arxiv.org/abs/math-ph/0206043 [pg 11]
    """
    n = d + 1
    diag = _rng.standard_normal(n)
    k = np.arange(1, n)               # [1,2,…,d]
    off = np.sqrt(_rng.chisquare(2 * k)) / np.sqrt(2.0)
    return eigh_tridiagonal(diag, off, lapack_driver="stev", eigvals_only=True)
    
