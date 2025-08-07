from math import factorial
import numpy as np
from numpy.linalg import lstsq
from numpy.fft import fft
from numpy.polynomial.legendre import leggauss, legvander,Legendre
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.hermite_e import herme2poly, hermevander,hermefit, HermiteE
from numpy.polynomial.polynomial import Polynomial

from scipy.special import roots_hermitenorm, eval_hermitenorm,roots_legendre

#=========== Polynomial Utilities ===========

def legendre_orthonormal_matrix(t, d):
    V = legvander(t, d)            
    ks = np.arange(d + 1)
    sc = np.sqrt(2.0/(2*ks + 1))    
    return V / sc, sc

def hermite_orthonormal_matrix(t, d): 
    H = np.vstack([eval_hermitenorm(k, t) for k in range(d+1)]).T
    scale_vec = np.ones(d+1)
    return H, scale_vec

def vandermonde(t, cols):
    return np.vander(t, N=cols, increasing=True)
    
def build_polynomial(coeffs, basis, scale_vec):
    if basis == "monomial":
        return Polynomial(coeffs)

    elif basis == "hermite":
        c_herm = coeffs / scale_vec
        return HermiteE(c_herm)

    elif basis == "legendre":
        c_leg = coeffs / scale_vec
        return Legendre(c_leg)
    else:
        raise ValueError("basis must be 'monomial', 'hermite', or 'legendre'")

def best_polynomial_approximation(f, d, quad_deg=256,
                                  basis="hermite",
                                  interval=(-1.0, 1.0)):    
    if basis == "legendre":
        a, b = interval
        x0, w0 = roots_legendre(quad_deg)
        x = 0.5*(b - a)*x0 + 0.5*(a + b)
        w = w0 * (b - a) / 2
        y = f(x)
    
        leg = Legendre.fit(x, y, deg=d, domain=interval, w=np.sqrt(w), rcond=None)
        c_star = leg.coef
        
        #Orthonormalize
        ks = np.arange(len(c_star))
        c_star_orth = c_star * np.sqrt((2*ks + 1) / (b - a))
        
        return leg, c_star_orth
        
    elif basis == "hermite":
        x, w = roots_hermitenorm(quad_deg)
        y = f(x)
        c_h = hermefit(x, y, deg=d, w=np.sqrt(w))
        her = HermiteE(c_h)

        #Orthonormalize
        ks = np.arange(len(c_h))
        c_star_orth = c_h * np.sqrt([float(factorial(k)) for k in ks])
        
        return her, c_star_orth
        
    elif basis == "monomial":
        a, b = interval
        x0, w0 = roots_legendre(quad_deg)
        x = 0.5*(b - a)*x0 + 0.5*(a + b)
        w = w0 * (b - a) / 2
        y = f(x)
        poly = Polynomial.fit(x, y, deg=d, domain=interval, w=np.sqrt(w), rcond=None)
        return poly, poly.coef
    else:
        raise ValueError("basis must be 'hermite', 'legendre', or 'monomial'")

#=========== Fourier Utilities ===========

def true_fourier_coeffs(f, d, M=4096):
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    z_fine = np.exp(1j * theta)
    
    f_vals = f(z_fine)
    fft_coeffs = np.fft.fft(f_vals) / M

    coeffs = np.zeros(d + 1, dtype=complex)
    coeffs[0] = fft_coeffs[0]  # c_0
    
    for k in range(1, min(d + 1, M // 2 + 1)): #Nysquist hack 
        coeffs[k] = fft_coeffs[k]  
    
    return coeffs