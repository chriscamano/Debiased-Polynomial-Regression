from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np 
from tqdm import tqdm 

from util import best_polynomial_approximation,true_fourier_coeffs
from estimators import *

from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from scipy.special import roots_legendre,roots_hermitenorm

#================= Polynomial Experiments =================

#----------------- Helpers -----------------
def eval_on_grid(f, grid):
    grid = np.asarray(grid, dtype=float)
    return f(grid)

def run_trials(f, d, n, grid, R,interval, estimator,measure):
    G = grid.size
    mean = np.zeros(G, dtype=float)
    M2   = np.zeros(G, dtype=float)  
    # Streaming mean and std for large number of runs (std breaks on large matrices)
    for r in  tqdm(range(1, R+1)):
        out   = estimator(f, d, n,interval, measure,x_eval=grid)
        y     = out['p_eval']        
        delta = y - mean
        mean += delta / r
        M2   += delta * (y - mean)     

    variance = M2 / (R - 1)           
    std      = np.sqrt(variance)
    return mean, std
    
def weighted_l2_error(f, poly, measure, interval, quad_deg):
    if measure == "uniform":
        x0, w0 = roots_legendre(quad_deg)
        a, b = interval
        x = 0.5*(b - a)*x0 + 0.5*(a + b)
        w = w0 * (b - a) / 2
    else: 
        x, w = roots_hermitenorm(quad_deg)
        w = w / np.sqrt(2 * np.pi)
    
    y_true = f(x)
    y_pred = poly(x)
    return np.sqrt(np.dot(w, (y_true - y_pred)**2))
    
#----------------- Experiment 1: Regression -----------------
def run_experiment1(f, d, n, R, interval=(-1.0, 1.0), step=1e-4,
                    quad_deg_proj=1024,measure="uniform"):
    a, b  = interval
    grid  = np.arange(a, b + step/2, step)
    true_y = eval_on_grid(f, grid)

    if measure == "uniform": 
        pstar,c   = best_polynomial_approximation(f, d, quad_deg=quad_deg_proj,basis="legendre",interval=interval)
    else: #Gaussian measure
        pstar,c   = best_polynomial_approximation(f, d, quad_deg=quad_deg_proj,basis="hermite",interval=interval)

    best_y  = pstar(grid)
    
    mean_unb, std_unb = run_trials(f, d, n, grid, R, interval,debiased_regression,measure)
    mean_bia, std_bia = run_trials(f, d, n, grid, R, interval,leverage_score_regression,measure)
    
    return dict(
        grid=grid,
        true_y=true_y,
        best_y=best_y,
        mean_unb=mean_unb, std_unb=std_unb,
        mean_bia=mean_bia, std_bia=std_bia,
        interval=interval
    )

#----------------- Experiment 4: Epsilon vs n   -----------------
def run_experiment4(
    f,
    d_values=(5, 10, 15, 20),
    n_runs=100,
    interval=(-1, 1),
    measure="uniform",
    quad_deg_proj=1024,
    quad_deg_err=1024,
    verbose=False
):
    if measure not in ("uniform", "gaussian"):
        raise ValueError("measure must be 'uniform' or 'gaussian'")
    
    base_n_values = np.arange(3, 63, 3)  
    
    results = {}
    for d in d_values:
        # filter out n < d
        n_values = base_n_values[base_n_values >d]
        m = len(n_values)
        
        eps_deb = np.zeros((m, n_runs))
        eps_lev = np.zeros((m, n_runs))
        
        basis      = "legendre" if measure=="uniform" else "hermite"
        poly_star, _ = best_polynomial_approximation(
            f, d, quad_deg=quad_deg_proj, basis=basis, interval=interval
        )
        E_star    = weighted_l2_error(
            f, poly_star, measure, interval, quad_deg=quad_deg_err
        )
        if verbose:
            print(f"\n d={d:2d}, baseline ‖f−p*‖₂ = {E_star:.3e}")
        
        for i, n in enumerate(tqdm(n_values, desc=f"d={d} samples")):
            for r in range(n_runs):
                out = debiased_regression(f, d, n, interval, measure, x_eval=None)
                E_hat = weighted_l2_error(
                    f, out["poly"], measure, interval, quad_deg=quad_deg_err
                )
                eps_deb[i, r] = E_hat  / E_star - 1.0
                
                out = leverage_score_regression(f, d, n, interval, measure, x_eval=None)
                E_hat = weighted_l2_error(
                    f, out["poly"], measure, interval, quad_deg=quad_deg_err
                )
                eps_lev[i, r] = E_hat  / E_star - 1.0
            
            if verbose and r < 3:
                print(f"   n={n:2d}: ε_debiased={eps_deb[i,0]:.3e}, ε_leverage={eps_lev[i,0]:.3e}")
        
        results[d] = {
            "n_values":     n_values,
            "eps_debiased": eps_deb,
            "eps_leverage": eps_lev
        }
    
    return {
        "d_values": list(d_values),
        "results":   results
    }

#================= Fourier Experiments =================

#----------------- Helpers -----------------
def run_trials_real(f, d, n, z_grid, R, estimator):
    G = z_grid.size
    mean = np.zeros(G, dtype=float)
    M2   = np.zeros(G, dtype=float)

    for r in tqdm(range(1, R+1)):
        out = estimator(f, d, n, z_grid=z_grid)
        y   = out['p_eval']     
        y_real = np.real(y)      

        delta = y_real - mean
        mean += delta / r
        M2   += delta * (y_real - mean)

    variance = M2 / (R - 1)
    std      = np.sqrt(variance)
    return mean, std
    
def evaluate_polynomial_on_circle(coeffs, z_grid):
    result = np.zeros_like(z_grid, dtype=complex)
    for k, c_k in enumerate(coeffs):
        result += c_k * (z_grid ** k)
    return result  
    
def run_experiment2(f, d, n, R,
                    M_fft=4096,
                    measure="uniform",
                    N_grid=2001):
    
    theta = np.linspace(0, 2*np.pi, N_grid, endpoint=False)
    z_grid = np.exp(1j * theta)
    true_y = np.real(f(z_grid))
    true_coefs = true_fourier_coeffs(f, d, M=M_fft)
    pstar_vals = evaluate_polynomial_on_circle(true_coefs, z_grid)
    best_y = np.real(pstar_vals)

    mean_unb, std_unb = run_trials_real(
        f, d, n, z_grid, R,
        estimator=debiased_fourier_regression,
    )
    mean_bia, std_bia = run_trials_real(
        f, d, n, z_grid, R,
        estimator=leverage_score_fourier_regression,
    )

    rmse_unb = np.sqrt(np.mean((best_y - mean_unb)**2))
    rmse_bia = np.sqrt(np.mean((best_y - mean_bia)**2)) #RMSE for now 

    return {
        "theta":    theta,
        "grid":   z_grid,
        "true_y":   true_y,
        "best_y":   best_y,
        "mean_unb": mean_unb,
        "std_unb":  std_unb,
        "mean_bia": mean_bia,
        "std_bia":  std_bia,
        "rmse_unb": rmse_unb,
        "rmse_bia": rmse_bia,
        "n":        n
    }

def run_experiment4_fourier(
    f,
    d_values=(5, 10, 15, 20),
    n_runs=100,
    measure="uniform",
    M_fft=4096,
    N_grid=2001,
    base_n_values=None,
    verbose=False
):
    if measure not in ("uniform", "gaussian"):
        raise ValueError("measure must be 'uniform' or 'gaussian'")
    
    if base_n_values is None:
        base_n_values = np.arange(3, 63, 3)
    
    theta   = np.linspace(0, 2*np.pi, N_grid, endpoint=False)
    z_grid  = np.exp(1j * theta)
    
    results = {}
    for d in d_values:
        n_values = base_n_values[base_n_values > d]
        m = len(n_values)
        
        eps_deb = np.zeros((m, n_runs))
        eps_lev = np.zeros((m, n_runs))
        
        true_coefs = true_fourier_coeffs(f, d, M=M_fft)
        pstar_vals = evaluate_polynomial_on_circle(true_coefs, z_grid)
        E_star = np.linalg.norm(f(z_grid) - pstar_vals) / np.sqrt(N_grid)
        if verbose:
            print(f"\n d={d:2d}, baseline ‖f−p*‖₂ = {E_star:.3e}")
        
        for i, n in enumerate(tqdm(n_values, desc=f"d={d} samples")):
            for r in range(n_runs):
                out_deb = debiased_fourier_regression(
                    f, d, n, z_grid
                )
                E_hat_deb = np.linalg.norm(f(z_grid) - out_deb["poly"](z_grid)) / np.sqrt(N_grid)
                eps_deb[i, r] = E_hat_deb / E_star - 1.0
                
                out_lev = leverage_score_fourier_regression(
                    f, d, n, z_grid
                )
                E_hat_lev = np.linalg.norm(f(z_grid) - out_lev["poly"](z_grid)) / np.sqrt(N_grid)
                eps_lev[i, r] = E_hat_lev / E_star - 1.0
            
            if verbose and r < 3:
                print(f"   n={n:2d}: ε_debiased={eps_deb[i,0]:.3e}, ε_leverage={eps_lev[i,0]:.3e}")
        
        results[d] = {
            "n_values":     n_values,
            "eps_debiased": eps_deb,
            "eps_leverage": eps_lev
        }
    
    return {
        "d_values": list(d_values),
        "results":   results
    }
