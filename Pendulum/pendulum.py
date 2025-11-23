"""
This is the code for the Pendulum Project in PHY324

Jack Cheng 1010266695
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PLOT_RAW_DATA = True

def quadratic(t, a, b, c):
    return a*t**2 + b*t + c

def refine_peak(i, time, theta, window, sigma_theta=None):
    """Refine peak around index i with quadratic fit, propagate uncertainty"""
    left = max(0, i-window)
    right = min(len(time), i+window+1)
    t_slice = time[left:right]
    th_slice = theta[left:right]

    # Fit quadratic
    if sigma_theta is not None:
        sigma = np.full_like(th_slice, sigma_theta)
        p_opt, p_cov = curve_fit(quadratic, t_slice, th_slice,
                                 sigma=sigma, absolute_sigma=True)
    else:
        p_opt, p_cov = curve_fit(quadratic, t_slice, th_slice)
    a, b, c = p_opt
    Sigma = p_cov

    # Vertex
    t_max = -b / (2*a)
    theta_max = c - b**2/(4*a)

    # Gradients for error propagation
    dt_da = b / (2*a**2)
    dt_db = -1/(2*a)
    dt_dc = 0
    J_t = np.array([dt_da, dt_db, dt_dc])

    dth_da = (b**2)/(4*a**2)
    dth_db = -b/(2*a)
    dth_dc = 1
    J_th = np.array([dth_da, dth_db, dth_dc])

    # Variances
    var_t = J_t @ Sigma @ J_t.T
    var_th = J_th @ Sigma @ J_th.T

    return t_max, theta_max, np.sqrt(var_t), np.sqrt(var_th)


def main():
    print("Hello welcome to Pendulum project")

    # Load all data files
    # short
    # medium
    # Long
    print("Loading Data...")
    Langle_Mlen_time, Langle_Mlen_theta = np.loadtxt('new_data/Langle_Mlen.txt', delimiter=',', skiprows=2, unpack=True, usecols=(0,1))
    Mangle_Mlen_time, Mangle_Mlen_theta = np.loadtxt('new_data/Mangle_Mlen.txt', delimiter=',', skiprows=2, unpack=True, usecols=(0,1))
    Mangle_Llen_time, Mangle_Llen_theta = np.loadtxt('new_data/Mangle_Llen.txt', delimiter=',', skiprows=2, unpack=True, usecols=(0,1))
    Mangle_Slen_time, Mangle_Slen_theta = np.loadtxt('new_data/Mangle_Slen.txt', delimiter=',', skiprows=2, unpack=True, usecols=(0,1))
    Sangle_Mlen_time, Sangle_Mlen_theta = np.loadtxt('new_data/Sangle_Mlen.txt', delimiter=',', skiprows=2, unpack=True, usecols=(0,1))

    titles = ["Large angle Medium Length", "Medium angle Medium length", "Medium angle Long length", "Medium angle Short length", "Small angle Medium length"]
    times = [Langle_Mlen_time, Mangle_Mlen_time, Mangle_Llen_time, Mangle_Slen_time, Sangle_Mlen_time]
    thetas = [Langle_Mlen_theta, Mangle_Mlen_theta, Mangle_Llen_theta, Mangle_Slen_theta, Sangle_Mlen_theta]

    if PLOT_RAW_DATA:
        for t, th, title in zip(times, thetas, titles):
            plt.figure()
            plt.plot(t, th)
            plt.xlabel("Time (s)")
            plt.ylabel("Theta (rad)")
            plt.title(title)
        plt.show()


    # Now let's do stuff

    # Q1: Verify or refute the claim that the period T is independent of amplitude θ0. 
    # You can do this for a single length and mass combination. Make sure you get to large angles (close to π/2).
    #   Provide a quantitative estimate of the asymmetry of your pendulum. 
    #   You can do this as part of the work in the previous bullet point.

    '''
    Plan.
    Only for a single length, so just use the 3 M_len files.

    1) Determine the asymetrry of each.
    2) Plot all 3 on the same axis just to see what's up.
    3) Use check in 1 logic to check if T independent of amplitude? (I think no)

    '''


    # Q2: Verify or refute the claim that the decay is exponential, and determine the time constant τ . You can do this for a single length.
    '''
    Plan.
    Only for a single length, so just use the 3 M_len files.

    1) find peaks.
    2) fit the peaks to an exponential curve
    3) comment on the goodness of fit, to determine if this decay is well described by an exponential.
    '''

    # Q3: Verify or refute the claim that the period depends on L as stated: T = 2(L + D)1/2 (provided that D2 ≪ L2).
    '''
    Use the 3 M_angle files

    1) Calculate average time period of each of the 3 files
    2) Plot
    3) Comment on fit.
    
    '''

    # Q4: investigate the effect of L + D and θ0 on τ . 
    # If you find a trend, attempt to find a function which fits your data. 
    # You do not need to have a theoretical justification for your function; you do need to justify your choice in terms of goodness of fit in some fashion. 
    # As no prediction was given for this, there is nothing to verify or refute. Since you lack a prediction, you should stick with basic fit functions: 
    # linear functions, power series, power-laws, exponential functions.
    '''
    idek man...
    '''


    print("what the heck is this.")
    


if __name__ == "__main__":
    main()
