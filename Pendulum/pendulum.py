"""
This is the code for the Pendulum Project in PHY324

Jack Cheng 1010266695
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

PLOT_RAW_DATA = False
PlOT_T_VS_AMPLITUDE = True
PLOT_SYMMETRY = True
PLOT_EXPONENTIAL = True

LONG, MEDIUM, SHORT = 0.594, 0.450, 0.357
DISTANCE_TO_CM = 0.01

def constant(x, c0):
    return c0 * np.ones_like(x)

def linear(x, m, c):
    """
    y = mx + c
    """
    return m*x + c

def quadratic(t, a, b, c):
    return a*t**2 + b*t + c

def exponential(t, theta_0, tau):
    """
    y = theta_0 e^(-t/tau)
    """
    return theta_0 * np.exp(-t/tau)

def power_law(x, a, b):
    return a * x**b

def sqrt_model(x, a, b):
    return a * np.sqrt(x) + b



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

def chi2_red(model, x, y, yerr, params):
    res = y - model(x, *params)
    return np.sum((res / yerr)**2) / (len(y) - len(params))

def format_uncertainty(value, uncertainty):
    """
    Format value ± uncertainty so that:
    - uncertainty has 1 significant figure
    - value is rounded to same decimal place
    """
    
    if uncertainty == 0:
        return f"{value} ± 0"

    # order of magnitude of uncertainty
    exponent = int(np.floor(np.log10(abs(uncertainty))))

    # number of decimal places to round to
    decimals = -exponent if exponent < 0 else 0

    # round both value and uncertainty
    unc_rounded = round(uncertainty, decimals)
    val_rounded = round(value, decimals)

    return f"{val_rounded:.{decimals}f} ± {unc_rounded:.{decimals}f}"

def refine_each_peak(time, theta):
    """
    Returns refined_times, refined_thetas, time_errs, theta_errs
    """
    peaks, _ = find_peaks(theta)
    refined_times, refined_thetas = [], []
    time_errs, theta_errs = [], []

    # refine each peak
    for i in peaks:
        if theta[i] > 0:  # only positive peaks
            t_max, th_max, dt, dth = refine_peak(i, time, theta,
                                                 sigma_theta=0.05,
                                                 window=3)
            refined_times.append(t_max)
            refined_thetas.append(th_max)
            time_errs.append(dt)
            theta_errs.append(max(dth,0.25))

    refined_times = np.array(refined_times)
    refined_thetas = np.array(refined_thetas)
    time_errs = np.array(time_errs)
    theta_errs = np.array(theta_errs)

    # plt.figure()
    # plt.plot(refined_times, refined_thetas)
    # plt.title("TEST TEST")
    
    # ---- Remove outliers in refined_thetas based on neighbor deviation ----
    # Compute magnitude of local changes
    diffs = np.abs(np.diff(refined_thetas))

    # Robust scale of typical variation
    threshold = 3 * np.median(diffs)

    keep_indices = []

    for i in range(len(refined_thetas)):
        if i == 0:
            # Only right neighbor
            right_diff = abs(refined_thetas[i] - refined_thetas[i+1])
            if right_diff < threshold:
                keep_indices.append(i)

        elif i == len(refined_thetas) - 1:
            # Only left neighbor
            left_diff = abs(refined_thetas[i] - refined_thetas[i-1])
            if left_diff < threshold:
                keep_indices.append(i)

        else:
            # Interior point — keep if consistent with either neighbor
            left_diff  = abs(refined_thetas[i] - refined_thetas[i-1])
            right_diff = abs(refined_thetas[i] - refined_thetas[i+1])

            if left_diff < threshold or right_diff < threshold:
                keep_indices.append(i)

    keep_indices = np.array(keep_indices)

    # ---- Apply mask to all arrays ----
    refined_times  = refined_times[keep_indices]
    refined_thetas = refined_thetas[keep_indices]
    time_errs      = time_errs[keep_indices]
    theta_errs     = theta_errs[keep_indices]

    # Discard data with massive uncertainties.
    mask = theta_errs < 10
    refined_times  = refined_times[mask]
    refined_thetas = refined_thetas[mask]
    time_errs      = time_errs[mask]
    theta_errs     = theta_errs[mask]

    # plt.figure()
    # plt.plot(refined_times, refined_thetas)
    # plt.title("TEST TEST after refining")
    # plt.show()

    return refined_times, refined_thetas, time_errs, theta_errs

def run_time_amplitude_dependence(time, theta, label):
    """Essentially Perform check-in 1 analysis on a single dataset."""

    print(f"\n=== Running Check-In 1 for: {label} ===")

    # --- Find peaks ---
    refined_times, refined_thetas, time_errs, theta_errs = refine_each_peak(time, theta)

    # Periods and uncertainties
    time_periods = refined_times[1:] - refined_times[:-1]
    time_periods_err = np.sqrt(time_errs[1:]**2 + time_errs[:-1]**2)

    # Omit last angle (no next peak)
    x_values = refined_thetas[:-1]
    y_values = time_periods
    y_unc = time_periods_err

    # ---- FILTER: discard useless and bad data.
    mask = y_values < 1.5
    x_values = x_values[mask]
    y_values = y_values[mask]
    y_unc    = y_unc[mask]

    mask = y_values > 1.2
    x_values = x_values[mask]
    y_values = y_values[mask]
    y_unc    = y_unc[mask]

    mask = y_unc < 0.1
    x_values = x_values[mask]
    y_values = y_values[mask]
    y_unc    = y_unc[mask]

    models = {
        "Constant":  (constant, 1),
        "Linear":    (linear,   2),
        "Quadratic": (quadratic, 3)
    }
    
    colors = {
        "Constant" : "red",
        "Linear" : "orange",
        "Quadratic" : "green"
    }

    fit_params   = {}
    fit_errors   = {}
    chi2_dict    = {}
    chi2_red_dict = {}
    residuals_dict = {}

    # ---------- PERFORM ALL FITS ----------
    for name, (func, npars) in models.items():
        p_opt, p_cov = curve_fit(func, x_values, y_values,
                                 sigma=y_unc, absolute_sigma=True,
                                 maxfev=10000)
        sd = np.sqrt(np.diag(p_cov))

        # store results
        fit_params[name] = p_opt
        fit_errors[name] = sd

        # chi-square
        residuals = y_values - func(x_values, *p_opt)
        chi2 = np.sum((residuals / y_unc)**2)
        ndof = len(y_values) - npars
        chi2_red = chi2 / ndof

        chi2_dict[name] = chi2
        chi2_red_dict[name] = chi2_red
        residuals_dict[name] = residuals

        # print summary
        print(f"\n=== {name} Fit ===")
        for i, (par, err) in enumerate(zip(p_opt, sd)):
            print(f"p{i} = {par:.6f} ± {err:.6f}")
        print(f"Chi^2       = {chi2:.3f}")
        print(f"Chi^2_red   = {chi2_red:.3f}")

    # ---------- PLOTTING ----------
    if PlOT_T_VS_AMPLITUDE:
        x_lin = np.linspace(min(x_values), max(x_values), 1000)

        # --- Main fit plot ---
        plt.figure()
        plt.errorbar(x_values, y_values, yerr=y_unc,
                     ls='', marker='o', capsize=2,
                     label="Data", zorder=1)

        # plot all fits
        plt.plot(x_lin, constant(x_lin, *fit_params["Constant"]),
                 label="Constant Fit", zorder=2, color = colors["Constant"])
        plt.plot(x_lin, linear(x_lin, *fit_params["Linear"]),
                 label="Linear Fit", zorder=3, color = colors["Linear"])
        plt.plot(x_lin, quadratic(x_lin, *fit_params["Quadratic"]),
                 label="Quadratic Fit", zorder=4, color=colors["Quadratic"])

        plt.xlabel("Max Angle (Degrees)")
        plt.ylabel("Time Period (s)")
        plt.title(f"Period vs Angle ({label})")
        plt.legend()

        # --- Residuals plot ---
        plt.figure()
        for name in models.keys():
            plt.errorbar(x_values, residuals_dict[name], yerr=y_unc,
                         ls='', marker='o', capsize=2,
                         label=f"{name} Residuals", color=colors[name])

        plt.axhline(0, color="gray", linestyle="--")
        plt.xlabel("Max Angle (Degrees)")
        plt.ylabel("Residuals (s)")
        plt.title(f"Residuals ({label})")
        plt.legend()

        plt.show()

def fit_exponential_decay(refined_times, refined_thetas, refined_theta_errs, title):
    """returns tau, tau_err"""
    x_values = refined_times
    y_values = refined_thetas
    y_unc = refined_theta_errs

    # Fit line
    p_opt, p_cov = curve_fit(exponential, x_values, y_values, sigma=y_unc, absolute_sigma=True, p0=(y_values[0], 10))
    # p_opt, p_cov = curve_fit(exponential, x_values, y_values, p0=(y_values[0], 10))   # if i have no uncertainty yet
    sd = np.sqrt(np.diag(p_cov))

    print("== Exponential Fit ==")
    print("FITTED PARAMETERS:")
    print("theta0 =", p_opt[0], sd[0])
    print("tau =", p_opt[1], sd[1])

    residuals = y_values - exponential(x_values, *p_opt)
    chi2 = np.sum((residuals / y_unc)**2)
    ndof = len(y_values) - len(p_opt)
    chi2_red = chi2 / ndof
    print(f"Reduced Exponential: Chi^2 = {chi2_red:.2f}")

    if PLOT_EXPONENTIAL:
        x_lin = np.linspace(min(x_values), max(x_values), 1000)
        plt.figure()
        plt.errorbar(x_values, y_values, yerr=y_unc, ls='', marker='o', capsize=2, zorder=1)
        # plt.plot(x_values, y_values, ls='', marker='o')
        plt.plot(x_lin, exponential(x_lin, *p_opt),
                label="Fit", zorder=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (degrees)")
        plt.title(f"Exponential Fit ({title})")
        plt.legend()

        # and the residuals plot for the exponential
        plt.figure()
        plt.errorbar(
            x_values,
            residuals,
            yerr=y_unc,
            ls='', marker='o', capsize=3
        )
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Residual (degrees)")
        plt.title(f"Residuals of Exponential Fit ({title})")
        plt.grid(True)

        plt.show()

    return p_opt[1], sd[1]

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

    Mlen_thetas = [Langle_Mlen_theta, Mangle_Mlen_theta, Sangle_Mlen_theta]
    Mlen_times = [Langle_Mlen_time, Mangle_Mlen_time, Sangle_Mlen_time]
    Mlen_tiles = ["Large angle Medium Length", "Medium angle Medium length", "Small angle Medium length"]

    Mangle_times = [Mangle_Llen_time, Mangle_Mlen_time, Mangle_Slen_time]
    Mangle_thetas = [Mangle_Llen_theta, Mangle_Mlen_theta, Mangle_Slen_theta]
    Mangle_titles = ["Medium angle Long length", "Medium angle Medium length",  "Medium angle Short length"]

    # Question 0: Plot the raw data for myself.
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
    for t, th, title in zip(Mlen_times, Mlen_thetas, Mlen_tiles):
        print(f"==========================={title}===============================")

        run_time_amplitude_dependence(t, th, title)
        refined_times, refined_thetas, _ , refined_theta_errs = refine_each_peak(t, th)
        run_time_amplitude_dependence(t, -th, title+" (Negative)")
        refined_times_neg, refined_thetas_neg, _, _ = refine_each_peak(t, -th)

        if PLOT_SYMMETRY:
            # Find the symmetry
            plt.figure()
            plt.plot(refined_times, refined_thetas, label='positive')
            plt.plot(refined_times_neg, refined_thetas_neg, label='negative')
            plt.legend()
            plt.show()

            # ----- Build common time grid over the overlapping domain -----
            t_min = max(refined_times.min(), refined_times_neg.min())
            t_max = min(refined_times.max(), refined_times_neg.max())

            t_common = np.linspace(t_min, t_max, 2000)

            # ----- Interpolate both curves onto this common grid -----
            f_pos = interp1d(refined_times, refined_thetas, kind='linear',
                            bounds_error=False, fill_value="extrapolate")
            f_neg = interp1d(refined_times_neg, refined_thetas_neg, kind='linear',
                            bounds_error=False, fill_value="extrapolate")

            theta_pos_interp = f_pos(t_common)
            theta_neg_interp = f_neg(t_common)

            # ----- Compute asymmetry curve -----
            asymmetry_curve = theta_pos_interp - theta_neg_interp

            # ----- Quantify asymmetrry -----
            unsymmetry_metric = np.mean(asymmetry_curve)
            print("Unsymmetry metric:", unsymmetry_metric)

            # ----- Plot -----
            plt.figure()
            plt.axhline(0, color='black', linewidth=0.8)
            plt.plot(t_common, asymmetry_curve)
            plt.axhline(unsymmetry_metric, color='red', linestyle='--', label=f"Mean = {unsymmetry_metric:.5f}")
            plt.xlabel("Time")
            plt.ylabel("Theta_pos - Theta_neg")
            plt.title(f"System Asymmetry ({title})")
            plt.legend()
            plt.show()

        # Q2: Verify or refute the claim that the decay is exponential, and determine the time constant τ . You can do this for a single length.
        '''
        Plan.
        Only for a single length, so just use the 3 M_len files.

        1) find peaks.
        2) fit the peaks to an exponential curve
        3) comment on the goodness of fit, to determine if this decay is well described by an exponential.
        '''

        tau, tau_err = fit_exponential_decay(refined_times, refined_thetas, refined_theta_errs, title)

    # Q3: Verify or refute the claim that the period depends on L as stated: T = 2(L + D)1/2 (provided that D2 ≪ L2).
    '''
    Use the 3 M_angle files

    1) Calculate average time period of each of the 3 files
    2) Plot
    3) Comment on fit.
    
    '''
    T_list = []
    T_err_list = []
    L_list = [LONG, MEDIUM, SHORT]
    D = DISTANCE_TO_CM
    for t, th, title in zip(Mangle_times, Mangle_thetas, Mangle_titles):
        refined_times, refined_thetas, time_errs, theta_errs = refine_each_peak(t, th)

        # Compute individual periods
        periods = np.diff(refined_times)
        period_errs = np.sqrt(time_errs[1:]**2 + time_errs[:-1]**2)

        # Average T for this dataset
        T_avg = np.mean(periods)
        T_sem = np.std(periods, ddof=1) / np.sqrt(len(periods))

        T_list.append(T_avg)
        T_err_list.append(T_sem)

    # Convert to arrays
    T_list = np.array(T_list)
    T_err_list = np.array(T_err_list)
    L_eff = np.array(L_list) + D

    # ---- Plot measured average periods ----
    plt.figure()
    plt.errorbar(L_eff, T_list, yerr=T_err_list, fmt='o', capsize=4, label="Measured T")

    # ---- Plot theoretical line T = 2 sqrt(L + D) ----
    L_smooth = np.linspace(min(L_eff), max(L_eff), 200)
    T_theoretical = 2 * np.sqrt(L_smooth)

    plt.plot(L_smooth, T_theoretical, label=r"$T = 2\sqrt{L + D}$")

    plt.xlabel("L + D (m)")
    plt.ylabel("Average Period T (s)")
    plt.title("Period vs Effective Length")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Q4: investigate the effect of L + D and θ0 on τ . 
    # If you find a trend, attempt to find a function which fits your data. 
    # You do not need to have a theoretical justification for your function; you do need to justify your choice in terms of goodness of fit in some fashion. 
    # As no prediction was given for this, there is nothing to verify or refute. Since you lack a prediction, you should stick with basic fit functions: 
    # linear functions, power series, power-laws, exponential functions.
    '''
    Help!
    '''
    tau_list = []
    tau_err_list = []
    theta0_list = []

    titles = [i+"WOW" for i in titles]

    for t, th, title in zip(times, thetas, titles):

        # --- take first 100 points ---
        t = t[:1000]
        th = th[:1000]

        # --- refine peaks ---
        refined_times, refined_thetas, time_errs, theta_errs = refine_each_peak(t, th)

        # --- compute tau and tau uncertainty ---
        tau, tau_err = fit_exponential_decay(refined_times, refined_thetas, theta_errs, title)

        # --- estimate initial amplitude theta0 ---
        theta0 = np.max(np.abs(refined_thetas))

        # --- append results ---
        tau_list.append(tau)
        tau_err_list.append(tau_err)
        theta0_list.append(theta0)

    # Convert to arrays
    tau_list = np.array(tau_list)
    tau_err_list = np.array(tau_err_list)
    theta0_list = np.array(theta0_list)

    # Effective lengths L + D (matching 5 datasets)
    Leff_list = np.array([MEDIUM, MEDIUM, LONG, SHORT, MEDIUM]) + D
    
    print("\n=== τ vs Effective Length L_eff ===")
    # Perform fits
    p_lin_L, c_lin_L = curve_fit(linear,     Leff_list, tau_list, sigma=tau_err_list, absolute_sigma=True)
    p_pow_L, c_pow_L = curve_fit(power_law,  Leff_list, tau_list, sigma=tau_err_list, absolute_sigma=True)
    p_sqt_L, c_sqt_L = curve_fit(sqrt_model, Leff_list, tau_list, sigma=tau_err_list, absolute_sigma=True)

    # Compute reduced chi²
    chi_lin_L = chi2_red(linear,     Leff_list, tau_list, tau_err_list, p_lin_L)
    chi_pow_L = chi2_red(power_law,  Leff_list, tau_list, tau_err_list, p_pow_L)
    chi_sqt_L = chi2_red(sqrt_model, Leff_list, tau_list, tau_err_list, p_sqt_L)

    print(f"Linear χ² = {chi_lin_L:.3f}")
    print(f"Power-law χ² = {chi_pow_L:.3f}")
    print(f"Sqrt χ² = {chi_sqt_L:.3f}")

    # Pick best model
    chi_vals_L = np.array([chi_lin_L, chi_pow_L, chi_sqt_L])
    models_L = [linear, power_law, sqrt_model]
    params_L = [p_lin_L, p_pow_L, p_sqt_L]
    best_idx_L = np.argmin(chi_vals_L)

    best_model_L = models_L[best_idx_L]
    best_params_L = params_L[best_idx_L]

    print("\nBest model for τ(L_eff):", best_model_L.__name__)
    print("Best-fit parameters:", best_params_L)


    # Plot τ vs L_eff
    L_plot = np.linspace(min(Leff_list), max(Leff_list), 400)

    plt.figure()
    plt.errorbar(Leff_list, tau_list, yerr=tau_err_list, fmt='o', label="Data")
    plt.plot(L_plot, best_model_L(L_plot, *best_params_L), label=f"Best fit: {best_model_L.__name__}")
    plt.xlabel("Effective Length $L + D$ (m)")
    plt.ylabel("Damping Time Constant $\\tau$ (s)")
    plt.title("Dependence of Damping Constant on Effective Length")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



    # ----------------------------
    # Fit τ vs Initial Amplitude θ0
    # ----------------------------
    print("\n=== τ vs Initial Amplitude θ0 ===")

    # Perform fits
    p_lin_t, c_lin_t = curve_fit(linear,     theta0_list, tau_list, sigma=tau_err_list, absolute_sigma=True)
    p_pow_t, c_pow_t = curve_fit(power_law,  theta0_list, tau_list, sigma=tau_err_list, absolute_sigma=True)
    p_sqt_t, c_sqt_t = curve_fit(sqrt_model, theta0_list, tau_list, sigma=tau_err_list, absolute_sigma=True)

    sd_lin  = np.sqrt(np.diag(c_lin_t))
    sd_pow  = np.sqrt(np.diag(c_pow_t))
    sd_sqt  = np.sqrt(np.diag(c_sqt_t))

    # Compute reduced chi²
    chi_lin_t = chi2_red(linear,     theta0_list, tau_list, tau_err_list, p_lin_t)
    chi_pow_t = chi2_red(power_law,  theta0_list, tau_list, tau_err_list, p_pow_t)
    chi_sqt_t = chi2_red(sqrt_model, theta0_list, tau_list, tau_err_list, p_sqt_t)

    print(f"Linear χ² = {chi_lin_t:.3f}")
    print(f"Power-law χ² = {chi_pow_t:.3f}")
    print(f"Sqrt χ² = {chi_sqt_t:.3f}")

    # Pick best model
    chi_vals_t = np.array([chi_lin_t, chi_pow_t, chi_sqt_t])
    models_t = [linear, power_law, sqrt_model]
    params_t = [p_lin_t, p_pow_t, p_sqt_t]
    param_unc = [sd_lin, sd_pow, sd_sqt]

    best_idx_t = np.argmin(chi_vals_t)
    best_model_t = models_t[best_idx_t]
    best_params_t = params_t[best_idx_t]
    best_unc_t    = param_unc[best_idx_t]

    print("\nBest model for τ(θ0):", best_model_t.__name__)
    print("Best-fit parameters + uncertainties:")
    for i, p in enumerate(best_params_t):
        print(f"  p[{i}] = {p:.6g} ± {best_unc_t[i]:.2g}")


    # Plot τ vs θ0
    th_plot = np.linspace(min(theta0_list), max(theta0_list), 400)

    plt.figure()
    plt.errorbar(theta0_list, tau_list, yerr=tau_err_list, fmt='o', label="Data")
    plt.plot(th_plot, best_model_t(th_plot, *best_params_t), label=f"Best fit: {best_model_t.__name__}")
    plt.xlabel("Initial Amplitude $\\theta_0$ (degrees)")
    plt.ylabel("Damping Time Constant $\\tau$ (s)")
    plt.title("Dependence of Damping Constant on Initial Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    
    resid = tau_list - best_model_t(theta0_list, *best_params_t)
    plt.figure()
    plt.errorbar(theta0_list, resid, yerr=tau_err_list,
                fmt='o', capsize=3)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Initial Amplitude $\\theta_0$ (degrees)")
    plt.ylabel("Residuals (s)")
    plt.title(f"Residuals for Best Fit ({best_model_t.__name__})")
    plt.grid(alpha=0.3)


    plt.show()


    print("=== Program end ===")
    


if __name__ == "__main__":
    main()