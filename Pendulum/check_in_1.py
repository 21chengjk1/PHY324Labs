"""
This python file is to complete check in 1
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

PRINT_DATA_THETA_V_TIME = True

def linear(x, m, c):
    """
    y = mx + c
    """
    return m*x + c

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


def main():
    "Main function"
    print("Loading Data...")
    time, theta = np.loadtxt(
        'data/angleDegreesPendulum.txt',
        delimiter=',', skiprows=1000,  # skip 3 lines: blank, header, column names
        unpack=True
    )

    print("Refining Peaks...")
    peaks, _= find_peaks(theta)
    refined_times, refined_thetas = [], []
    time_errs, theta_errs = [], []
    for i in peaks:
        if theta[i] > 0:  # keep only positive peaks
            t_max, th_max, dt, dth = refine_peak(i, time, theta, sigma_theta=0.05, window=3)
            refined_times.append(t_max)
            refined_thetas.append(th_max)
            time_errs.append(dt)
            theta_errs.append(dth)

    refined_times = np.array(refined_times)
    refined_thetas = np.array(refined_thetas)
    time_errs = np.array(time_errs)
    theta_errs = np.array(theta_errs)

    # Periods and uncertainties
    time_periods = refined_times[1:] - refined_times[:-1]
    time_periods_err = np.sqrt(time_errs[1:]**2 + time_errs[:-1]**2)

    # Final regression: theta_max vs period
    y_values = time_periods
    x_values = refined_thetas[:-1]
    y_unc = time_periods_err
    x_unc = theta_errs[:-1]

    p_opt, p_cov = curve_fit(linear, x_values, y_values, sigma=y_unc, absolute_sigma=True)
    p_var = np.diag(p_cov)
    sd = np.sqrt(p_var)

    print("FITTED PARAMETERS:")
    print("m =", format_uncertainty(p_opt[0], sd[0]))
    print("c =", format_uncertainty(p_opt[1], sd[1]))


    # Calculate reduced chi-squared
    residuals = y_values - linear(x_values, *p_opt)
    chi2 = np.sum((residuals / y_unc)**2)
    ndof = len(y_values) - len(p_opt)  # degrees of freedom
    chi2_red = chi2 / ndof
    print(f"Reduced Chi^2 = {chi2_red:.2f}")


    # --- Plot for Period vs Angle ---
    plt.figure(2)
    x_linspace = np.linspace(min(x_values), max(x_values), 1000)
    plt.errorbar(x_values, y_values, yerr=y_unc,
                ls='', marker='o', label='Refined maxima', capsize=2, zorder=1)
    plt.plot(x_linspace, linear(x_linspace, *p_opt), label="Fit", color="green", zorder=2)
    plt.xlabel("Max Angle (Degrees)")
    plt.ylabel("Time period (s)")
    plt.title("Period v. Angle")
    plt.legend()

    plt.savefig("results/period_vs_angle.png", dpi=300, bbox_inches="tight")
    plt.show()


    # --- Residuals plot for Period v Angle ---
    plt.figure(3)
    plt.errorbar(
        x_values, residuals, yerr=y_unc,
        ls='', marker='o', color="blue", capsize=2
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Max Angle (Degrees)")
    plt.ylabel("Residuals (s)")
    plt.title("Residuals of Linear Fit")

    plt.savefig("results/period_vs_angle_residual.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()