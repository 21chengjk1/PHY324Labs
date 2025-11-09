"""
Unsubmitted File to mess around
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.constants import Boltzmann

SHOW_PLOTS = True
CROP_PLOTS = False

def linear(x, m, c):
    """
    y = mx + c
    """
    return m*x + c

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

    decimals += 1

    # round both value and uncertainty
    unc_rounded = round(uncertainty, decimals)
    val_rounded = round(value, decimals)

    # return f"{val_rounded:.2e} ± {unc_rounded:.2e}"

    return f"{value:.3e} ± {uncertainty:.3e}"

def check_unc(y_unc):
    print("y_unc: min, median, max:", np.nanmin(y_unc), np.nanmedian(y_unc), np.nanmax(y_unc))
    print("any zeros?:", np.any(y_unc == 0))
    print("any negative?:", np.any(y_unc < 0))
    print("any nan?:", np.any(np.isnan(y_unc)))
    print("any inf?:", np.any(np.isinf(y_unc)))

def main():
    """
    Main function. Look at the pseudocode
    LOAD the data so that we have x[beadnumber][datapointforthisbead] and same for y[][] 
    INITIALIZE x_squared_list[], y_squared_list[]
    
    FOR i taking on values 1,2,..., 13:
        INITIALIZE x_start = x[i][0]
        INITIALIZE y_start = y[i][0]
        FOR j taking on values 0 to last index of datapoints
            x_squared_list.append((x[i][j] - x_start)**2)
            y_squared_list.append((y[i][j] - y_start)**2)
        ENDFOR
    ENDFOR

    mean_r_sqaured = MEAN(x_squared_list) + MEAN(y_squared_list)
    INITIALIZE time[0, 0.5, 1,...] to be the length of mean_r_sqaured
    PLOT time on x axis, mean_r_sqaured on y axis
    """
    
    # Store all bead data
    x = []
    y = []

    # Load all 13 bead files
    for bead_num in range(1, 14):  # bead1.txt → bead13.txt
        filename = f"data/bead{bead_num}.txt"
        x_bead, y_bead = np.loadtxt(filename, skiprows=2, unpack=True)
        x.append(x_bead)
        y.append(y_bead)

    # Make sure all beads have the same number of datapoints
    n_points = min(len(arr) for arr in x)

    # Compute mean squared displacement as a function of time
    mean_r_squared_pixels = []
    mean_r_squared_pixels_unc = []

    for j in range(n_points):
        r2_values = []
        for i in range(13):  # beads
            dx = x[i][j] - x[i][0]
            dy = y[i][j] - y[i][0]
            r2_values.append(dx**2 + dy**2)
        mean_r_squared_pixels.append(np.mean(r2_values))
        mean_r_squared_pixels_unc.append(np.std(r2_values, ddof=1) / np.sqrt(len(r2_values)))

    mean_r_squared_pixels = np.array(mean_r_squared_pixels)
    mean_r_squared_pixels_unc = np.array(mean_r_squared_pixels_unc)

    # Time axis: 0, 0.5, 1, ...
    time = np.arange(n_points) * 0.5  # adjust 0.5s if needed

    # conversion factor: pixel to metres
    pixel_size_m = 0.12048e-6
    mean_r_squared = mean_r_squared_pixels * (pixel_size_m ** 2)
    mean_r_squared_unc = mean_r_squared_pixels_unc * (pixel_size_m ** 2)

    # Final regression
    y_values = mean_r_squared
    x_values = time
    y_unc = mean_r_squared_unc
    x_unc = ...

    # TEMPORARY FIX.
    y_unc[y_unc == 0] = 1e-20

    # y_values = mean_r_squared_pixels
    # x_values = time
    # y_unc = mean_r_squared_pixels_unc
    # x_unc = ...



    p_opt, p_cov = curve_fit(linear, x_values, y_values, sigma=y_unc, absolute_sigma=True, p0=(0.00000000000114, -0.0000000000069))
    # p_opt, p_cov = curve_fit(linear, x_values, y_values)  # If i don't have a y_unc yet
    p_var = np.diag(p_cov)
    sd = np.sqrt(p_var)

    print("FITTED PARAMETERS:")
    print("m =", format_uncertainty(p_opt[0], sd[0]))
    print("c =", format_uncertainty(p_opt[1], sd[1]))
    
    # Physical constants (SI units)
    T = 296.5                   # Kelvin
    # T = 305                   # This is how high the temp would need to be to get a good result... 32 degrees C! :O
    T_sd = 0.5

    # eta = 0.93e-3               # Pa s
    # eta_sd = 0.05e-3

    # Convert temperature to Celsius
    T_C = T - 273.15
    T_C_sd = T_sd

    # Reference viscosity at 20°C
    eta0 = 1.00e-3           # Pa·s  (1.00 cP)
    eta0_sd = 0.05e-3        # Pa·s uncertainty

    # Viscosity decreases 2% per °C increase
    rate = 0.02              # fractional change per degree

    # Compute viscosity at T_C
    delta_T = T_C - 20.0

    # Propagate uncertainties:
    rel_eta_unc_from_eta0 = eta0_sd / eta0
    rel_eta_unc_from_T = abs(np.log(1 - rate)) * T_C_sd
    rel_eta_unc_total = np.sqrt(rel_eta_unc_from_eta0**2 + rel_eta_unc_from_T**2)
    eta = eta0 * (1 - rate)**delta_T
    eta_sd = eta * rel_eta_unc_total

    r = 1.9e-6                  # radius of bead in metres
    r_sd = 0.1e-6

    # Extract slope from fit (m = 4D)
    m_fit = p_opt[0]
    m_sd = sd[0]

    # Compute D and its uncertainty
    D = m_fit / 4
    D_sd = m_sd / 4

    # Compute Boltzmann constant
    k = (6 * np.pi * eta * r * D) / T

    rel_unc_k = np.sqrt(
        (D_sd / D)**2 +
        (eta_sd / eta)**2 +
        (r_sd / r)**2 +
        (T_sd / T)**2
    )

    print("rel_unc_k = ", rel_unc_k)

    k_sd = k * rel_unc_k

    print("Diffusion constant D =", format_uncertainty(D, D_sd))
    print("Boltzmann constant k =", format_uncertainty(k, k_sd))
    print("So, an initial guess at k =", k)

    print("The accepted answer for Boltzmann's constant is:", Boltzmann)
    if Boltzmann >= (k - k_sd) and Boltzmann <= (k + k_sd):
        print("The accpeted value is within our estimated range")
    else:
        print("The accpeted value is OUTSIDE our estimated range")

    # Calculate reduced chi-squared
    residuals = y_values - linear(x_values, *p_opt)
    chi2 = np.sum((residuals / y_unc)**2)
    ndof = len(y_values) - len(p_opt)  # degrees of freedom
    chi2_red = chi2 / ndof
    print(f"Reduced Chi^2 = {chi2_red:.2f}")

    if SHOW_PLOTS:
        # Plot
        plt.figure(1)
        plt.title("Average Mean Squared Displacement against Time Elapsed")
        plt.errorbar(x_values, y_values, yerr=y_unc, ls='', marker='o', ms=2 ,label=r"$\langle r^2(t) \rangle$")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean squared displacement (metres ^ 2)")
        x_linspace = np.linspace(min(x_values), max(x_values), 100)
        plt.plot(x_linspace, linear(x_linspace, *p_opt), label="Fit", color="green")
        # PLOT THE ACCPTED VALUE AS WELL...
        # plt.plot(x_linspace, linear(x_linspace, ), label="accepted value", color="red")
        plt.legend()
        if CROP_PLOTS:
            plt.xlim(0, 30)
        plt.show()


        # Plot Residuals
        plt.figure(2)
        plt.title("Residuals: Average Mean Squared Displacement against Time Elapsed")
        plt.errorbar(
            x_values, residuals, yerr=y_unc,
            ls='', marker='o', color="blue", capsize=2
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Residuals (metres ^ 2)")

        # plt.savefig("results/period_vs_angle_residual.png", dpi=300, bbox_inches="tight")
        if CROP_PLOTS:
            plt.xlim(0, 30)
        plt.show()

    # ===============================

if __name__ == "__main__":
    main()