"""
Python file to do pulses in Cables Experiment.
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.constants import speed_of_light

# if SHOW_OSC_PLOTS:
#     plt.figure(n)
#     plt.plot(x, y)
#     plt.title(f"{os.path.basename(filename)} — L = {length} cm\nΔt = {time_delay*1e9:.2f} ns")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Voltage (V)")
#     plt.axvline(t1, color='r', linestyle='--')
#     plt.axvline(t2, color='r', linestyle='--')
#     plt.show()



SHOW_OSC_PLOTS = False
SHOW_FIT_PLOTS = True

def linear(x, m, c):
    """
    y = mx + c
    """
    return m*x + c

def find_time_delay(x, y, filename):
    """
    Finds time delay between the two main spikes.
    """
    # --- Estimate baseline and dynamic threshold ---
    baseline = np.median(y)
    y_temp = y - baseline
    y_temp = np.abs(y_temp)

    threshold = baseline + 0.5 * (y_temp)  # halfway toward max
    
    # --- Detect peaks above threshold ---

    # plt.figure(100)
    # plt.plot(x,y)
    # plt.show()

    peaks, _ = find_peaks(y_temp, height=threshold, distance=len(y)//20)

    # Require at least 2 peaks to measure delay
    if len(peaks) < 2:
        print("Less than two spikes detected", filename)
        return None, (-1,-1), -1, (-1,-1)

    # Pick the two largest (by height)
    peak_heights = y_temp[peaks]
    top_two = peaks[np.argsort(peak_heights)[-2:]]
    top_two = np.sort(top_two)

    t1, t2 = x[top_two]
    time_delay = abs(t2 - t1)

    if SHOW_OSC_PLOTS:
        plt.figure()
        plt.plot(x, y, label='data')
        plt.title(f"{filename}, time delay is {time_delay}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.axvline(t1, color='r', linestyle='--')
        plt.axvline(t2, color='r', linestyle='--')
        plt.show()

    Vinitial, Vreflected = y[top_two]
    attenuation = 20 * np.log10(abs(Vreflected / Vinitial))

    width1 = -1
    width2 = -1
    return time_delay, (t1, t2), width1, width2


def main():
    print("---START---")
    print("Hello Welcome to the Pulses in Cables experiment!")

    lengths = np.array([15.09, 30.5, 48.36, 75.84])  # m
    length_unc = 0.02

    # Create 4 lists of files
    csvs_15 = glob.glob("data/a-15m/*.csv")
    csvs_30 = glob.glob("data/b-30m/*.csv")
    csvs_48 = glob.glob("data/c-48m/*.csv")
    csvs_75 = glob.glob("data/d-75m/*.csv")

    the_csvs = [csvs_15, csvs_30, csvs_48, csvs_75]

    data_records = []  # list of dicts to store file data

    for n, (csv_lst, length) in enumerate(zip(the_csvs, lengths)):
        times = []
        for filename in csv_lst:
            x, y = np.loadtxt(filename, delimiter=',', skiprows=2, unpack=True)
            time_delay, (t1, t2), width1, width2 = find_time_delay(x, y, filename)                                                                             
            if time_delay is not None:
                times.append(time_delay)

        print(f'At length {length}, the time delays are:', times)
        times = np.array(times)

        avg_time_delay = np.mean(times)
        avg_time_delay_unc = np.std(times, ddof=1) / np.sqrt(len(times))

        # Store data
        data_records.append({
            'length': length,
            'avg_time_delay': avg_time_delay,
            'avg_time_delay_unc': avg_time_delay_unc
        })


    lengths = np.array([d['length'] for d in data_records])
    time_delays = np.array([d['avg_time_delay'] for d in data_records])
    time_delays_unc = np.array([d['avg_time_delay_unc'] for d in data_records])

    # ============= Plot Length vs. Time Delay ==========================================

    print("\n==Plot Length vs. Time Delay==")
    y_values = time_delays
    y_unc = np.mean(time_delays_unc)*0.8        # The uncertainty is caused by noise, take an average for consistency
    # y_unc = [3e-9]*len(y_values) 
    print("avg_td_unc :", time_delays_unc)
    x_values = lengths * 2          # Travels through twice, convert to m.

    p_opt, p_cov = curve_fit(linear, x_values, y_values, sigma=y_unc, absolute_sigma=True)
    # p_opt, p_cov = curve_fit(linear, x_values, y_values)  # If i don't have a y_unc yet
    p_var = np.diag(p_cov)
    sd = np.sqrt(p_var)

    print("FITTED PARAMETERS:")
    print("m =", p_opt[0], "+-", sd[0])
    print("c =", p_opt[1], "+-", sd[1])

    v = 1/p_opt[0]
    v_unc = sd[0] / (p_opt[0] ** 2)

    print(f"This means that the speed through the cable is {v/speed_of_light} +- {v_unc/speed_of_light} c")

    x_linspace = np.linspace(min(x_values), max(x_values), 100)

    # Calculate reduced chi-squared
    residuals = y_values - linear(x_values, *p_opt)
    chi2 = np.sum((residuals / y_unc)**2)
    ndof = len(y_values) - len(p_opt)  # degrees of freedom
    chi2_red = chi2 / ndof
    print(f"Reduced Chi^2 = {chi2_red:.2f}")

    if SHOW_FIT_PLOTS:
        plt.figure(100) 
        plt.errorbar(x_values, y_values, yerr=y_unc, color='blue', ls='', marker='o', label='Data')
        plt.plot(x_linspace, linear(x_linspace, *p_opt), label="Fit", color="green")
        plt.xlabel("Two Cable lengths (m)")
        plt.ylabel("Time Delay(s)")
        plt.title("Time Delay vs. Length")
        plt.legend()
        plt.grid(True)

        # Plot Residuals
        plt.figure(101)
        plt.title("Residuals: Time Delay vs. Length")
        plt.errorbar(
            x_values, residuals, yerr=y_unc,
            ls='', marker='o', color="blue", capsize=2
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Two Cable lengths (m)")
        plt.ylabel("Residuals (s)")
        plt.show()

    print("---END---")


if __name__ == '__main__':
    main()