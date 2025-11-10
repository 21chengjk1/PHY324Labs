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

SHOW_OSC_PLOTS = False
SHOW_FIT_PLOTS = False

def linear(x, m, c):
    """
    y = mx + c
    """
    return m*x + c

def find_time_delay(x, y):
    """
    Finds time delay between the two main spikes.
    """
    # --- Estimate baseline and dynamic threshold ---
    baseline = np.median(y)
    threshold = baseline + 0.5 * (np.max(y) - baseline)  # halfway toward max
    
    # --- Detect peaks above threshold ---
    peaks, _ = find_peaks(y, height=threshold, distance=len(y)//20)

    # Require at least 2 peaks to measure delay
    if len(peaks) < 2:
        raise ValueError("Less than two spikes detected — adjust threshold or check data")

    # Pick the two largest (by height)
    peak_heights = y[peaks]
    top_two = peaks[np.argsort(peak_heights)[-2:]]
    top_two = np.sort(top_two)

    t1, t2 = x[top_two]
    time_delay = abs(t2 - t1)


    Vinitial, Vreflected = y[top_two]
    attenuation = 20 * np.log10(abs(Vreflected / Vinitial))
    return time_delay, (t1, t2), attenuation, (Vinitial, Vreflected)


def main():
    print("---START---")
    print("Hello Welcome to the Pulses in Cables experiment!")

    lengths = np.array([15.09, 30.5, 48.36, 75.84])  # m
    length_unc = 0.02

    files = glob.glob("data/*.csv")

    data_records = []  # list of dicts to store file data

    for n, (filename, length) in enumerate(zip(files, lengths), start=1):
        x, y = np.loadtxt(filename, delimiter=',', skiprows=2, unpack=True)

        # Find time delay between pulses
        time_delay, (t1, t2), attenuation, (Vinitial, Vreflected) = find_time_delay(x, y)

        # Store everything
        data_records.append({
            'filename': os.path.basename(filename),
            'length': length,
            'x': x,
            'y': y,
            'time_delay': time_delay,
            'attenuation': attenuation
        })

        # Plot each waveform

        if SHOW_OSC_PLOTS:
            plt.figure(n)
            plt.plot(x, y)
            plt.title(f"{os.path.basename(filename)} — L = {length} cm\nΔt = {time_delay*1e9:.2f} ns")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.axvline(t1, color='r', linestyle='--')
            plt.axvline(t2, color='r', linestyle='--')

    if SHOW_OSC_PLOTS:
        plt.show()

    lengths = np.array([d['length'] for d in data_records])
    time_delays = np.array([d['time_delay'] for d in data_records])
    attenuations = np.array([d['attenuation'] for d in data_records])

    # ============= Plot Length vs. Time Delay ==========================================

    print("\n==Plot Length vs. Time Delay==")
    x_values = time_delays
    y_values = lengths * 2 # Travels through twice, convert to m.

    # p_opt, p_cov = curve_fit(linear, x_values, y_values, sigma=y_unc, absolute_sigma=True)
    p_opt, p_cov = curve_fit(linear, x_values, y_values)  # If i don't have a y_unc yet
    p_var = np.diag(p_cov)
    sd = np.sqrt(p_var)

    print("FITTED PARAMETERS:")
    print("m =", p_opt[0], "+-", sd[0])
    print("c =", p_opt[1], "+-", sd[1])

    print(f"This means that the speed through the cable is {p_opt[0]/speed_of_light} +- {sd[0]/speed_of_light} c")

    x_linspace = np.linspace(min(x_values), max(x_values), 100)

    if SHOW_FIT_PLOTS:
        plt.figure(100) 
        plt.plot(x_values, y_values, color='blue', ls='', marker='o', label='Data')
        plt.plot(x_linspace, linear(x_linspace, *p_opt), label="Fit", color="green")
        plt.xlabel("Time Delay (s)")
        plt.ylabel("Cable Length (m)")
        plt.title("Length vs. Time Delay")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ============= Plot Length vs. Attenuation ==========================================

    print("\n==Plot Length vs. Attenuation==")

    x_values = attenuations
    y_values = lengths * 2    # Metres.

    
    # p_opt, p_cov = curve_fit(linear, x_values, y_values, sigma=y_unc, absolute_sigma=True)
    p_opt, p_cov = curve_fit(linear, x_values, y_values)  # If i don't have a y_unc yet
    p_var = np.diag(p_cov)
    sd = np.sqrt(p_var)

    print("FITTED PARAMETERS:")
    print("m =", p_opt[0], "+-", sd[0])
    print("c =", p_opt[1], "+-", sd[1])

    print(f"This means that attenuation per cm is {p_opt[0]}")
    if SHOW_FIT_PLOTS:
        plt.figure(101)
        plt.plot(x_values, y_values, color='blue', ls='', marker='o', label='Data')

        x_linspace = np.linspace(min(x_values), max(x_values), 100)
        plt.plot(x_linspace, linear(x_linspace, *p_opt), label="Fit", color="green")

        plt.xlabel("Attenuation (dB)")
        plt.ylabel("Cable Length (cm)")
        plt.title("Length vs. Time Delay")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("---END---")


if __name__ == '__main__':
    main()