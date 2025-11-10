"""
Code written by Jack's lab partner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

t1, v1 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_15_10.csv", delimiter= ',',skiprows=2, unpack=True)
t2, v2 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_15_27.csv", delimiter= ',',skiprows=2, unpack=True)
t3, v3 =np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_15_51.csv", delimiter= ',',skiprows=2, unpack=True)
t4, v4 =np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_15_100.csv", delimiter= ',',skiprows=2, unpack=True)
t5, v5 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_30_10.csv", delimiter= ',',skiprows=2, unpack=True)
t6, v6 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_30_27.csv", delimiter= ',',skiprows=2, unpack=True)
t7, v7 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_30_51.csv", delimiter= ',',skiprows=2, unpack=True)
t8, v8 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_30_100.csv", delimiter= ',',skiprows=2, unpack=True)
t9, v9 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_48_10.csv", delimiter= ',',skiprows=2, unpack=True)
t10, v10 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_48_27.csv", delimiter= ',',skiprows=2, unpack=True)
t11, v11 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_48_51.csv", delimiter= ',',skiprows=2, unpack=True)
t12, v12 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_48_100.csv", delimiter= ',',skiprows=2, unpack=True)
t13 , v13 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_75_10.csv", delimiter= ',',skiprows=2, unpack=True)
t14 , v14 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_75_27.csv", delimiter= ',',skiprows=2, unpack=True)
t15 , v15 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_75_51.csv", delimiter= ',',skiprows=2, unpack=True)
t16 , v16 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\C_75_100.csv", delimiter= ',',skiprows=2, unpack=True)
t17 , v17 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\Open_C_15.csv", delimiter= ',',skiprows=2, unpack=True)
t18 , v18 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\Open_C_30.csv", delimiter= ',',skiprows=2, unpack=True)
t19 , v19 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\Open_C_48.csv", delimiter= ',',skiprows=2, unpack=True)
t20 , v20 = np.loadtxt(r"C:\Users\danie\Documents\Uoft\PHY324\Pulses\Experiment2\Open_C_75.csv", delimiter= ',',skiprows=2, unpack=True)


#estimate uncertainty in voltage using variance in noise before pulse:
uv= np.mean([np.std(v[:50], ddof=1) for v in [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16]])

#make list of time and voltage arrays for open circuit trials
topen = [t17, t18, t19, t20]
vopen = [v17, v18, v19, v20]

t = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16]
v = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16]
#make list of time and voltage arrays where skipping the trials with the 51 ohm resistor
tskip = [t1, t2, t4, t5, t6, t8, t9, t10, t12, t13, t14, t16]
vskip= [v1, v2, v4, v5, v6, v8, v9, v10, v12, v13, v14, v16]
#separate into sublists for each length
t = [[t1, t2, t4], [t5, t6, t8], [t9, t10, t12], [t13, t14, t16]]
v = [[v1, v2, v4], [v5, v6, v8], [v9, v10, v12], [v13, v14, v16]]


for i in range(4):
    for j in range(3):
        baseline = np.median(v[i][j][:50])  # first 50 samples as baseline
        v[i][j] = v[i][j] - baseline

vabs = []
for i in range(4):
    vabs.append([])
    for j in range(3):
        vabs[i].append(np.abs(v[i][j]))


PLOT_DATA = True
PLOT_SLOPES = True
PLOT_ATTENUATION = True


mu = 1
epsilon = 2.25

peak_signs = [[-1,-1, 1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1]]

peak_signs = np.array(peak_signs)



threshold = 0.25  # volts
skip_samples = 50  # samples to skip after first peak

max_times = []
max_values = []
max2_times = []
max2_values = []

for i in range(4):  # length sets
    times1, values1 = [], []
    times2, values2 = [], []

    for j in range(3):  # trials
        trace = vabs[i][j]  # absolute-value trace
        t_trace = t[i][j]

        # 1️ Find first peak above threshold
        peaks, properties = find_peaks(trace, height=threshold)

        if len(peaks) == 0:
            # no peaks found
            times1.append(None)
            values1.append(None)
            times2.append(None)
            values2.append(None)
            continue

        # First peak
        first_peak_idx = peaks[0]
        times1.append(t_trace[first_peak_idx])
        values1.append(trace[first_peak_idx])

        # 2️ Mask out samples immediately after first peak to skip noise/tail
        start_idx = first_peak_idx + skip_samples
        if start_idx >= len(trace):
            # no room for second peak
            times2.append(None)
            values2.append(None)
            continue

        # Remaining trace for second peak
        trace2 = trace[start_idx:]
        t_trace2 = t_trace[start_idx:]

        # Find second peak in remaining trace
        peaks2, properties2 = find_peaks(trace2, height=threshold)
        if len(peaks2) == 0:
            times2.append(None)
            values2.append(None)
        else:
            second_peak_idx = peaks2[0]  # first peak in remaining trace
            times2.append(t_trace2[second_peak_idx])
            values2.append(trace2[second_peak_idx])

    max_times.append(times1)
    max_values.append(values1)
    max2_times.append(times2)
    max2_values.append(values2)

# find the width of the first pulse at half the height for one trial to estimate uncertainty in time
indicies = []
for i  in range(len(v1)):
    threshold = 0.5*max_values[0][0]
    
    if v1[i] >= threshold:
        indicies.append(i)
u_time = 25*10**(-9)

print(f"Estimated uncertainty in time measurements u_time: {u_time} s")

    

if PLOT_DATA:
    for i in range(4):
        plt.figure()
        for j in range(3):
            plt.plot(t[i][j], vabs[i][j], label=f'Trial {j+1}')
            if max_times[i][j] is not None:
                plt.plot(max_times[i][j], max_values[i][j], 'ro')  # Mark first peak
            if max2_times[i][j] is not None:
                plt.plot(max2_times[i][j], max2_values[i][j], 'go')  # Mark second peak
        plt.title(f'Voltage vs Time for Length Set {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.grid()

max2_times = np.array(max2_times, dtype=np.float64)
max_times = np.array(max_times, dtype=np.float64)
max2_values = np.array(max2_values, dtype=np.float64)
max_values = np.array(max_values, dtype=np.float64)



diff_times = max2_times-max_times
lengths = [15.09, 30.50, 48.36, 75.84]  # lengths in meters
lengths = np.array(lengths)*2
ulength = 0.02  # uncertainty in length measurements







def linear(x, m, b):
    return m*x + b
chiquaredofs = []
slopes = []

for i in range(3):
    p_opt_lin, p_cov_lin = curve_fit(linear, lengths, diff_times.T[i], sigma=u_time, absolute_sigma=True)
    slopes.append(p_opt_lin[0])
    if PLOT_SLOPES:
        fig, axs = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        # Add a high suptitle for resistor info
        resistor_labels = ['10 Ω', '27 Ω', '100 Ω']
        fig.suptitle(f"Trial {i+1}: Resistor = {resistor_labels[i]}", fontsize=16, y=1.03)
        # Main plot (top)
        axs[0].plot(lengths, diff_times.T[i], 'o')
        axs[0].plot(lengths, linear(np.array(lengths), *p_opt_lin), 'r-')
        axs[0].errorbar(lengths, diff_times.T[i], yerr=u_time, fmt='o')
        axs[0].set_xlabel('Length (m)')
        axs[0].set_ylabel('Time Difference (s)')
        axs[0].grid()
        # Residuals (bottom)
        residuals = diff_times.T[i] - linear(np.array(lengths), *p_opt_lin)
        axs[1].plot(lengths, residuals, 'o')
        axs[1].errorbar(lengths, residuals, yerr=u_time, fmt='o')
        axs[1].axhline(0, color='red', linestyle='--')
        axs[1].set_xlabel('Length (m)')
        axs[1].set_ylabel('Residuals (s)')
        axs[1].grid()
    chisquaredof = np.sum(((diff_times.T[i] - linear(np.array(lengths), *p_opt_lin)) / u_time) ** 2)
    chiquaredofs.append(chisquaredof / (len(diff_times.T[i]) - 2))

print("speeds from slopes (m/s):", [1/s for s in slopes])
print("speed of light in vacuum (m/s):", 3e8)
print("speed compaired to speed of light in vacuum (%):", [abs((3e8 - (1/s)) / 3e8) * 100 for s in slopes])
print("Chi-squared per degree of freedom for each trial:", chiquaredofs)

    

#Compu_timee attenuation of each cable using the open circuit trials 

#zero each voltage not in the pulse
for i in range(4):
    baseline_open = np.median(vopen[i][:50])  # first 50 samples as baseline
    vopen[i] = vopen[i] - baseline_open


# Define skip samples individually for each cable length
skip_samples_open = [50, 60, 50, 50]  # adjust 2nd element if 30m needs more

threshold_open = 0.25  # volts

open_peak_times1 = []
open_peak_values1 = []
open_peak_times2 = []
open_peak_values2 = []

for i in range(4):
    trace = np.abs(vopen[i])  # absolu_timee value to catch negative pulses
    
    # --- First peak ---
    peaks1, props1 = find_peaks(trace, height=threshold_open)
    if len(peaks1) == 0:
        open_peak_times1.append(None)
        open_peak_values1.append(None)
        open_peak_times2.append(None)
        open_peak_values2.append(None)
        continue
    
    first_idx = peaks1[0]
    open_peak_times1.append(topen[i][first_idx])
    open_peak_values1.append(vopen[i][first_idx])
    
    # --- Second peak (skip samples specific to this cable) ---
    start_idx = first_idx + skip_samples_open[i]
    if start_idx >= len(trace):
        open_peak_times2.append(None)
        open_peak_values2.append(None)
        continue
    
    trace2 = trace[start_idx:]
    peaks2, props2 = find_peaks(trace2, height=threshold_open)
    
    if len(peaks2) == 0:
        open_peak_times2.append(None)
        open_peak_values2.append(None)
    else:
        second_idx = start_idx + peaks2[0]
        open_peak_times2.append(topen[i][second_idx])
        open_peak_values2.append(vopen[i][second_idx])

# Convert to arrays
open_peak_times = np.array([open_peak_times1, open_peak_times2]).T
open_peak_values = np.array([open_peak_values1, open_peak_values2]).T

# Calculate attenuation for each cable length
def attenuation_uncertainty(V1, V2, uV1, uV2):
    """
    Compu_timee uncertainty in attenuation (dB) from voltage uncertainties.
    
    Parameters
    ----------
    V1 : float
        First peak voltage (inpu_time)
    V2 : float
        Second peak voltage (ou_timepu_time)
    uV1 : float
        Uncertainty in V1
    uV2 : float
        Uncertainty in V2
    
    Returns
    -------
    uA : float
        Uncertainty in attenuation (dB)
    """
    if V1 == 0 or V2 == 0:
        return None
    factor = 20 / np.log(10)  # conversion factor for log10
    uA = factor * np.sqrt( (uV1 / V1)**2 + (uV2 / V2)**2 )
    return uA
attenuations = []
attenuation_uncertainties = []
for i in range(4):
    V1 = open_peak_values[i][0]
    V2 = open_peak_values[i][1]
    if V1 is not None and V2 is not None and V1 != 0:
        attenuation = 20 * np.log10(np.abs(V2) / np.abs(V1))
        attenuations.append(attenuation)
        attenuation_uncertainties.append(attenuation_uncertainty(V1, V2, uv, uv))
    else:
        attenuations.append(None)
print("Attenuations (dB) for each cable length (15m, 30m, 48m, 75m):", attenuations)
print("Uncertainties in attenuations (dB):", attenuation_uncertainties)

if PLOT_ATTENUATION:
    plt.figure()
    plt.plot(lengths, attenuations, 'o-')
    plt.xlabel('Cable Length')
    plt.ylabel('Attenuation (dB)')
    plt.title('Cable Attenuation vs Length')
    plt.errorbar(lengths, attenuations, yerr=attenuation_uncertainties, fmt='o')
    plt.grid()


#Code to determine characteristic impedance of the cables  

resistance = [10, 27, 100]  # known resistor values in ohms
def characteristic_impedance(V_reflect, V_incident, R_load):
    return -(V_reflect*R_load-V_incident*R_load)/(V_incident+V_reflect)


def Z0_uncertainty(Vi, Vr, R_load, uV):
    """
    Compu_timee uncertainty in characteristic impedance Z0 from voltage uncertainty.
    
    Parameters
    ----------
    Vi : float
        Incident voltage (signed, in volts)
    Vr : float
        Reflected voltage (signed, in volts)
    R_load : float
        Load resistance in ohms
    uV : float
        RMS voltage uncertainty (same for Vi and Vr)
        
    Returns
    -------
    uZ0 : float
        Uncertainty in Z0 (ohms)
    """
    dZ_dVi = R_load * (2 * Vr) / (Vi + Vr)**2
    dZ_dVr = -R_load * (2 * Vi) / (Vi + Vr)**2
    uZ0 = np.sqrt((dZ_dVi * uV)**2 + (dZ_dVr * uV)**2)
    return uZ0

max2_values = max2_values*peak_signs
characteristic_impedances = []
characteristic_impedances_uncertainty = []
for i in range(4):
    dummy_list=[]
    dummy_uncertainty_list=[]
    for j in range(3):
        dummy_list.append(characteristic_impedance(max2_values[i][j], max_values[i][j], resistance[j]))
        dummy_uncertainty_list.append(Z0_uncertainty(max_values[i][j], max2_values[i][j], resistance[j], uv))
    characteristic_impedances.append(np.mean(dummy_list))
    characteristic_impedances_uncertainty.append(np.mean(dummy_uncertainty_list))





print("Characteristic impedances (ohms) for each cable length (15m, 30m, 48m, 75m):", characteristic_impedances)
print("Uncertainties in characteristic impedances (ohms):", characteristic_impedances_uncertainty)




plt.show()









