"""
This python file is to complete check in 1
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def linear(x, m, c):
    """
    y = mx + c
    """
    return m*x + c

def plot_linear(p_opt, label, color):
    """
    Helper to plot the fitted function
    """
    x_linspace = np.linspace(x_values[0], x_values[-1], 1000)
    plt.plot(x_linspace, linear(x_linspace, *p_opt),
             label=label, lw=2, color=color)


time, theta = np.loadtxt(
    'data/angleDegreesPendulum.txt',
    delimiter=',', skiprows=1000,  # skip 3 lines: blank, header, column names
    unpack=True
)

peaks, _= find_peaks(theta)
max_times = [time[i] for i in peaks if theta[i] > 0]
max_thetas = [theta[i] for i in peaks if theta[i] > 0]

time_periods = [max_times[i+1] - max_times[i] for i in range(len(max_times) - 1)]
# print(time_periods)

# ====================================

# plt.figure(1)
# plt.title("Position on time of mass")
# plt.plot(time, theta, ls='', marker='o', label='Data', markersize=2)
# plt.plot(max_times, max_thetas, ls='', marker='o', label='Max', markersize=2)
# plt.xlabel("Time (s)")
# plt.ylabel("Angle (degrees)")
# plt.legend()
# plt.show()
# =====================================

x_values = max_thetas[:-1]
x_unc = None #[5] * len(max_thetas - 1)

y_values = time_periods
y_unc = [0.016] * len(time_periods)


p_opt, p_cov = curve_fit(linear, x_values, y_values, sigma=y_unc, absolute_sigma=True)
p_var = np.diag(p_cov)
sd = np.sqrt(p_var)

print(f"m = {p_opt[0]} +- {sd[0]}")
print(f"c = {p_opt[1]} +- {sd[1]}")

plt.figure(2)
plt.title("Position on time of mass")
plt.errorbar(max_thetas[:-1], time_periods, yerr=y_unc ,ls='', marker='o', label='Data', markersize=2)
plot_linear(p_opt, "Fit", "green")
plt.xlabel("Theta (degrees)")
plt.ylabel("Time periods (s)")
plt.legend()
plt.show()
