"""
This python file outputs the data and attempts to fit a damped cosine.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def damped_cosine(t, theta0, tau, T, phi0):
    """
    theta(t) = theta0 * exp(-t/tau) * cos(2*pi*t/T + phi0)
    """
    return theta0 * np.exp(-t/tau) * np.cos(2*np.pi*t/T + phi0)

    
def plot_fit(p_opt, label, color):
    """
    Helper to plot the fitted function
    """
    t_fit = np.linspace(x_values[0], x_values[-1], 1000)
    plt.plot(t_fit, damped_cosine(t_fit, *p_opt),
             label=label, lw=2, color=color)

time, theta = np.loadtxt(
    'data/angleDegreesPendulum.txt',
    delimiter=',', skiprows=1000,  # skip 3 lines: blank, header, column names
    unpack=True
)

x_values = time
x_unc = None

y_values = theta
y_unc = None

print(x_values[0])
print(y_values[0])

# p0 values guess
theta0_guess = -31          # start amplitude
tau_guess = 120             # based on starting and ending
T_guess = 1.5               # guess period
phi0_guess = 0.0            # guess 0

p0 = (theta0_guess, tau_guess, T_guess, phi0_guess)

# bounds = ([ -70, 80, 1, -2*np.pi],
#           [  -30, 150, 2,  2*np.pi])

p_opt, p_cov = curve_fit(damped_cosine, x_values, y_values, p0=p0, maxfev=100000)
p_var = np.diag(p_cov)
sd = np.sqrt(p_var)

print(f"theta0 = {p_opt[0]} +- {sd[0]}")


# T = (2*np.pi) / p_opt[1]
print(f"tau = {p_opt[1]} +- {sd[1]}")
# print(f"So, T = 2pi / k = {T}")

print(f"T = {p_opt[2]} +- {sd[2]}")

print(f"phi0 = {p_opt[3]} +- {sd[3]}")


plt.figure(1)
plt.title("Position on time of mass")
plt.plot(x_values, y_values, ls='', marker='o', label='Data', markersize=2)
# plot_fit(p_opt, "Fit", "green")

# plot_fit((-50, 120, 1.302, -3), "Manual guess", "orange")

plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.legend()
plt.show()
