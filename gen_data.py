# Created by Scott Sims 11/23/2021
# Rayleigh-Plesset Data Generation for Multiscale Hierarchical Time-Steppers with Residual Neural Networks

import os
import numpy as np
# import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile

#=========================================================
# Input Arguments
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)

for key in D:
    globals() [str(key)] = D[key]
    # transforms key-names from dictionary into global variables, then assigns those variables their respective key-values

#=========================================================
# Constants
#=========================================================
n_steps = np.int64(model_steps * 2**k_max)
t_max = dt * n_steps
t_space = np.linspace(0, t_max, n_steps + 1)
rel_tol = 1e-10
abs_tol = 1e-10
EPS = np.finfo(float).eps
#------------------
global time_constant
time_constant = R0 * (rho / p0)**(1/2)
global Re
v = u / rho
if (v < EPS):
    Re = np.inf
else:
    Re = (R0 / v) * (p0 / rho) ** (1 / 2)
global S_hat
S_hat = p0 * R0 / S
global Ca
Ca = (p0 - pv) / p0
# w2 = ( 3*k*(p0-pv) / (R0**2) + 2 * (3*k - 1) * S / (R0**3) - 8 * (u**2) / (rho*R0**4) )/rho
# tau = Rref * (rho / p0) ** (1 / 2)

#=========================================================
# Directories and Paths
#=========================================================
data_folder = 'data_dt={}_steps={}_freq={}-{}_amp={}-{}_train-val-test={}-{}-{}'.format(dt, n_steps, freq_min, freq_max, amp_min, amp_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

param_source = os.path.abspath(os.path.join(os.getcwd(), "parameters.yml"))
param_dest = os.path.abspath(os.path.join(data_dir, "parameters.yml"))
copyfile(param_source, param_dest)

#=========================================================
# Pressure Simulation
#=========================================================
def pressure(t, amp, freq):
    # w = np.random.uniform(w_min, w_max, size=None)
    # normalized pressure (p(t) - p0)/p0
    return amp * np.sin(freq * 2 * np.pi * t)

#=========================================================
# ODE - Rayleigh-Plesset
#=========================================================
def ode_rp(t, y_init, freq):
    R, Rdot = y_init
    #----------------------------------------------------
    # CONSTANTS
    global p0, pv, Rref, S, u, rho
    global EPS, time_constant, Re, Ca, exponent, S_hat
    #---------------------------------------------------
    # PRESSURE WAVE (function of real time, not simulation time)
        # archived line: t_sound = t * Rref * (rho / p0) ** (1/2)
    Cp = pressure(t, amp=0.5*p0, freq=freq) / p0  # in the paper, (p(t) - p0) / p0
    #---------------------------------------------------
    # SYSTEM OF ODEs, the 1st and 2nd derivatives
    dRdt = np.zeros(2)
    dRdt[0] = Rdot
    temp = -(3 / 2) * Rdot ** 2 - (4 / Re) * Rdot / R - (2 / S_hat) * (1 / R) + (2 / S_hat + Ca) * R ** (
                -3 * exponent) - Cp - Ca
    dRdt[1] = temp / R
    #---------------------------------------------------
    return dRdt

#=========================================================
# Data Generation
#=========================================================
np.random.seed(2)
#--------------------------------------------------------
# simulate training trials
train_data = np.zeros((n_train, n_steps + 1, n_inputs))
P = np.zeros(n_steps+1)
print('generating training trials ...')
freq_samples = np.random.uniform(freq_min, freq_max, n_train)
for i in range(n_train):
    freq = freq_samples[i]
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = pressure(t, amp=0.6 * p0, freq=freq) / p0  # in the paper, (p(t) - p0) / p0
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(freq,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    train_data[i, :, :n_outputs] = sol.y.T
    train_data[i, :, n_outputs:] = P.reshape(n_steps+1,1)

np.save(os.path.join(data_dir, 'train_D{}.npy'.format(2**k_max)), train_data)

for k in range(0, k_max):
    step_size = np.int64(2**k)
    slice_size = np.int64(model_steps * step_size)
    num_slices = np.int64(n_steps/slice_size)
    N = n_train * num_slices
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    print('num_slices = {}'.format(num_slices))
    print('slice_size = {}'.format(slice_size))
    print('N = {}'.format(N))
    print('step_size = {}'.format(step_size))
    for j in range(1, num_slices+1):
        idx_start = (j-1) * slice_size
        idx_end = j * slice_size
        idx_slices = np.array(list(range(j-1, N-num_slices+j, num_slices)))
        slice_data[idx_slices, :, :] = train_data[:, idx_start:idx_end+1, :]

    np.save(os.path.join(data_dir, 'train_D{}.npy'.format(step_size)), slice_data)

#--------------------------------------------------------
# simulate validation trials
val_data = np.zeros((n_val, n_steps + 1, n_inputs))
print('generating validation trials ...')
freq_samples = np.random.uniform(freq_min, freq_max, n_val)
for i in range(n_val):
    freq = freq_samples[i]
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = pressure(t, amp=0.6 * p0, freq=freq) / p0  # in the paper, (p(t) - p0) / p0
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(freq,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    val_data[i, :, :n_outputs] = sol.y.T
    val_data[i, :, n_outputs:] = P.reshape(n_steps+1,1)

np.save(os.path.join(data_dir, 'val_D{}.npy'.format(2**k_max)), val_data)

for k in range(0, k_max):
    step_size = np.int64(2**k)
    slice_size = np.int64(model_steps * step_size)
    num_slices = np.int64(n_steps/slice_size)
    N = n_val * num_slices
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    for j in range(1, num_slices+1):
        idx_start = (j-1) * slice_size
        idx_end = j * slice_size
        idx_slices = np.array(list(range(j-1, N-num_slices+j, num_slices)))
        slice_data[idx_slices, :, :] = val_data[:, idx_start:idx_end+1, :]

    np.save(os.path.join(data_dir, 'val_D{}.npy'.format(step_size)), slice_data)
#--------------------------------------------------------
# simulate test trials
test_data = np.zeros((n_test, n_steps + 1, n_inputs))
freq_samples = np.linspace(freq_min, freq_max, n_test)
amp_samples = np.linspace(amp_min, amp_max, n_test)
print('generating testing trials ...')

for i in range(n_test):
    freq = freq_samples[i]
    amp = amp_samples[i]
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = pressure(t, amp=0.6 * p0, freq=freq) / p0  # in the paper, (p(t) - p0) / p0
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(freq,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    test_data[i, :, :n_outputs] = sol.y.T
    test_data[i, :, n_outputs:] = P.reshape(n_steps+1,1)

np.save(os.path.join(data_dir, 'test.npy'), test_data)

print('data generation complete')


#=========================================================
# Plot 3 Samples of Data (if num_plots=3)
#=========================================================
num_plots = 4
j_samples = np.int64(np.round(np.linspace(0, n_test-1, num_plots)))
fig, axs = plt.subplots(num_plots, 1, figsize=(plot_x_dim, 1.1*plot_y_dim*num_plots))

for it in range(0, num_plots):
    j = j_samples[it]
    R_t = test_data[j, :, 0]
    Rdot_t = test_data[j, :, 1]
    P_t = test_data[j, :, 2]
    axs[it].plot(t_space, P_t, color='tab:red', label='$P(t)$')
    axs[it].plot(t_space, R_t, color='tab:blue', label='$R(t)$')
    axs[it].legend(fontsize=legend_fontsize, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    axs[it].set_xlabel('t / $t_0$',fontsize=x_label_fontsize)
    axs[it].set_ylabel('R / $R_0$',fontsize=y_label_fontsize)
    axs[it].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    parameters = '$P(t)=A \sin(2 \pi f \ t)$ \n $R(t=0)=$ {:.3f}\n $ \dot{{R}} (t=0)=$ {:.3f}\n'.format(R_t[0], Rdot_t[0])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    axs[it].text(0.6*t_max, max(R_t), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)

fig.tight_layout(pad=3.0)
plt.show()
file_fig_data = os.path.abspath(os.path.join(data_dir, "data_sample.png"))
plt.savefig(file_fig_data, dpi=300)
