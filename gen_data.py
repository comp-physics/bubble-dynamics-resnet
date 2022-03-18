# Created by Scott Sims 11/23/2021
# Rayleigh-Plesset Data Generation for Multiscale Hierarchical Time-Steppers with Residual Neural Networks

import os
import pdb
import numpy as np
import my_sound as ms
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
    globals()[str(key)] = D[key]
    print('{}: {}'.format(str(key), D[key]))
    # transforms key-names from dictionary into global variables, then assigns those variables their respective key-values

#=========================================================
# CALCULATE CONSTANTS used in simulation
#=========================================================
def calculate_constants():
    global u, R0, p0, pv, exponent, S, rho
    EPS = np.finfo(float).eps
    # time_constant = R0 * (rho / p0) ** (1 / 2)  # traditional normalization constant
    v = u / rho
    if (v < EPS):
        Re = np.inf
    else:
        Re = (R0 / v) * (p0 / rho) ** (1 / 2)
    S_hat = p0 * R0 / S
    Ca = (p0 - pv) / p0
    freq_natural = 3 * exponent * Ca + 2 * (3 * exponent - 1) / (S_hat * R0)
    freq_natural = np.sqrt(freq_natural / (R0 ** 2))
    T_natural = 1/freq_natural
    print(f"T_natural = {T_natural}")
    return T_natural, Ca, Re, S_hat, v
#--------------------------------------------------------
def chop_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) + 0)

#=========================================================
# ODE - Rayleigh-Plesset
#=========================================================
def ode_rp(t, y_init, sound):
    R, Rdot = y_init
    #----------------------------------------------------
    # CONSTANTS
    global Re, Ca, exponent, S_hat
    #---------------------------------------------------
    # pressure of sound wave
    Cp = sound.pressure(t)   #  = (p(t) - p0) / p0
    #---------------------------------------------------
    # SYSTEM OF ODEs, the 1st and 2nd derivatives
    y = np.zeros(2)
    y[0] = Rdot
    temp = -(3 / 2) * Rdot ** 2 - (4 / Re) * Rdot / R - (2 / S_hat) * (1 / R) + (2 / S_hat + Ca) * R ** (-3 * exponent) - Cp - Ca
    y[1] = temp / R
    #---------------------------------------------------
    return y

#=========================================================
# Constants
#=========================================================
global T_natural, Ca, Re, S_hat, v
T_natural, Ca, Re, S_hat, v = calculate_constants()
# freq_range = [ freq_min * freq_natural, freq_max * freq_natural ]
freq_range = [1/period_max, 1/period_min]
amp_range = [amp_min, amp_max]
#---------------------------------------------------
n_steps = np.int64(model_steps * 2**k_max)
print(f"n_steps = {n_steps}")
t_max = dt * n_steps
print(f"t_final = {t_max}")
t_space = np.linspace(0, t_max, n_steps + 1)
rel_tol = 1e-10
abs_tol = 1e-10
#=========================================================
# Directories and Paths
#=========================================================
data_folder = 'data_dt={}_steps={}_period={}-{}_amp={}-{}_train+val+test={}+{}+{}'.format(dt, n_steps, period_min, period_max, amp_min, amp_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

param_source = os.path.abspath(os.path.join(os.getcwd(), "parameters.yml"))
param_dest = os.path.abspath(os.path.join(data_dir, "parameters.yml"))
copyfile(param_source, param_dest)
#=========================================================
# Data Generation
#=========================================================
np.random.seed(2)
P = np.zeros(n_steps+1)
Pdot = np.zeros(n_steps+1)
#--------------------------------------------------------
# simulate training trials
train_data = np.zeros((n_train, n_steps + 1, n_inputs))
print('==============================')
print('generating training trials ...')
for i in range(n_train):
    #print(f"| train-{i} |")
    sound = ms.SoundWave(amp_range, freq_range, n_waves)
    # ----------------------------------
    # PRINT sum of amplitudes
    #temp = 0
    #for wave in sound.waves:
    #    temp += wave.amplitude
    #print(f"sum of amps after = {temp}")
    #----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, (p(t) - p0) / p0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    # PRINT average of |P(t)| and |Pdot(t)|
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    #print(f"mean(|Pdot(t)|) = {np.mean(np.abs(Pdot))}")
    #----------------------------------
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    train_data[i, :, :n_outputs] = sol.y.T
    train_data[i, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))

np.save(os.path.join(data_dir, 'train_D{}.npy'.format(2**k_max)), train_data)

for k in range(0, k_max):
    step_size = np.int64(2**k)
    slice_size = np.int64(model_steps * step_size)
    num_slices = np.int64(n_steps/slice_size)
    N = n_train * num_slices
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    for j in range(1, num_slices+1):
        idx_start = (j-1) * slice_size
        idx_end = j * slice_size
        idx_slices = np.array(list(range(j-1, N-num_slices+j, num_slices)))
        slice_data[idx_slices, :, :] = train_data[:, idx_start:idx_end+1, :]

    np.save(os.path.join(data_dir, 'train_D{}.npy'.format(step_size)), slice_data)

#--------------------------------------------------------
# simulate validation trials
val_data = np.zeros((n_val, n_steps + 1, n_inputs))
print('==============================')
print('generating validation trials ...')
for i in range(n_val):
    #print(f"| val-{i} |")
    sound = ms.SoundWave(amp_range, freq_range, n_waves)
    # ----------------------------------
    # PRINT sum of amplitudes
    #temp = 0
    #for wave in sound.waves:
    #    temp += wave.amplitude
    #print(f"sum of amplitudes = {temp}")
    #----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, (p(t) - p0) / p0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    # PRINT average of |P(t)| and |Pdot(t)|
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    #print(f"mean(|Pdot(t)|) = {np.mean(np.abs(Pdot))}")
    # ----------------------------------
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    val_data[i, :, :n_outputs] = sol.y.T
    val_data[i, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))

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
print('==============================')
print('generating testing trials ...')
for i in range(n_test):
    #print(f"| test-{i} |")
    sound = ms.SoundWave(amp_range, freq_range, n_waves)
    # ----------------------------------
    # PRINT sum of amplitudes
    #temp = 0
    #for wave in sound.waves:
    #    temp += wave.amplitude
    #print(f"sum of amplitudes = {temp}")
    #----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, (p(t) - p0) / p0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    # PRINT average of |P(t)| and |Pdot(t)|
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    #print(f"mean(|Pdot(t)|) = {np.mean(np.abs(Pdot))}")
    # ----------------------------------
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    test_data[i, :, :n_outputs] = sol.y.T
    test_data[i, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))

np.save(os.path.join(data_dir, 'test.npy'), test_data)
print('==============================')
print('data generation complete')
print('==============================')

#=========================================================
# Plot 3 Samples of Data (if num_plots=3)
#=========================================================
num_plots = 4
j_samples = np.int64(np.round(np.linspace(0, n_test-1, num_plots)))
figR, axsR = plt.subplots(num_plots, 1, figsize=(plot_x_dim, 1.1 * plot_y_dim * num_plots))
figP, axsP = plt.subplots(num_plots, 1, figsize=(plot_x_dim, 1.1 * plot_y_dim * num_plots))
figPdot, axsPdot = plt.subplots(num_plots, 1, figsize=(plot_x_dim, 1.1 * plot_y_dim * num_plots))

for idx in range(0, num_plots):
    j = j_samples[idx]
    R = test_data[j, :, 0]
    Rdot = test_data[j, :, 1]
    P = test_data[j, :, 2]
    Pdot = test_data[j, :, 3]
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    #print(f"mean(|Pdot(t)|) = {np.mean(np.abs(Pdot))}")
    parameters = f"$C_p(t)= \sum \ A_k \ \sin(2 \pi f_k (t - t_k))$ \n \n $R(t=0)=$ {R[0]} \n \n $ \dot{{R}} (t=0)=$ {Rdot[0]}\n "
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    #-------------------------------------------------------------------------------
    axsR[idx].plot(t_space, R, color='tab:blue', label='$R(t)$')
    #axsR[idx].legend(fontsize=legend_fontsize, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    axsR[idx].set_ylim(ymin=0.5*min(R))
    axsR[idx].set_xlabel('t / $t_0$',fontsize=x_label_fontsize)
    axsR[idx].set_ylabel('R / $R_0$',fontsize=y_label_fontsize)
    axsR[idx].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axsR[idx].text(0.0*t_max, max(R), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
    #-------------------------------------------------------------------------------
    axsP[idx].plot(t_space, P, color='tab:red', label='$P(t)$')
    axsP[idx].set_xlabel('t / $t_0$',fontsize=x_label_fontsize)
    axsP[idx].set_ylabel('$C_p(t)$',fontsize=y_label_fontsize)
    axsP[idx].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axsP[idx].text(0.0*t_max, max(P), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
    #-------------------------------------------------------------------------------
    axsPdot[idx].plot(t_space, Pdot, color='tab:red', label='$P(t)$')
    axsPdot[idx].set_xlabel('t / $t_0$',fontsize=x_label_fontsize)
    axsPdot[idx].set_ylabel('$ \\frac{d}{dt}C_p$',fontsize=y_label_fontsize)
    axsPdot[idx].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axsPdot[idx].text(0.0*t_max, max(Pdot), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)

file_fig_data_radius = os.path.abspath(os.path.join(data_dir, "test_data_sample_radius.png"))
file_fig_data_pressure = os.path.abspath(os.path.join(data_dir, "test_data_sample_pressure.png"))
file_fig_data_pressure_dot = os.path.abspath(os.path.join(data_dir, "test_data_pressure_dot.png"))

figR.tight_layout(pad=2.0)
figP.tight_layout(pad=2.0)
figPdot.tight_layout(pad=2.0)

figR.savefig(file_fig_data_radius, dpi=300)
figP.savefig(file_fig_data_pressure, dpi=300)
figPdot.savefig(file_fig_data_pressure_dot, dpi=300)

figR.show()
figP.show()
figPdot.show()