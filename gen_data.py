# ## updated by Scott Sims 03/18/2022
# Rayleigh-Plesset Data Generation for Multiscale Hierarchical Time-Steppers with Residual Neural Networks

import os
import pdb
import numpy as np
import my_sound
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile

#=========================================================
# Read input arguments from YAML file: paramters.yml
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)
for key in D:
    globals()[str(key)] = D[key]
    print('{}: {}'.format(str(key), D[key]))
    # transforms key-names from dictionary into global variables, then assigns those variables their respective key-values
#=========================================================
# Function: Calculate physical constants with normalized units of time and length, "simulation units"
#=========================================================
def calculate_constants():
    global u, R0, P0, Pv, exponent, S, rho
    EPS = np.finfo(float).eps
    time_constant = R0 * (rho / P0) ** (1 / 2)  # traditional normalization constant
    v = u / rho
    if (v < EPS):
        Re = np.inf
    else:
        Re = (R0 / v) * (P0 / rho) ** (1 / 2)
    S_hat = P0 * R0 / S
    Ca = (P0 - Pv) / P0
    freq_natural = np.sqrt( 3 * exponent * Ca + 2 * (3 * exponent - 1) / S_hat )
    T_natural = 1/freq_natural
    print(f"time-constant = {time_constant} sec")
    print(f"Re = {np.round(Re, decimals=3)}")
    print(f"S_hat = {np.round(S_hat, decimals=3)}")
    print(f"Ca = {np.round(Ca, decimals=3)}")
    print(f"freq_natural = {np.round(freq_natural, decimals=3)} time-units^[-1]")
    print(f"T_natural = {np.round(T_natural, decimals=3)} time-units")
    return T_natural, Ca, Re, S_hat, v
#--------------------------------------------------------
def chop_to_one_digit(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) + 0)
#=========================================================
# Function: Calculates number of steps for total duration 
#=========================================================
def calculate_steps():
    global n_periods, period_max, dt, model_steps, k_max
    max_steps = model_steps * 2**k_max # max slice size
    return np.int64( max_steps * np.ceil( n_periods * period_max / (dt * max_steps)) )
#=========================================================
# Function: ODE - Rayleigh-Plesset
#=========================================================
def ode_rp(t, y_init, sound):
    R, Rdot = y_init
    #----------------------------------------------------
    # CONSTANTS
    global Re, Ca, exponent, S_hat
    #---------------------------------------------------
    # pressure of sound wave
    Cp = sound.pressure(t)   #  = epsilon(t) = (p(t) - P0) / P0
    #---------------------------------------------------
    # SYSTEM OF ODEs, the 1st and 2nd derivatives
    y = np.zeros(2)
    y[0] = Rdot
    y[1] = -(3 / 2) * Rdot ** 2 - (4 / Re) * Rdot / R - (2 / S_hat) * (1 / R) + (2 / S_hat + Ca) * R ** (-3 * exponent) - Cp - Ca
    y[1] = y[1] / R
    #---------------------------------------------------
    return y

#=========================================================
# Calculate Constants
#=========================================================
global T_natural, Ca, Re, S_hat, v
T_natural, Ca, Re, S_hat, v = calculate_constants()
freq_range = [1/period_max, 1/period_min]
amp_range = [amp_min, amp_max]
n_steps = calculate_steps()
#---------------------------------------------------
print(f"n_steps = {n_steps}")
t_final = dt * n_steps
print(f"t_final = {t_final}")
t_space = np.arange(0,n_steps+1,dt)
#---------------------------------------------------
# PAUSE script
#---------------------------------------------------
print('TO QUIT, PRESS [q] THEN [ENTER]')
print('OTHERWISE, PRESS [c] THEN [ENTER]')
pdb.set_trace()
#=========================================================
# Directories and Paths
#=========================================================
data_folder = f"data_dt={dt}_n-steps={n_steps}_m-steps={model_steps}_k-max={k_max}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_n-waves={n_waves}_train+val+test={n_train}+{n_val}+{n_test}"
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
#--------------------------------
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
# simulate training duration
#--------------------------------------------------------
train_data = np.zeros((n_train, n_steps + 1, n_inputs))
print('==============================')
print('generating training trials ...')
y_init = [R_init, Rdot_init]
for idx in range(n_train):
    sound = my_sound.SoundWave(amp_range, freq_range, n_waves)
    #----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, eps(t) = (p(t) - P0) / P0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    sol = solve_ivp(ode_rp, t_span=[0, t_final], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    train_data[idx, :, :n_outputs] = sol.y.T
    train_data[idx, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))
# save training samples to file
np.save(os.path.join(data_dir, f"train.npy"), train_data)
# slice each training sample into smaller samples for each time-stepper
for k in range(0, k_max):
    step_size = np.int64(2**k)
    slice_size = np.int64(model_steps * step_size)
    num_slices = np.int64(n_steps/slice_size)
    N = n_train * num_slices
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    #------------------------------------
    for j in range(1, num_slices+1):
        idx_start = (j-1) * slice_size
        idx_end = j * slice_size
        idx_slices = np.array(list(range(j-1, j+N-num_slices, num_slices)))
        slice_data[idx_slices, :, :] = train_data[:, idx_start:idx_end+1, :]
    # save training slices to file
    np.save(os.path.join(data_dir, f"train_D{step_size}.npy"), slice_data)
#--------------------------------------------------------
# simulate validation trials
#--------------------------------------------------------
val_data = np.zeros((n_val, n_steps + 1, n_inputs))
print('==============================')
print('generating validation trials ...')
for idx in range(n_val):
    sound = my_sound.SoundWave(amp_range, freq_range, n_waves)
    # ----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, = (p(t) - P0) / P0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    sol = solve_ivp(ode_rp, t_span=[0, t_final], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    val_data[idx, :, :n_outputs] = sol.y.T
    val_data[idx, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))
# save validation samples to file
np.save(os.path.join(data_dir, f"val.npy"), val_data)
# slice samples for each time-stepper
for k in range(0, k_max):
    step_size = np.int64(2**k)
    slice_size = np.int64(model_steps * step_size)
    num_slices = np.int64(n_steps/slice_size)
    N = n_val * num_slices
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    #------------------------------------
    for j in range(1, num_slices+1):
        idx_start = (j-1) * slice_size
        idx_end = j * slice_size
        idx_slices = np.array(list(range(j-1, j+N-num_slices, num_slices)))
        slice_data[idx_slices, :, :] = val_data[:, idx_start:idx_end+1, :]
    # save validation samples to file
    np.save(os.path.join(data_dir, f"val_D{step_size}.npy"), slice_data)
#--------------------------------------------------------
# simulate test trials
#--------------------------------------------------------
test_data = np.zeros((n_test, n_steps + 1, n_inputs))
print('==============================')
print('generating testing trials ...')
for idx in range(n_test):
    #print(f"| test-{i} |")
    sound = my_sound.SoundWave(amp_range, freq_range, n_waves)
    # ----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, = (p(t) - P0) / P0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    sol = solve_ivp(ode_rp, t_span=[0, t_final], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    test_data[idx, :, :n_outputs] = sol.y.T
    test_data[idx, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))

np.save(os.path.join(data_dir, f"test.npy"), test_data)
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
    axsR[idx].text(0.0*t_final, max(R), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
    #-------------------------------------------------------------------------------
    axsP[idx].plot(t_space, P, color='tab:red', label='$P(t)$')
    axsP[idx].set_xlabel('t / $t_0$',fontsize=x_label_fontsize)
    axsP[idx].set_ylabel('$C_p(t)$',fontsize=y_label_fontsize)
    axsP[idx].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axsP[idx].text(0.0*t_final, max(P), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
    #-------------------------------------------------------------------------------
    axsPdot[idx].plot(t_space, Pdot, color='tab:red', label='$P(t)$')
    axsPdot[idx].set_xlabel('t / $t_0$',fontsize=x_label_fontsize)
    axsPdot[idx].set_ylabel('$ \\frac{d}{dt}C_p$',fontsize=y_label_fontsize)
    axsPdot[idx].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axsPdot[idx].text(0.0*t_final, max(Pdot), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)

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