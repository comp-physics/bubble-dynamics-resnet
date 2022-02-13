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
# Constants
#=========================================================
def chop_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x))))+0)
#========================================================
global EPS, S_hat, v, Re, time_constant, Ca
EPS = np.finfo(float).eps
v = u / rho
if (v < EPS):
    Re = np.inf
else:
    Re = (R0 / v) * (p0 / rho) ** (1 / 2)
S_hat = p0 * R0 / S
Ca = (p0 - pv) / p0
#--------------------------------------------------
freq_natural = 3 * exponent * Ca + 2 * (3 * exponent - 1) / (S_hat * R0)
freq_natural = np.sqrt(freq_natural / (R0 ** 2))
T_natural = 1 / freq_natural
print(f"T_natural = {T_natural}")
dt = chop_to_1(T_natural)
print(f"dt = {dt}")
freq_range = [ freq_min * freq_natural, freq_max * freq_natural ]
amp_range = [amp_min, amp_max]
#---------------------------------------------------
n_steps = np.int64(model_steps * 2**k_max)
print(f"n_steps = {n_steps}")
t_max = dt * n_steps
t_space = np.linspace(0, t_max, n_steps + 1)
rel_tol = 1e-10
abs_tol = 1e-10
#=========================================================
# Directories and Paths
#=========================================================
data_folder = 'data_dt={}_steps={}_freq={}-{}_amp={}-{}_train+val+test={}+{}+{}'.format(dt, n_steps, freq_min, freq_max, amp_min, amp_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

param_source = os.path.abspath(os.path.join(os.getcwd(), "parameters.yml"))
param_dest = os.path.abspath(os.path.join(data_dir, "parameters.yml"))
copyfile(param_source, param_dest)

#=========================================================
# ODE - Rayleigh-Plesset
#=========================================================
def ode_rp(t, y_init, sound):
    R, Rdot = y_init
    #----------------------------------------------------
    # CONSTANTS
    global p0, pv, Rref, S, u, rho
    global EPS, time_constant, Re, Ca, exponent, S_hat
    #---------------------------------------------------
    # PRESSURE WAVE (function of real time, not simulation time)
        # archived line: t_sound = t * Rref * (rho / p0) ** (1/2)
    Cp = sound.pressure(t)   # in the paper, (p(t) - p0) / p0
    #---------------------------------------------------
    # SYSTEM OF ODEs, the 1st and 2nd derivatives
    dRdt = np.zeros(2)
    dRdt[0] = Rdot
    temp = -(3 / 2) * Rdot ** 2 - (4 / Re) * Rdot / R - (2 / S_hat) * (1 / R) + (2 / S_hat + Ca) * R ** (-3 * exponent) - Cp - Ca
    dRdt[1] = temp / R
    #---------------------------------------------------
    return dRdt

#=========================================================
# Data Generation
#=========================================================
np.random.seed(2)
P = np.zeros(n_steps+1)
Pdot = P
#--------------------------------------------------------
# simulate training trials
train_data = np.zeros((n_train, n_steps + 1, n_inputs))
print('generating training trials ...')
for i in range(n_train):
    sound_wave = ms.SoundWave(amp_range, freq_range, n_waves)
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound_wave.pressure(t)  # in the paper, (p(t) - p0) / p0
        Pdot[j] = sound_wave.pressure_dot(t)
    #print(f"num waves = {len(sound_wave.sound)}")
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(sound_wave,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    train_data[i, :, :n_outputs] = sol.y.T
    train_data[i, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))

np.save(os.path.join(data_dir, 'train_D{}.npy'.format(2**k_max)), train_data)

for k in range(0, k_max):
    step_size = np.int64(2**k)
    slice_size = np.int64(model_steps * step_size)
    num_slices = np.int64(n_steps/slice_size)
    N = n_train * num_slices
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    #print('num_slices = {}'.format(num_slices))
    #print('slice_size = {}'.format(slice_size))
    #print('N = {}'.format(N))
    #print('step_size = {}'.format(step_size))
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
for i in range(n_val):
    sound_waves = ms.SoundWave(amp_range, freq_range, n_waves)
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound_waves.pressure(t)  # in the paper, (p(t) - p0) / p0
        Pdot[j] = sound_waves.pressure_dot(t)
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(sound_waves,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
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
print('generating testing trials ...')
for i in range(n_test):
    sound_waves = ms.SoundWave(amp_range, freq_range, n_waves)
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound_waves.pressure(t)  # in the paper, (p(t) - p0) / p0
        Pdot[j] = sound_waves.pressure_dot(t)
    #print(f"mean(|P(t)|) = {np.mean(np.abs(P))}")
    y_init = [R_init, Rdot_init]
    sol = solve_ivp(ode_rp, t_span=[0, t_max], y0=y_init, args=(sound_waves,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    test_data[i, :, :n_outputs] = sol.y.T
    test_data[i, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))

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
    #axs[it].plot(t_space, P_t, color='tab:red', label='$P(t)$')
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
