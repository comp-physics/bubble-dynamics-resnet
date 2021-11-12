# Created by Scott Sims 11/09/2021
# Rayleigh-Plesset Data Generation for Multiscale Hierarchical Time-Steppers with Residual Neural Networks

import os
import numpy as np
# import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile

#=========================================================
# Command Line Arguments
#=========================================================
with open("parameters.yml", 'r') as f:
    dictionary = yaml.safe_load(f)#, Loader=yaml.FullLoader)
#---------------------------------------
system = dictionary['system']
dt = dictionary['dt']
k_max = dictionary['k_max']
steps_min = dictionary['steps_min']
global u
u = dictionary['u']
P_min = dictionary['P_min']
P_max = dictionary['P_max']
R_min = dictionary['R_min']
R_max = dictionary['R_max']
R_test = dictionary['R_test']
Rdot_min = dictionary['Rdot_min']
Rdot_max = dictionary['Rdot_max']
Rdot_test = dictionary['Rdot_test']
n_train = dictionary['n_train']
n_val = dictionary['n_val']
n_test = dictionary['n_test']
num_layers = dictionary['num_layers']
layer_size = dictionary['layer_size']
num_inputs = dictionary['num_inputs']

#=========================================================
# Constants
#=========================================================
n_steps = np.int64(steps_min * 2**k_max)
tmax = dt*n_steps
t = np.linspace(0, tmax, n_steps + 1)
rel_tol = 1e-10
abs_tol = 1e-10
global EPS
EPS = np.finfo(float).eps
# w2 = ( 3*k*(P0-Pv) / (R0**2) + 2 * (3*k - 1) * S / (R0**3) - 8 * (u**2) / (rho*R0**4) )/rho
# tau = Rref * (rho / P0) ** (1 / 2)

#=========================================================
# Directories and Paths
#=========================================================
data_folder = 'data_dt={}_steps={}_P={}-{}_R={}-{}_(train|val|test)=({}|{}|{}).pt'.format(dt, n_steps, P_min, P_max, R_min, R_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

param_source = os.path.abspath(os.path.join(os.getcwd(), "parameters.yml"))
param_dest = os.path.abspath(os.path.join(data_dir, "parameters.yml"))
copyfile(param_source, param_dest)


#=========================================================
# ODE - Rayleigh-Plesset
#=========================================================
def ode_rp(t, R):
    # R[0] = P(t)
    # R[1] = R(t)
    # R[2] = R'(t)
    #----------------------------------------------------
    # CONSTANTS
    global u, EPS
    R0 = 100e-6  # m                 50.00e-6
    # u = 8.63e-4  # Pa * s      1.00e-3
    Pv = 3.538e3  # Pa             2.34e3
    exponent = 1.4
    P0 = 0.8e5  # Pa             3.28e4
    S = 7.17e-2  # N / m          7.28e-2
    rho = 9.963e2  # kg / (m ^ 3)   9.98e2
    Rref = R0
    #----------------------------------------------------
    # PARAMETERS
    v = u / rho
    S_hat = P0 * Rref / S
    Ca = (P0 - Pv) / P0
    if (v < EPS):
        Re = np.inf
    else:
        Re = (Rref / v) * (P0 / rho) ** (1 / 2)
    #---------------------------------------------------
    # SYSTEM OF ODEs, the 1st and 2nd derivatives
    Cp = R[0] - 1
    dRdt = np.zeros(3)
    dRdt[0] = 0
    dRdt[1] = R[2]
    total = -(3 / 2) * R[2] ** 2 - (4 / Re) * R[2] / R[1] - (2 / S_hat) * (1 / R[1]) + (2 / S_hat + Ca) * (R[1]) ** (
                -3 * exponent) - Cp - Ca
    dRdt[2] = total / R[1]
    #---------------------------------------------------
    return dRdt

#=========================================================
# Data Generation
#=========================================================
# simulation parameters
np.random.seed(2)

# simulate training trials
train_data = np.zeros((n_train, n_steps + 1, num_inputs))
print('generating training trials ...')
for i in range(n_train):
    P = np.random.uniform(P_min, P_max)
    R = np.random.uniform(R_min, R_max)
    Rdot = np.random.uniform(Rdot_min, Rdot_max)
    y_init = [P, R, Rdot]
    sol = solve_ivp(ode_rp, t_span=[0, tmax], y0=y_init, t_eval=t, method='RK45', rtol=rel_tol, atol=abs_tol)
    train_data[i, :, :] = sol.y.T

# simulate validation trials
val_data = np.zeros((n_val, n_steps + 1, num_inputs))
print('generating validation trials ...')
for i in range(n_val):
    P = np.random.uniform(P_min, P_max)
    R = np.random.uniform(R_min, R_max)
    Rdot = np.random.uniform(Rdot_min, Rdot_max)
    y_init = [P, R, Rdot]
    sol = solve_ivp(ode_rp, t_span=[0, tmax], y0=y_init, t_eval=t, method='RK45', rtol=rel_tol, atol=abs_tol)
    val_data[i, :, :] = sol.y.T

# simulate test trials
test_data = np.zeros((n_test, n_steps + 1, num_inputs))
print('generating testing trials ...')
P_samples = np.linspace(P_min, P_max, n_test)
for i in range(n_test):
    P_test = P_samples[i]
    y_init = [P_test, R_test, Rdot_test]
    sol = solve_ivp(ode_rp, t_span=[0, tmax], y0=y_init, t_eval=t, method='RK45', rtol=rel_tol, atol=abs_tol)
    test_data[i, :, :] = sol.y.T

#=========================================================
# Save Data
#=========================================================
np.save(os.path.join(data_dir, 'train.npy'), train_data)
np.save(os.path.join(data_dir, 'val.npy'), val_data)
np.save(os.path.join(data_dir, 'test.npy'), test_data)
print('data generation complete')
#=========================================================
# Plot 3 Samples of Data (if num_plots=3)
#=========================================================
num_plots = 3
j_samples = np.int64(np.round(np.linspace(0, n_test-1, num_plots)))
fig, axs = plt.subplots(num_plots, 1, figsize=(30, 10*3*num_plots))


for it in range(0, num_plots):
    j = j_samples[it]
    P_t = test_data[j, :, 0]
    R_t = test_data[j, :, 1]
    Rdot_t = test_data[j, :, 2]
    axs[it].plot(t, P_t, color='tab:red', label='$P(t)$')
    axs[it].plot(t, R_t, color='tab:blue', label='$R(t)$')
    # plt.plot(t, Rdot_t, color='tab:green', label='$\dot{R}(t)$')
    axs[it].legend(fontsize=30, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    axs[it].set_xlabel('t / $t_0$')
    axs[it].set_ylabel('R / $R_0$')
    parameters = '$P(t=0)=$ {0:.3f}\n $R(t=0)=$ {1:.3f}\n $ \dot{{R}} (t=0)=$ {2:.3f}\n'.format(P_t[0], R_t[0], Rdot_t[0])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    axs[it].text(0.6*tmax, max(R_t), parameters, fontsize=14, verticalalignment='top', bbox=props)

plt.show()
file_fig_data = os.path.abspath(os.path.join(data_dir, "data_sample.png"))
plt.savefig(file_fig_data, dpi=300)
