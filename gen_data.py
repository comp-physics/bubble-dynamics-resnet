# ## adapted by Scott Sims 05/11/2022
# Rayleigh-Plesset Data Generation for Multiscale Time-Steppers with Residual Neural Networks

#----------------------------------
# universal packages
#----------------------------------
import os
import pdb
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile
#----------------------------------
# local packages
#----------------------------------
import bubble_methods as bub

#=========================================================
# Read input arguments from YAML file: paramters.yml
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)
for key in D:
    globals()[str(key)] = D[key]
    print('{}: {}'.format(str(key), D[key]))
    # transforms key-names from dictionary into global variables, then assigns those variables their respective key-values
#--------------------------------------------------------
def chop_to_one_digit(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) + 0)

#=========================================================
bubble = bub.Bubble(D)
#----------------------------------------------------
# PRINT CHARACTERISTICS
#----------------------------------------------------
print(f"--------------- SUMMARY -----------------")
print(f"equlibrium radius = {bubble.R_eq*1e6: .2f} micro-meters")
print(f"time-constant = {bubble.time_constant: .3e} sec")
print(f"------- NONDIMENSIONAL PARAMETERS -------")
print(f"Re = {bubble.reynolds: .2e}")
print(f"S_hat = {bubble.S: .4e}")
print(f"Ca = {bubble.Ca: .4f}")
print(f"natural frequency = {bubble.freq_natural: .4f}")
print(f"natural period = {bubble.T_natural: .4f}")
print(f"-----------------------------------------")
#=========================================================
# Initialize Intervals of Frequencies and Periods
#=========================================================
freq_range = [1/period_max, 1/period_min]
amp_range = [amp_min, amp_max]
n_steps = bub.get_num_steps(dt, model_steps, k_max, period_min, n_periods)
#---------------------------------------------------
print(f"n_steps = {n_steps}")
t_final = dt * (n_steps)
print(f"t_final = {t_final}")
t_space = np.arange(0,t_final+dt,dt)[0:n_steps+1]
print(f"t_space = {t_space[0:3]}...{t_space[n_steps-2:n_steps+1]}")
#---------------------------------------------------
# PAUSE script
#---------------------------------------------------
#print('TO QUIT, PRESS [q] THEN [ENTER]')
#print('OTHERWISE, PRESS [c] THEN [ENTER]')
#pdb.set_trace()


#=========================================================
# Directories and Paths
#=========================================================
data_folder = f"data_dt={dt}_n-steps={n_steps}_m-steps={model_steps}_k={k_min}-{k_max}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_n-waves={n_waves}_train+val+test={n_train}+{n_val}+{n_test}"
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
np.random.seed(3)
P = np.zeros(n_steps+1)
Pdot = np.zeros(n_steps+1)
#--------------------------------------------------------
# simulate training duration
#--------------------------------------------------------
train_data = np.zeros((n_train, n_steps + 1, n_inputs))
#time_samples = np.random.uniform(t_final/3,2*t_final/3, n_train)
print('==============================')
print('generating training trials ...')
y_init = [R_init, Rdot_init]
for idx in range(n_train):
    #----------------------------------
    sound = bub.SoundWave(amp_range, freq_range, n_waves)
    root = fsolve(sound.pressure, period_max, xtol=1e-6)
    t0 = root[0]
    for j in range(n_waves):
        sound.waves[j].phase = sound.waves[j].phase + (2*np.pi*sound.waves[j].freq)*t0
    #----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, eps(t) = (p(t) - P0) / P0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    sol = solve_ivp(bubble.rhs_rp, t_span=[0.0, t_final+dt], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    train_data[idx, 0, :n_outputs] = y_init
    train_data[idx, 1:, :n_outputs] = sol.y.T[0:n_steps]
    train_data[idx, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))
# save training samples to file
np.save(os.path.join(data_dir, f"train.npy"), train_data)
# slice each training sample into smaller samples for each time-stepper
for k in range(k_min, k_max+1):
    step_size = np.int64(np.round(2**k))
    slice_size = np.int64(np.round(model_steps * step_size))
    num_slices = np.int64(np.floor(n_steps/slice_size))
    N = np.int64(n_train * num_slices * slice_size)
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    for m in range(slice_size):
        #------------------------------------
        for j in range(1, num_slices):
            idx_start = (j-1)*slice_size + m
            idx_end = j*slice_size + m
            #idx_slices = np.array(list(range(j-1, j+N, num_slices))) # j+N-num_slices
            #idx_slices = idx_slices + m*slide_size*np.ones((2*num_slices, ), dtype=int)
            idx = m * num_slices + (j-1)
            idx_slices = np.array(list(range(idx, idx+n_train)))
            slice_data[idx_slices, :, :] = train_data[:, idx_start:idx_end+1, :]
    # save training slices to file
    np.save(os.path.join(data_dir, f"train_D{step_size}.npy"), slice_data)
#--------------------------------------------------------
# simulate validation trials
#--------------------------------------------------------
val_data = np.zeros((n_val, n_steps + 1, n_inputs))
# time_samples = np.random.uniform(t_final/3, 2*t_final/3, n_val)
print('==============================')
print('generating validation trials ...')
for idx in range(n_val):
    #----------------------------------
    sound = bub.SoundWave(amp_range, freq_range, n_waves)
    root = fsolve(sound.pressure, period_max, xtol=1e-6)
    t0 = root[0]
    for j in range(n_waves):
        sound.waves[j].phase = sound.waves[j].phase + (2*np.pi*sound.waves[j].freq)*t0
    #----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, = (p(t) - P0) / P0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    sol = solve_ivp(bubble.rhs_rp, t_span=[0.0, t_final+dt], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    val_data[idx, 0, :n_outputs] = y_init
    val_data[idx, 1:, :n_outputs] = sol.y.T[0:n_steps]
    val_data[idx, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))
# save validation samples to file
np.save(os.path.join(data_dir, f"val.npy"), val_data)
# slice samples for each time-stepper
for k in range(k_min, k_max+1):
    step_size = np.int64(np.round(2**k))
    slice_size = np.int64(np.round(model_steps * step_size))
    num_slices = np.int64(np.floor(n_steps/slice_size))
    N = np.int64(n_val * num_slices * slice_size)
    slice_data = np.zeros((N, slice_size + 1, n_inputs))
    for m in range(slice_size):
        #------------------------------------
        for j in range(1, num_slices):
            idx_start = (j-1)*slice_size + m
            idx_end = j*slice_size + m
            #idx_slices = np.array(list(range(j-1, j+N, num_slices))) # j+N-num_slices
            #idx_slices = idx_slices + m*slide_size*np.ones((2*num_slices, ), dtype=int)
            idx = m * num_slices + (j-1)
            idx_slices = np.array(list(range(idx, idx+n_val)))
            slice_data[idx_slices, :, :] = val_data[:, idx_start:idx_end+1, :]
    # save validation samples to file
    np.save(os.path.join(data_dir, f"val_D{step_size}.npy"), slice_data)
#--------------------------------------------------------
# simulate test trials
#--------------------------------------------------------
test_data = np.zeros((n_test, n_steps + 1, n_inputs))
#time_samples = np.random.uniform(t_final/3,2*t_final/3, n_test)
print('==============================')
print('generating testing trials ...')
for idx in range(n_test):
    #----------------------------------
    sound = bub.SoundWave(amp_range, freq_range, n_waves)
    root = fsolve(sound.pressure, period_max, xtol=1e-6)
    t0 = root[0]
    for j in range(n_waves):
        sound.waves[j].phase = sound.waves[j].phase + (2*np.pi*sound.waves[j].freq)*t0
    # ----------------------------------
    for j in range(n_steps + 1):
        t = dt * j
        P[j] = sound.pressure(t)  # in the paper, = (p(t) - P0) / P0
        Pdot[j] = sound.pressure_dot(t)
    # ----------------------------------
    sol = solve_ivp(bubble.rhs_rp, t_span=[0.0, t_final+dt], y0=y_init, args=(sound,), t_eval=t_space, method='LSODA', rtol=rel_tol, atol=abs_tol)
    test_data[idx, 0, :n_outputs] = y_init
    test_data[idx, 1:, :n_outputs] = sol.y.T[0:n_steps]
    test_data[idx, :, n_outputs:] = np.column_stack((P.reshape(n_steps+1,1), Pdot.reshape(n_steps+1,1)))
np.save(os.path.join(data_dir, f"test.npy"), test_data)

#=========================================================
# Plot Samples of Data
#=========================================================
print('==============================')
print('generating images ...')
print('==============================')
j_samples = range(0, np.min([n_test,16]))

for j in j_samples:

    R = test_data[j, :, 0]
    Rdot = test_data[j, :, 1]
    P = test_data[j, :, 2]
    fig, axs = plt.subplots(2, 1, figsize=(plot_x_dim, 1.1 * plot_y_dim * 2))
    parameters = f"{amp_min} $ \leq max[ C_p(t) ] \leq $ {amp_max}  \n \n  {period_min}$ \leq T_i \leq ${period_max} | $ 1 \leq i \leq ${n_waves} \n \n $R(0)=${R[0]} | $ \\frac{{dR}}{{dt}}(0)=${Rdot[0]}\n"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    #-------------------------------------------------------------------------------
    axs[0].plot(t_space, R, color='tab:blue', label='$R(t)$', linewidth=line_width)
    #axs[0].legend(fontsize=legend_fontsize, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    axs[0].set_ylim(ymin=0.8*min(R),ymax=1.2*max(R))
    axs[0].set_xlabel('t / $t_c$',fontsize=x_label_fontsize)
    axs[0].set_ylabel('R / $R_0$',fontsize=y_label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axs[0].text(0.0*t_final, max(R), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
    #-------------------------------------------------------------------------------
    axs[1].plot(t_space, P, color='tab:red', label='$P(t)$',  linewidth=line_width)
    axs[1].set_xlabel('t / $t_c$', fontsize=x_label_fontsize)
    axs[1].set_ylabel('$C_p(t)$', fontsize=y_label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    # axs[1].text(0.0*t_final, max(P), parameters, fontsize=box_fontsize, verticalalignment='top', bbox=props)
    #-------------------------------------------------------------------------------
    file_fig_data = os.path.abspath(os.path.join(data_dir, f"test_data_{j}.png" ))
    fig.tight_layout(pad=2.0)
    fig.savefig(file_fig_data, dpi=200)
    plt.close(fig) # or maybe plt.clf()

print("images saved")
print('==============================')