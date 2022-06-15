# # Multiscale HiTS Visuals (updated visuals)

# ## created by Scott Sims 06/14/2022
#=========================================================
# IMPORT PACKAGES
#=========================================================
import os
import sys
import pdb
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
#from datetime import datetime
#--------------------------------------------------
print("=================")
print("CURRENT DIRECTORY")
print(os.getcwd())
module_dir = os.path.abspath( os.path.join(os.getcwd(),'src'))
if module_dir not in sys.path:
    sys.path.append(module_dir)
import ResNet as net
import bubble_methods as bub
#=========================================================
# Input Arguments
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)
for key in D:
    globals()[str(key)] = D[key]
    print(f"{str(key)}: {D[key]}")
    # transforms key-names from dictionary into global variables, then assigns them the dictionary-values
#print('TO CONTINUE, PRESS [c] THEN [ENTER]')
#print('TO QUIT, PRESS [q] THEN [ENTER]')
#pdb.set_trace()
#=========================================================
# Directories and Paths
#=========================================================
n_steps = bub.get_num_steps(dt, model_steps, step_sizes[-1], period_min, n_periods)
print(f"number of time-steps = {n_steps}")
data_folder = f"data_dt={dt}_n-steps={n_steps}_m-steps={model_steps}_delta={step_sizes}_n-waves={n_wave}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_train+val+test={n_train}+{n_val}+{n_test}"
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
model_folder = f"models_dt={dt}_steps={n_steps}_m-steps={model_steps}_delta={step_sizes}_n-waves={n_wave}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_{n_lr}-lr={learn_rate_min}-{learn_rate_max}_n-batches={n_batch}_nresnet={n_input}+{n_layer}x{n_neuron}+{n_output}"
model_dir = os.path.join(os.getcwd(), 'models', model_folder)
if not os.path.exists(data_dir):
    sys.exit(f"Cannot find folder ../data/{data_folder} in current directory")
if not os.path.exists(model_dir):
    sys.exit(f"Cannot find folder ../models/{model_folder} in current directory")
#--------------------------------------------------
# file names for figures
file_fig_mse_ladder = f"plot_mse_ladder_{system}.png"
file_fig_mse_hamiltonian = f"plot_mse_hamiltonian_{system}.png"
file_fig_mse_kinetic = f"plot_mse_kinetic_{system}.png"
file_fig_mse_potential = f"plot_mse_potential_{system}.png"

#========================================================================
# Load Data and Models (then prepare some globals)
#========================================================================
# load validation set and test set
test_data = np.load(os.path.join(data_dir, 'test.npy'))
#--------------------------------------------------
# load models
models = list()
for step_size in step_sizes:
    model_name = f"model_D{step_size}.pt"
    models.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
num_k = len(models)
print(f"{num_models} models loaded for time-stepping:")
print(f"model index:  k = {k_min} .. {k_max}")
print(f"step size:  2^k = {step_sizes[0]} .. {step_sizes[k_max-k_min]}" )
print(f"dt = {dt}")
print(f"step size:  (2^k)dt = {dt*step_sizes[0]} .. {dt*step_sizes[k_max-k_min]} \n")
#--------------------------------------------------
# fix model consistencies trained on gpus (optional)
for model in models:
    model.device = 'cpu'
    model._modules['increment']._modules['activation'] = torch.nn.ReLU()
#--------------------------------------------------
# shared info
n_steps = test_data.shape[1] - 1
t_space = [dt*(step+1) for step in range(n_steps)]    # = 1,2, ... , n (list)
criterion = torch.nn.MSELoss(reduction='none')

#========================================================================
# Create Directories and Files for Figures
#========================================================================
# create directory for figures
figure_dir = model_dir
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
#--------------------------------------------------
# paths for figures
file_fig_mse_ladder = os.path.abspath( os.path.join(figure_dir, file_fig_mse_ladder) )
file_fig_mse_hamiltonian = os.path.abspath( os.path.join(figure_dir, file_fig_mse_hamiltonian) )
file_fig_mse_kinetic = os.path.abspath( os.path.join(figure_dir, file_fig_mse_kinetic) )
file_fig_mse_potential = os.path.abspath( os.path.join(figure_dir, file_fig_mse_potential) )

#========================================================================
# Generate Req(Cp) using root-finding
#========================================================================
delta_cp = np.round(10**(-n_decimal_cp),decimals=n_decimal_cp)
num_cp = np.int64(np.round((amp_max-(-amp_min))/delta_cp + 1))
Req_array, Cp_array = bub.get_Req_vs_Cp(-amp_max,amp_max,num_cp,Ca,S)
Req_array = np.array(Req_array)
Cp_array = np.array(Cp_array)

#==========================================================
# MSE of LADDER descent
#==========================================================
data_Pdot = list()
data_Rdot = list()
data_Hdot = list()
data_KE = list()
data_PE = list()
data_mse = list()
data_rung = list()
mse_ladder = np.zeros((n_delta,n_rungs))
for k in range(len(step_sizes)):
    #--------------------------------------------------
    Pdot_list = list()
    Rdot_list = list()
    Hdot_list = list()
    KE_list = list()
    PE_list = list()
    mse_list = list()
    rung_list = list()
    #--------------------------------------------------
    models = models[k]
    step_size = step_sizes[k]
    for idx_test in range(n_test):
        #--------------------------------------------------
        R = test_data[idx_test,:,0]
        Rdot = test_data[idx_test,:,1]
        P = test_data[idx_test,:,2]
        Pdot = test_data[idx_test,:,3]
        #--------------------------------------------------
        # find all critical points and store their indices
        idx_critical = list()
        for j in range(n_steps+1):
            if Rdot[j]*Rdot[j+1] < 0: # if it is a critical point
                if Rdot[j] < Rdot[j+1]: # if a minimum
                    if R[j] < R[j+1]:
                        idx_critical.append([j,0])
                    else:
                        idx_critical.append([j+1,0])
                elif Rdot[j] > Rdot[j+1]: # if a maximum
                    if R[j] > R[j+1]:
                        idx_critical.append([j,1])
                    else:
                        idx_critical.append([j+1,1])
        #--------------------------------------------------
        # find all intervals with cusps, determined by "steep_Rdot"
        idx_descent = list()
        for m in range(len(idx_critical)):
            if idx_critical[m][1] == 1: # if it is a maximum
                idx_top = idx_critical[m][0]
                idx_bot = idx_critical[m+1][0]
                j = idx_top
                end_loop = False
                while(end_loop == False):
                    j = j+1
                    if np.abs(Rdot[j]) > steep_Rdot:
                        end_loop = True
                        idx_descent.append([idx_top, idx_bot])
                    elif j == idx_bot:
                        end_loop = True
        #--------------------------------------------------
        # calculate MSE of ladder down descent
        for idx_top, idx_bot in idx_descent:
            R_descent = np.linspace(R[idx_top], R[idx_bot], num=n_rungs+2).tolist()
            R_ladder = R_descent[1:-1]
            j = idx_top
            if (idx_bot + step_size) <= n_steps:
                num_ladders = num_ladders + 1
                for m in range(n_rungs):
                    r = R_ladder[m]
                    while( R[j] > r ):
                        j = j+1
                    #--------------------------------------------------
                    if( np.abs(r-R[j]) < np.abs(r-R[j-1]) ):
                        idx = j
                    else:
                        idx = j-1
                    #--------------------------------------------------
                    y_pred = model.uni_scale_forecast( torch.tensor(test_data[idx_test:idx_test+1, idx, :n_output]).float(), n_steps=1, y_known=torch.tensor(test_data[idx_test:idx_test+1, idx:idx+step_size+1, n_output:]).float() )
                    mse = (y_preds[0, -1, 1].detach().numpy() - R[idx+step_size])**2
                    mse_ladder[k,m] += mse
                    #--------------------------------------------------
                    idx_cp = np.around(num_cp*(P[idx] - Cp_array[0])/(Cp_array[-1] - Cp_array[0]), decimals=0)
                    Pdot_list.append(Pdot[idx])
                    Rdot_list.append(Rdot[idx])
                    Hdot_list.append(bub.Hdot(R[idx], Req_array[idx_cp], Pdot[idx])) # assumes zero viscoity for now (i.e. Reynolds = infty)
                    KE_list.append(bub.KE(R[idx], Rdot[idx]))
                    PE_list.append(bub.PE(R[idx], Req_array[idx_Cp], P[idx], Ca, S, poly_index))
                    mse_list.append(mse)
                    rung_list.append(m)
    #--------------------------------------------------
    data_Pdot.append(Pdot_list)
    data_Rdot.append(Rdot_list)
    data_Hdot.append(Hdot_list)
    data_KE.append(KE_list)
    data_PE.append(PE_list)
    data_mse.append(mse_list)
    data_rung.append(rung_list)
#========================================================================
for m in range(n_rungs):
    mse_ladder[m] = mse_ladder[m]/num_ladders