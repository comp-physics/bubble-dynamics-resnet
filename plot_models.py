# # Multiscale HiTS Visuals (updated visuals)

# ## adapted by Scott Sims 05/19/2022
# ## created by Yuying Liu, 05/07/2020
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
from bubble_methods import get_num_steps
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
n_steps = get_num_steps(dt, model_steps, k_max, period_min, n_periods)
print(f"number of time-steps = {n_steps}")
data_folder = f"data_dt={dt}_n-steps={n_steps}_m-steps={model_steps}_k={k_min}-{k_max}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_{n_wave}waves_train+val+test={n_train}+{n_val}+{n_test}"
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
model_folder = f"models_dt={dt}_steps={n_steps}_m-steps={model_steps}_k={k_min}-{k_max}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_lr{n_lr}={learn_rate_min}-{learn_rate_max}_resnet={n_input}+{n_layer}x{n_neuron}+{n_output}"
model_dir = os.path.join(os.getcwd(), 'models', model_folder)
if not os.path.exists(data_dir):
    sys.exit(f"Cannot find folder ../data/{data_folder} in current directory")
if not os.path.exists(model_dir):
    sys.exit(f"Cannot find folder ../models/{model_folder} in current directory")
#--------------------------------------------------
# file names for figures
file_fig_uniscale = f"plot_uniscale_{system}.png"
file_fig_mse_models = f"plot_MSE_models_{system}.png"
file_fig_mse_multiscale = f"plot_MSE_multiscale_{system}.png"
file_fig_multiscale = f"plot_multiscale_{system}.png"

#========================================================================
# Load Data and Models (then prepare some globals)
#========================================================================
# load validation set and test set
test_data = np.load(os.path.join(data_dir, 'test.npy'))
#--------------------------------------------------
# list of k-values: k = 0 ... k_max
ks = list(range(k_min, k_max+1))
step_sizes = [ np.int64(np.round(2**k)) for k in ks ]
num_models = len(ks)
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
file_fig_uniscale = os.path.abspath( os.path.join(figure_dir, file_fig_uniscale) )
file_fig_mse_models = os.path.abspath( os.path.join(figure_dir, file_fig_mse_models) )
file_fig_mse_multiscale = os.path.abspath( os.path.join(figure_dir, file_fig_mse_multiscale) )
file_fig_multiscale = os.path.abspath( os.path.join(figure_dir, file_fig_multiscale) )

#==========================================================
# Plot Predictions of Individual ResNet time-steppers
#==========================================================
idx = 0
# iterate_k = iter(ks)
colors= plt.cm.rainbow(np.linspace(0, 1, len(ks))) # iter(plt.cm.rainbow(np.linspace(0, 1, len(ks))))
fig, axs = plt.subplots(num_models, 1, figsize=(plot_x_dim, plot_y_dim*num_models*1.3))
print(f"individual time-stepper predictions")
for j in range(len(models)):
    print(f"j = {j}")
    model = models[j]
    rgb = colors[j] # next(colors)
    # k = ks[j]     # next(iterate_k)
    #print(torch.tensor(test_data[idx, 0, :n_output]).float().shape)
    #print(torch.tensor(test_data[idx, :, n_output:]).float().shape)
    y_preds = model.uni_scale_forecast( torch.tensor(test_data[idx:idx+1, 0, :n_output]).float(), n_steps=n_steps, y_known=torch.tensor(test_data[idx:idx+1, :, n_output:]).float() )
    R = y_preds[0, 0:n_steps, 1].detach().numpy()
    axs[j].plot(t_space, test_data[idx, 0:n_steps, 1], linestyle='-', color='gray', linewidth=10, label='R(t)')
    axs[j].plot(t_space, R, linestyle='--', color=rgb, linewidth=6, label='$\Delta t = ${}dt'.format(step_sizes[j]) )
    axs[j].legend(fontsize=legend_fontsize, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.17))
    axs[j].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axs[j].grid(axis='y')
plt.show()
plt.savefig(file_fig_uniscale)
print("figure saved")

#==========================================================
# Plot Log(MSE) of Predictions (individual models)
#==========================================================
# uniscale time-stepping with NN
preds_mse = list()
times = list()
for model in models:
    start = time.time()
    y_preds = model.uni_scale_forecast( torch.tensor(test_data[:, 0, :n_output]).float(), n_steps=n_steps, y_known=torch.tensor(test_data[:, :, n_output:]).float() )
    end = time.time()
    times.append(end - start)
    preds_mse.append(criterion(torch.tensor(test_data[:, 1:, 0]).float(), y_preds[:,:,0]).mean(-1)) # CHECK THIS! CHECK THIS!
#----------------------------------------------------------
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
colors= plt.cm.rainbow(np.linspace(0, 1, len(ks))) # iter( plt.cm.rainbow(np.linspace(0, 1, len(ks))) )
dot_sizes = np.linspace(1,20,len(preds_mse)) # iter(( np.linspace(1,20,len(preds_mse)) ))
t_array = np.array(t_space)
m_steps = n_steps-1
max_log = 0
min_log = 0
print("MSE of individual time-steppers")
for j in range(len(ks)):
    print(f"j = {j}")
    k = ks[j]
    err = preds_mse[j]
    err = err.mean(0).numpy()
    rgb = colors[j]
    n_forward = np.int64( np.round( m_steps / step_sizes[j] ) )
    key = np.int64( np.round( np.linspace(0,m_steps,n_forward+1) ) )
    t_key = t_array[key]
    log_err_key = np.log10(err[key])
    plt.plot(t_key, log_err_key, 'o', fillstyle='full', linestyle='-', linewidth=3, markersize=dot_sizes[j], color=rgb, label='$\Delta\ t$={}dt'.format(step_sizes[j]))
    #max_log = max_log + min(0, np.max(log_err_k[1:])) # accumulate maximum log(MSE) < 0 in order to calculate a average-ceiling < 0

min_log = np.min(err) # err = preds_mse[k_max] from last iteration above
d_log = np.abs(max_log-min_log)
mid_log = np.mean( [min_log, max_log] )
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.24))
plt.title('time-steps without interpolation', y=1.0, pad=-40, fontsize=title_fontsize)
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('log(MSE)',fontsize=y_label_fontsize)
plt.grid(axis = 'y')
plt.ylim(ymin=mid_log-d_log, ymax=mid_log+d_log)
plt.show()
plt.savefig(file_fig_mse_models)
print("figure saved")

#==========================================================
# Choose Range of Models that Minimize MSE (when combined)
#==========================================================
# cross validation (model selections) 
start_idx = 0
end_idx = len(ks)
best_mse = 1e+5
val_data = np.load(os.path.join(data_dir, 'val_D{}.npy'.format(k_max)))
# choose the largest time step
print("choose larget time-stepper")
for j in range(start_idx, end_idx+1):
    print(f"j = {j}")
    step_size = step_sizes[j]
    y_preds = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, 0, :n_output]).float().to('cpu'), n_steps=n_steps, models=models[start_idx:j+1], y_known=torch.tensor(val_data[:, :, n_output:]).float().to('cpu'))
    mse = criterion(torch.tensor(val_data[:, 1:, :n_output]).float(), y_preds).mean().item()
    if mse <= best_mse:
        end_idx = j
        best_mse = mse
#----------------------------------------------------------
# choose the smallest time step
print("choose smallest time-stepper")
for j in range(start_idx, end_idx+1):
    print(f"j = {j}")
    step_size = step_sizes[j]
    y_preds = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, 0, :n_output]).float().to('cpu'), n_steps=n_steps, models=models[j:end_idx+1], y_known=torch.tensor(val_data[:, :, n_output:]).float().to('cpu'))
    mse = criterion(torch.tensor(val_data[:, 1:, :n_output]).float(), y_preds).mean().item()
    if mse <= best_mse:
        start_idx = j
        best_mse = mse
#----------------------------------------------------------
models = models[start_idx:(end_idx+1)]
num_k = len(models)
print('{} models chosen for Multiscale HiTS:'.format(num_k) )
print('   k    = {} .. {}'.format(start_idx, end_idx) )
print('  2^k   = {} .. {}'.format(2**(start_idx+k_min), 2**(end_idx+k_min) ) )
print('(2^k)dt = {} .. {}\n'.format(dt*2**(start_idx+k_min), dt*2**(end_idx+k_min) ) )
del val_data

#==========================================================
# Plot Log(MSE) for Multi-scale vs Single
#==========================================================
# multiscale time-stepping with NN
start = time.time()
y_preds, model_key = net.vectorized_multi_scale_forecast(torch.tensor(test_data[:, 0, :n_output]).float().to('cpu'), n_steps=n_steps, models=models, y_known=torch.tensor(test_data[:, :, n_output:]).float().to('cpu'), key=True)
end = time.time()
multiscale_time = end - start
multiscale_preds_mse = criterion(torch.tensor(test_data[:, 1:, :n_output]).float(), y_preds).mean(-1)
# added additional argument to function 'vectorized_multi_scale_forecast( ... , key=True)' in order to data of each individual ResNet
model_key = model_key.detach().numpy()
#model_key_plus = np.delete(model_key, np.argwhere(model_key==0) )
#----------------------------------------------------------
# visualize forecasting error at each time step    
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
colors= plt.cm.rainbow(np.linspace(0, 1, len(ks))) # iter(plt.cm.rainbow(np.linspace(0, 1, len(ks))))
print("MSE of multiscale time-steppers")
for j in range(len(preds_mse)+1):
    print(f"j = {j}")
    err = preds_mse[j].mean(0).detach().numpy()
    rgb = colors[j]
    plt.plot(t_space, np.log10(err), linestyle='-', color=rgb, linewidth=4, label='$\Delta\ t$={}dt'.format(step_sizes[j]))
multiscale_err = multiscale_preds_mse.mean(0).detach().numpy()
plt.plot(t_space, np.log10(multiscale_err), linestyle='-', color='k', linewidth=4, label='multiscale')
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.2))
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('log(MSE)',fontsize=y_label_fontsize)
plt.grid(axis = 'y')
plt.ylim(ymin=min_log-d_log)
plt.savefig(file_fig_mse_multiscale)


#==========================================================
# Plot Multiscale Predictions with a color-key for each chosen model
#==========================================================
idx = 0
dot_min = 6
dot_max = 10
#-------------------------------------------------------------
t = np.linspace(0, (n_steps-1)*dt, n_steps)  # = 0,1,2, ... , (n-1) (numpy array)
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
#gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.5)
#ax0 = fig.add_subplot(gs[0, :])
#------------------------------------------------------------
P = y_preds[idx, 0:n_steps, 0].detach().numpy()
R = y_preds[idx, 0:n_steps, 1].detach().numpy()
#Rdot = y_preds[idx, 0:n_steps, 2].detach().numpy()
#-------------------------------------------------------------
#plt.plot(t_space, test_data[idx, 0:n_steps, 0], color='gray', linestyle='-', linewidth=10, label='P(t)' )
plt.plot(t_space, test_data[idx, 0:n_steps, 1], color='lightgray', linestyle='-', linewidth=10, label='R(t)')
#-------------------------------------------------------------
#plt.plot(t, P, color='black', linestyle='--', linewidth=5)
#plt.plot(t, R, color='black', linestyle='--', linewidth=5, label='learned')
#plt.plot(t, Rdot, color='black', linestyle='--', linewidth=5)
#-------------------------------------------------------------
high = end_idx
low = np.max( [high-2,start_idx] ) # accounts for the undesired possibility that (high-2)<0
ks = list(range(low,high+1))
iterate_k = iter( reversed(ks) )
colors = iter( reversed(plt.cm.rainbow(np.linspace(0, 1, len(ks))) ) )
dot_sizes = iter( reversed( np.linspace(dot_max,dot_min,len(ks)) ) )
for idx in reversed(range(1,len(ks)+1)):
    rgb = next(colors)
    k = next(iterate_k)
    dot_size = next(dot_sizes)
    t_model = np.delete(t_space, np.where(model_key != idx) )
    R_model = np.delete(R, np.where(model_key != idx) )
    plt.plot( t_model, R_model, 'o', fillstyle='full', markersize=dot_size, markeredgewidth=0.0 , color=rgb, label='$\Delta t$={}dt'.format(step_sizes[k]) )
#-------------------------------------------------------------
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('f(t)', rotation=0, fontsize=y_label_fontsize)
num_cols = np.int64( np.ceil( (2+len(models))/2 ) )
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=num_cols, bbox_to_anchor=(0.5, 1.22))
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.grid( axis='y')
plt.savefig(file_fig_multiscale)

#=========================================================
# Print Computation Time (sec)
#=========================================================
print('ensembled multiscale compute time = {:.4f} s'.format(multiscale_time))
for i in range(len(times)):
    print('{:.2f} | single scale compute time = {:.4f} s'.format(step_sizes[i]*dt, times[i]))

