# # Multiscale HiTS Visuals (updated visuals)

# ## updated by Scott Sims 10/19/2021
# ## created by Yuying Liu, 05/07/2020


import os
import sys
import time
import torch
import numpy as np
#import scipy.interpolate
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import yaml
from datetime import datetime

module_dir = os.path.abspath( os.path.join(os.getcwd(),'src') )
if module_dir not in sys.path:
    sys.path.append(module_dir)
    
import ResNet as net

#=========================================================
# Command Line Arguments
#=========================================================
# print("| system = {0:s} | dt={1:} | k_max={2:} | num_inputs={3:} | num_layers={4:} | layer_size={5:} |".format(system,dt,k_max,num_inputs,num_layers,layer_size) )
file_parameters = open("parameters.yaml", 'r')
dictionary = yaml.load(file_parameters, loader=yaml.FullLoader)
file_parameters.close()
#---------------------------------------
system = dictionary[system]
dt = dictionary[dt]
k_max = dictionary[k_max]
steps_min = dictionary[steps_min]
steps_max = dictionary[steps_max]
P_min = dictionary[P_min]
P_max = dictionary[P_max]
R_min = dictionary[R_min]
R_max = dictionary[R_max]
R_test = dictionary[R_test]
Rdot_min = dictionary[Rdot_min]
Rdot_max = dictionary[Rdot_max]
Rdot_test = dictionary[Rdot_test]
n_train = dictionary[n_train]
n_val = dictionary[n_val]
n_test = dictionary[n_test]
num_layers = dictionary[num_layers]
layer_size = dictionary[layer_size]
num_inputs = dictionary[num_inputs]

#=========================================================
# Directories and Paths
#=========================================================
n_steps = np.int64(steps_min * 2**k_max)
data_folder = 'data_dt={}_steps={}_P={}-{}_R={}-{}_(train|val|test)=({}|{}|{}).pt'.format(dt, n_steps, P_min, P_max, R_min, R_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
model_folder = 'models_dt={}_P={}-{}_R={}-{}_inputs={}_resnet={}x{}.pt'.format(dt, P_min, P_max, R_min, R_max, num_inputs, num_layers, layer_size)
model_dir = os.path.join(os.getcwd(), 'models', model_folder)
if not os.path.exists(data_dir):
    sys.exit("Cannot find folder ../data/{} in current directory".format(data_folder))
if not os.path.exists(model_dir):
    sys.exit("Cannot find folder ../models/{} in current directory".format(data_folder))

# file names for figures
file_fig_uniscale = 'plot_uniscale_{}.png'.format(system)
file_fig_mse_models = 'plot_MSE_models_{}.png'.format(system)
file_fig_mse_multiscale = 'plot_MSE_multiscale_{}.png'.format(system)
file_fig_multiscale = 'plot_multiscale_{}.png'.format(system)

# plot properties
plot_x_dim = 30
plot_y_dim = 10
title_fontsize = 40
legend_fontsize = 30
x_label_fontsize = 30
y_label_fontsize = 30
x_axis_fontsize = 30
y_axis_fontsize = 30

#========================================================================
# Load Data and Models (then prepare some globals)
#========================================================================
# load validation set and test set
val_data = np.load(os.path.join(data_dir, 'val.npy'))
test_data = np.load(os.path.join(data_dir, 'test.npy'))

# list of k-values: k = 0 ... k_max
ks = list(range(k_max+1))
step_sizes = [2**k for k in ks]
num_models = k_max+1

# load models
models = list()
for step_size in step_sizes:
    model_name = 'model_D{}.pt'.format(step_size)
    models.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
num_k = len(models)
print('{} models loaded for time-stepping:'.format(num_models) ) 
print('   k    = {} .. {}'.format(0, k_max) )
print('  2^k   = {} .. {}'.format(1, step_sizes[k_max] ) )
print('t-steps = {} .. {} \n'.format(dt, dt*step_sizes[k_max]) )

# fix model consistencies trained on gpus (optional)
for model in models:
    model.device = 'cpu'
    model._modules['increment']._modules['activation'] = torch.nn.ReLU()

# shared info
n_steps = test_data.shape[1] - 1
t = [dt*(step+1) for step in range(n_steps)]    # = 1,2, ... , n (list)
criterion = torch.nn.MSELoss(reduction='none')

#========================================================================
# Create Directories and Files for Figures
#========================================================================
# date and time
now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

# create directory for figures
#figs_folder = 'figures_dt={}_steps={}_P={}-{}_R0={}_inputs={}_resnet={}x{}.pt'.format(dt, n_steps, P_min, P_max, R_test, num_inputs, num_layers, layer_size)
figure_dir = model_dir
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# paths for figures
file_fig_uniscale = os.path.abspath( os.path.join(figure_dir, file_fig_uniscale) )
file_fig_mse_models = os.path.abspath( os.path.join(figure_dir, file_fig_mse_models) )
file_fig_mse_multiscale = os.path.abspath( os.path.join(figure_dir, file_fig_mse_multiscale) )
file_fig_multiscale = os.path.abspath( os.path.join(figure_dir, file_fig_multiscale) )

#==========================================================
# Plot Single-Scale Time-Steppers
#==========================================================
# uniscale time-stepping with NN
preds_mse = list()
times = list()
for model in models:
    start = time.time()
    y_preds = model.uni_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps=n_steps)
    end = time.time()
    times.append(end - start)
    preds_mse.append(criterion(torch.tensor(test_data[:, 1:, :]).float(), y_preds).mean(-1))

#---------------------------------------------------------
# plot predictions of individual models
idx = 0
iterate_k = iter(ks)
colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(ks))))
fig, axs = plt.subplots(num_models, 1, figsize=(plot_x_dim, plot_y_dim*num_models*1.3))

for model in models:
    rgb = next(colors)
    k = next(iterate_k)
    y_preds = model.uni_scale_forecast(torch.tensor(test_data[idx:idx+1, 0, :]).float(), n_steps=n_steps)
    #P = y_preds[0, 0:n_steps, 0].detach().numpy()
    R = y_preds[0, 0:n_steps, 1].detach().numpy()
    #Rdot = y_preds[0, 0:n_steps, 2].detach().numpy()
    #axs[k].plot(t, test_data[idx, 0:n_steps, 0], linestyle='-', color='lightgray', linewidth=10, label='P(t)' )
    axs[k].plot(t, test_data[idx, 0:n_steps, 1], linestyle='-', color='gray', linewidth=10, label='R(t)')
    #axs[k].plot(t, test_data[idx, 0:n_steps, 2], linestyle='-', color='darkgray', linewidth=10, label='$\dot{{R}}(t)$')
    #axs[k].plot(t, P, linestyle='--', color='black', linewidth=6, label='learned')
    axs[k].plot(t, R, linestyle='--', color=rgb, linewidth=6, label='$\Delta t = ${}dt'.format(step_sizes[k]) )
    #axs[k].plot(t, Rdot, linestyle='--', color='black', linewidth=6)
    axs[k].legend(fontsize=legend_fontsize, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.17))
    axs[k].tick_params(axis='both', which='major', labelsize=x_axis_fontsize)
    axs[k].grid( axis='y' )
    #R_max = np.max( test_data[idx, 0:n_steps, 1] )
    #axs[k].set_ylim( [0, R_max*1.05] )
plt.show()
plt.savefig(file_fig_uniscale)


#==========================================================
# Plot Log(MSE) of Prediction Errors (individual models)
#==========================================================
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
colors=iter( (plt.cm.rainbow(np.linspace(0, 1, len(ks)))))
dot_sizes = iter( ( np.linspace(1,20,len(preds_mse)) ) )
t_np = np.array(t)
m_steps = n_steps-1
max_log = 0
min_log = 0

for k in range(k_max+1):
    err = preds_mse[k]
    mean = err.mean(0).detach().numpy()
    rgb = next(colors)
    n_forward = np.int64( np.round( m_steps / 2**k ) )
    key = np.int64( np.round( np.linspace(0,m_steps,n_forward+1) ) )
    t_k = t_np[key]
    mean_k = mean[key]
    log_mean_k = np.log10(mean_k)
    n = len(mean_k)
    plt.plot(t_k, log_mean_k, 'o', fillstyle='full', linestyle='-', linewidth=3, markersize=next(dot_sizes), color=rgb, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
    max_log = max_log + min( [0, np.max(log_mean_k[1:])] )


max_log = max_log / k_max
min_log = np.min( preds_mse[k_max] )
d_log = np.abs(max_log-min_log)
mid_log = np.mean( [min_log, max_log] )
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.24))
plt.title('time-steps without interpolation', y=1.0, pad=-40, fontsize=title_fontsize)
plt.xticks(fontsize=x_axis_fontsize)
plt.yticks(fontsize=y_axis_fontsize)
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('log(MSE)',fontsize=y_label_fontsize)
plt.grid(axis = 'y')
plt.ylim(ymin=mid_log-d_log, ymax=mid_log+d_log)
plt.show()
plt.savefig(file_fig_mse_models)


#==========================================================
# Choose Range of Models that Minimize MSE (when combined)
#==========================================================
# cross validation (model selections) 
start_idx = 0
end_idx = k_max
best_mse = 1e+5
# choose the largest time step
for i in range(len(models)):
    y_preds = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, 0, :]).float(), n_steps=n_steps, models=models[:len(models)-i])
    mse = criterion(torch.tensor(val_data[:, 1:, :]).float(), y_preds).mean().item()
    if mse <= best_mse:
        end_idx = len(models)-i
        best_mse = mse

        # choose the smallest time step
for i in range(end_idx):
    y_preds = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, 0, :]).float(), n_steps=n_steps, models=models[i:end_idx])
    mse = criterion(torch.tensor(val_data[:, 1:, :]).float(), y_preds).mean().item()
    if mse <= best_mse:
        start_idx = i
        best_mse = mse
        

models = models[start_idx:(end_idx+1)]
num_k = len(models)
print('{} models chosen for Multiscale HiTS:'.format(num_k) ) 
print('   k    = {} .. {}'.format(start_idx, end_idx) )
print('  2^k   = {} .. {}'.format(2**start_idx, 2**end_idx ) )
print('t-steps = {} .. {}\n'.format(dt*2**start_idx, dt*2**end_idx ) )


#==========================================================
# Plot Log(MSE) for Multi-scale vs Single
#==========================================================
# multiscale time-stepping with NN
start = time.time()
y_preds, model_key = net.vectorized_multi_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps=n_steps, models=models, key=True)
end = time.time()
multiscale_time = end - start
multiscale_preds_mse = criterion(torch.tensor(test_data[:, 1:, :]).float(), y_preds).mean(-1)
# added additional argument to function 'vectorized_multi_scale_forecast( ... , key=True)' in order to data of each individual ResNet
model_key = np.array( model_key )
#model_key_plus = np.delete(model_key, np.argwhere(model_key==0) )


# visualize forecasting error at each time step    
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(ks))))
multiscale_err = multiscale_preds_mse.mean(0).detach().numpy()
for k in range(len(preds_mse)):
    err = preds_mse[k]
    mean = err.mean(0).detach().numpy()
    rgb = next(colors)
    plt.plot(t, np.log10(mean), linestyle='-', color=rgb, linewidth=4, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
plt.plot(t, np.log10(multiscale_err), linestyle='-', color='k', linewidth=4, label='multiscale')
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.2))
plt.xticks(fontsize=x_axis_fontsize)
plt.yticks(fontsize=y_axis_fontsize)
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
#plt.plot(t, test_data[idx, 0:n_steps, 0], color='gray', linestyle='-', linewidth=10, label='P(t)' )
plt.plot(t, test_data[idx, 0:n_steps, 1], color='lightgray', linestyle='-', linewidth=10, label='R(t)')
#plt.plot(t, test_data[idx, 0:n_steps, 2], color='darkgray', linestyle='-', linewidth=10, label='$\dot{{R}}(t)$')
#-------------------------------------------------------------
#plt.plot(t, P, color='black', linestyle='--', linewidth=5)
#plt.plot(t, R, color='black', linestyle='--', linewidth=5, label='learned')
#plt.plot(t, Rdot, color='black', linestyle='--', linewidth=5)
#-------------------------------------------------------------
high = end_idx
low = np.max( [high-2,start_idx] )
ks = list(range(low,high+1))
iterate_k = iter( reversed(ks) )
colors = iter( reversed(plt.cm.rainbow(np.linspace(0, 1, len(ks))) ) )
dot_sizes = iter( reversed( np.linspace(dot_max,dot_min,len(ks)) ) )

for idy in reversed(range(1,len(ks)+1)):
    rgb = next(colors)
    k = next(iterate_k)
    dot_size = next(dot_sizes)
    t_model = np.delete(t, np.where(model_key != idy) )
    R_model = np.delete(R, np.where(model_key != idy) )
    plt.plot( t_model, R_model, 'o', fillstyle='full', markersize=dot_size, markeredgewidth=0.0 , color=rgb, label='$\Delta t$={}dt'.format(step_sizes[k]) )
#-------------------------------------------------------------
#ax0.set_xlabel('time',fontsize=x_label_fontsize)
#ax0.set_ylabel('R(t)',fontsize=y_label_fontsize)
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('f(t)', rotation=0, fontsize=y_label_fontsize)
num_cols = np.int64( np.ceil( (2+len(models))/2 ) )
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=num_cols, bbox_to_anchor=(0.5, 1.22))
#ax0.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=x_axis_fontsize)
plt.yticks(fontsize=y_axis_fontsize)
plt.grid( axis='y')
plt.savefig(file_fig_multiscale)

#==========================================================
# Print Computation Time (sec)
#=========================================================
print('ensembled multiscale compute time = {:.4f} s'.format(multiscale_time))
for i in range(len(times)):
    print('{:.2f} | single scale  compute time = {:.4f} s'.format(step_sizes[i]*dt, times[i]))

