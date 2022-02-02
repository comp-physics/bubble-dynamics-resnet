# # Multiscale HiTS Visuals (updated visuals)

# ## updated by Scott Sims 11/23/2021
# ## created by Yuying Liu, 05/07/2020


import os
import sys
import pdb
import time
import torch
import numpy as np
#import scipy.interpolate
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import yaml
#from datetime import datetime

module_dir = os.path.abspath( os.path.join(os.getcwd(),'src'))
if module_dir not in sys.path:
    sys.path.append(module_dir)
    
import ResNet as net

#=========================================================
# Input Arguments
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)

for key in D:
    globals()[str(key)] = D[key]
    print('{} : {}'.format(str(key), D[key]))
    # transforms key-names from dictionary into global variables, then assigns them the dictionary-values

#print('PRESS [c] TO CONTINUE. PRESS [q] TO QUIT.')
#pdb.set_trace()
#=========================================================
# Directories and Paths
#=========================================================
n_steps = np.int64(model_steps * 2**k_max)
data_folder = 'data_dt={}_steps={}_freq={}-{}_amp={}-{}_train+val+test={}+{}+{}'.format(dt, n_steps, freq_min, freq_max, amp_min, amp_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
model_folder = 'models_dt={}_steps={}_freq={}-{}_amp={}-{}_resnet={}+{}x{}+{}'.format(dt, n_steps, freq_min, freq_max, amp_min, amp_max, n_inputs, n_layers, width, n_outputs)
model_dir = os.path.join(os.getcwd(), 'models', model_folder)
if not os.path.exists(data_dir):
    sys.exit("Cannot find folder ../data/{} in current directory".format(data_folder))
if not os.path.exists(model_dir):
    sys.exit("Cannot find folder ../models/{} in current directory".format(model_folder))

# file names for figures
file_fig_uniscale = 'plot_uniscale_{}.png'.format(system)
file_fig_mse_models = 'plot_MSE_models_{}.png'.format(system)
file_fig_mse_multiscale = 'plot_MSE_multiscale_{}.png'.format(system)
file_fig_multiscale = 'plot_multiscale_{}.png'.format(system)



#========================================================================
# Load Data and Models (then prepare some globals)
#========================================================================
# load validation set and test set
test_data = np.load(os.path.join(data_dir, 'test.npy'))
print('test_data.shape() = {}'.format(test_data.shape))
# list of k-values: k = 0 ... k_max
ks = list(range(0,k_max+1))
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
#now = datetime.now()
#date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

# create directory for figures
#figs_folder = 'figures_dt={}_steps={}_P={}-{}_R0={}_inputs={}_resnet={}x{}.pt'.format(dt, n_steps, freq_min, freq_max, R_test, num_inputs, num_layers, width)
figure_dir = model_dir
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# paths for figures
file_fig_uniscale = os.path.abspath( os.path.join(figure_dir, file_fig_uniscale) )
file_fig_mse_models = os.path.abspath( os.path.join(figure_dir, file_fig_mse_models) )
file_fig_mse_multiscale = os.path.abspath( os.path.join(figure_dir, file_fig_mse_multiscale) )
file_fig_multiscale = os.path.abspath( os.path.join(figure_dir, file_fig_multiscale) )

#==========================================================
# Plot Predictions of Individual ResNet time-steppers
#==========================================================
#print('n_inputs: {}'.format(models[0].n_inputs))
#print('n_outpus: {}'.format(models[0].n_outputs))
num_plots = np.min([num_models,5])
temp_indices = list(range(num_modles-num_plots,num_models))

idx = 0
k=0
model_idx=iter(temp_indices)
colors=iter(plt.cm.rainbow(np.linspace(0, 1, num_plots)))
fig, axs = plt.subplots(num_plots, 1, figsize=(plot_x_dim, plot_y_dim*num_plots*1.2))

for model in models[temp_indices]:
    rgb = next(colors)
    j = next(model_idx)
    y_preds = model.uni_scale_forecast( torch.tensor(test_data[idx:idx+1, 0, :n_outputs]).float(), n_steps=n_steps, y_known=torch.tensor(test_data[idx:idx+1, :, n_outputs:]).float() )
    R = y_preds[0, :n_steps, 1].detach().numpy()
    axs[k].plot(t, test_data[idx, 0:n_steps, 1], linestyle='-', color='gray', linewidth=10, label='R(t)')
    axs[k].plot(t, R, linestyle='--', color=rgb, linewidth=6, label='$\Delta t = ${}dt'.format(step_sizes[j]) )
    axs[k].legend(fontsize=legend_fontsize, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.17))
    axs[k].tick_params(axis='both', which='major', labelsize=axis_fontsize)
    axs[k].grid(axis='y')
plt.show()
plt.savefig(file_fig_uniscale)

#==========================================================
# Plot Log(MSE) of Predictions (individual models)
#==========================================================
# uniscale time-stepping with NN
preds_mse = list()
times = list()
for model in models:
    start = time.time()
    y_preds = model.uni_scale_forecast( torch.tensor(test_data[:, 0, :n_outputs]).float().to('cpu'), n_steps=n_steps, y_known=torch.tensor(test_data[:, :, n_outputs:]).float().to('cpu') )
    end = time.time()
    times.append(end - start)
    preds_mse.append(criterion(torch.tensor(test_data[:, 1:, :n_outputs]).float(), y_preds).mean(-1))

#----------------------------------------------------------
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
colors=iter( (plt.cm.rainbow(np.linspace(0, 1, len(ks)))))
dot_sizes = iter( ( np.linspace(1,20,len(preds_mse)) ) )
t_array = np.array(t)
m_steps = n_steps-1
max_log = 0
min_log = 0

for k in range(0,k_max+1):
    err = preds_mse[k]
    err = err.mean(0).numpy()
    rgb = next(colors)
    n_forward = np.int64( np.round( m_steps / 2**k ) )
    key = np.int64( np.round( np.linspace(0,m_steps,n_forward+1) ) )
    t_k = t_array[key]
    log_err_k = np.log10(err[key])
    plt.plot(t_k, log_err_k, 'o', fillstyle='full', linestyle='-', linewidth=3, markersize=next(dot_sizes), color=rgb, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
    #max_log = max_log + min(0, np.max(log_err_k[1:])) # accumulate maximum log(MSE) < 0 in order to calculate a average-ceiling < 0


#max_log = max_log / k_max # average-ceiling < 0 of log(MSE)
min_log = np.min(err) # err = preds_mse[k_max] from last iteration above
print('min_log = {}'.format(min_log))
d_log = np.abs(max_log-min_log)
mid_log = np.mean( [min_log, max_log] )
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.24))
plt.title('time-steps without interpolation', y=1.0, pad=-40, fontsize=title_fontsize)
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('log(MSE)',fontsize=y_label_fontsize)
plt.grid(axis = 'y')
#plt.ylim(ymin=mid_log-d_log, ymax=mid_log+d_log)
plt.show()
plt.savefig(file_fig_mse_models)

#==========================================================
# Choose Range of Models that Minimize MSE (when combined)
#==========================================================
# cross validation (model selections) 
start_idx = 0
end_idx = k_max   # or len(models)-1
best_mse = 1e+5
val_data = np.load(os.path.join(data_dir, 'val_D{}.npy'.format(np.int(2**k_max))))
# choose the largest time step
for k in range(0, k_max+1):
    step_size = np.int64(2**k)
    y_preds = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, 0, :n_outputs]).float().to('cpu'), n_steps=n_steps, models=models[:len(models)-k], y_known=torch.tensor(val_data[:, :, n_outputs:]).float().to('cpu'))
    mse = criterion(torch.tensor(val_data[:, 1:, :n_outputs]).float(), y_preds).mean().item()
    if mse <= best_mse:
        end_idx = len(models)-k
        best_mse = mse

        # choose the smallest time step
for k in range(0, end_idx):
    step_size = np.int64(2**k)
    y_preds = net.vectorized_multi_scale_forecast(torch.tensor(val_data[:, 0, :n_outputs]).float().to('cpu'), n_steps=n_steps, models=models[k:end_idx], y_known=torch.tensor(val_data[:, :, n_outputs:]).float().to('cpu'))
    mse = criterion(torch.tensor(val_data[:, 1:, :n_outputs]).float(), y_preds).mean().item()
    if mse <= best_mse:
        start_idx = k
        best_mse = mse
        

models = models[start_idx:(end_idx+1)]
num_k = len(models)
print('{} models chosen for Multiscale HiTS:'.format(num_k) ) 
print('   k    = {} .. {}'.format(start_idx, end_idx) )
print('  2^k   = {} .. {}'.format(2**start_idx, 2**end_idx ) )
print('t-steps = {} .. {}\n'.format(dt*2**start_idx, dt * 2**end_idx ) )
del val_data

#==========================================================
# Plot Log(MSE) for Multi-scale vs Single
#==========================================================
# multiscale time-stepping with NN
start = time.time()
y_preds, model_key = net.vectorized_multi_scale_forecast(torch.tensor(test_data[:, 0, :n_outputs]).float().to('cpu'), n_steps=n_steps, models=models, y_known=torch.tensor(test_data[:, :, n_outputs:]).float().to('cpu'), key=True)
end = time.time()
multiscale_time = end - start
print('y_preds.shape = {}'.format(y_preds.shape))
print('test_data.shape = {}'.format(test_data.shape))
multiscale_preds_mse = criterion(torch.tensor(test_data[:, 1:, :n_outputs]).float(), y_preds).mean(-1)
# added additional argument to function 'vectorized_multi_scale_forecast( ... , key=True)' in order to data of each individual ResNet
model_key = model_key.detach().numpy()
#model_key_plus = np.delete(model_key, np.argwhere(model_key==0) )


# visualize forecasting error at each time step    
fig = plt.figure(figsize=(plot_x_dim, plot_y_dim))
colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(ks))))
multiscale_err = multiscale_preds_mse.mean(0).detach().numpy()
for k in range(len(preds_mse)):
    err = preds_mse[k]
    err = err.mean(0).detach().numpy()
    rgb = next(colors)
    plt.plot(t, np.log10(err), linestyle='-', color=rgb, linewidth=4, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
plt.plot(t, np.log10(multiscale_err), linestyle='-', color='k', linewidth=4, label='multiscale')
plt.legend(fontsize=legend_fontsize, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.2))
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlabel('time',fontsize=x_label_fontsize)
plt.ylabel('log(MSE)',fontsize=y_label_fontsize)
plt.grid(axis = 'y')
#plt.ylim(ymin=min_log-d_log)
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
    t_model = np.delete(t, np.where(model_key != idx) )
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

#==========================================================
# Print Computation Time (sec)
#=========================================================
print('ensembled multiscale compute time = {:.4f} s'.format(multiscale_time))
for i in range(len(times)):
    print('{:.2f} | single scale compute time = {:.4f} s'.format(step_sizes[i]*dt, times[i]))

