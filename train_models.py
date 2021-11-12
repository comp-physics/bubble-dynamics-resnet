# ## updated by Scott Sims, 11/09/2021
# ## created by Yuying Liu, 04/30/2020

import os
import sys
import torch
import numpy as np
import yaml
from shutil import copyfile

module_path = os.path.abspath(os.path.join(os.getcwd(),'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import ResNet as net

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
steps_max = dictionary['steps_max']
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
batch_size = dictionary['batch_size']
num_layers = dictionary['num_layers']
layer_size = dictionary['layer_size']
num_inputs = dictionary['num_inputs']
#---------------------------------------
print("ResNet Architecture: {0:}-in | {1:}x{2:} | {3:}-out".format(num_inputs, num_layers, layer_size, num_inputs))
arch = [num_inputs]
for j in range(num_layers):
    arch.append(layer_size)
arch.append(num_inputs)
#---------------------------------------
lr = 1e-3                     # learning rate
max_epoch = 100000            # the maximum training epoch
#=========================================================
# Directories and Paths
#=========================================================
n_steps = np.int64(steps_min * 2**k_max)
data_folder = 'data_dt={}_steps={}_P={}-{}_R={}-{}_(train|val|test)=({}|{}|{}).pt'.format(dt, n_steps, P_min, P_max, R_min, R_max, n_train, n_val, n_test)
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
model_folder = 'models_dt={}_steps={}_P={}-{}_R={}-{}_inputs={}_resnet={}x{}.pt'.format(dt, n_steps, P_min, P_max, R_min, R_max, num_inputs, num_layers, layer_size)
model_dir = os.path.join(os.getcwd(), 'models', model_folder)
if not os.path.exists(data_dir):
    sys.exit("Cannot find folder ../data/{} in current directory".format(data_folder))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

param_source = os.path.abspath(os.path.join(os.getcwd(), "parameters.yml"))
param_dest = os.path.abspath(os.path.join(model_dir, "parameters.yml"))
copyfile(param_source, param_dest)

#=========================================================
# Load Data
#=========================================================
train_data = np.load(os.path.join(data_dir, 'train.npy'))
val_data = np.load(os.path.join(data_dir, 'val.npy'))
test_data = np.load(os.path.join(data_dir, 'test.npy'))
n_train = train_data.shape[0]
print('n_train = {}'.format(n_train) )
n_val = val_data.shape[0]
n_test = test_data.shape[0]
n_steps = test_data.shape[1] - 1

#=========================================================
# Train Models, each with step_size = 2^k
#=========================================================
for k in range(k_max+1):

    step_size = 2**k
    model_steps = np.int64(n_steps / step_size)
    model_steps = np.min( [model_steps, steps_max] )

    # create dataset object
    dataset = net.DataSet(train_data, val_data, test_data, dt, step_size, model_steps)
    model_name = 'model_D{}.pt'.format(step_size)

    # create/load model object
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(os.path.join(model_dir, model_name), map_location=device)
        model.device = device
    except:
        print('create model {} ...'.format(model_name))
        model = net.ResNet(arch=arch, dt=dt, step_size=step_size)

    # training
    model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name))

import plot_models