#!/usr/bin/env python
# coding: utf-8

# ## Train ResNets

# ### created by Yuying Liu, 04/30/2020

# This script is a template for training neural network time-steppers for different systems and different time scales. To reproduce the results in the paper, one needs to obtain all 11 neural network models for each nonlinear system under study. For setup details, please refer to Table 2 in the paper.
import os
import sys
import torch
import numpy as np

module_path = os.path.abspath(os.path.join(os.getcwd(),'src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net

#=========================================================
# Command Line Arguments
#=========================================================
num_arg = len(sys.argv)
if num_arg != 7:
    msg = "Expected 7 arguments but counted {}.".format(num_arg)
    msg = msg + "\n| system | dt | k_max | num_inputs | num_layers | layer_size | batch_size |"
    sys.exit(msg)

system = sys.argv[0]
dt = sys.argv[1]
k_max = sys.argv[2]
num_inputs = sys.argv[3]
num_layers = sys.argv[4]
layer_size = sys.argv[5]
batch_size = sys.argv[6]

print("ResNet Architecture: {0:}-in | {1:}x{2:} | {3:}-out".format(num_inputs, num_layers, layer_size, num_inputs))
arch = [num_inputs]
for j in range(num_layers):
    arch.append(layer_size)
arch.append(num_inputs)

lr = 1e-3                     # learning rate
max_epoch = 100000            # the maximum training epoch

#=========================================================
# Directories and Paths
#=========================================================
data_dir = os.path.join(os.getcwd(), '/data/', system)
model_dir = os.path.join(os.getcwd(), '/models/', system)
if not os.path.exists(data_dir):
    sys.exit("Cannot find folder ../data/{} in current directory".format(system))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#=========================================================
# Load Data
#=========================================================
train_data = np.load(os.path.join(data_dir, 'train.npy'))
val_data = np.load(os.path.join(data_dir, 'val.npy'))
test_data = np.load(os.path.join(data_dir, 'test.npy'))
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]
n_steps = test_data.shape[1] - 1

#=========================================================
# Train Models, each with step_size = 2^k
#=========================================================
for k in range(k_max+1):

    step_size = 2**k
    num_steps = np.int64(n_steps / step_size)

# create dataset object
    dataset = net.DataSet(train_data, val_data, test_data, dt, step_size, num_steps)
    model_name = 'model_D{}_{}_{}x{}.pt'.format(step_size,num_inputs,num_layers,layer_size)

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

