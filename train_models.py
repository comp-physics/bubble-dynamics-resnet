# ## adapted by Scott Sims, 05/19/2022
# ## created by Yuying Liu, 04/30/2020
import os
import sys
import pdb
import torch
import numpy as np
import yaml
from shutil import copyfile
#-----------------------------------------------------
module_path = os.path.abspath(os.path.join(os.getcwd(),'src'))
if module_path not in sys.path:
    sys.path.append(module_path)
#-----------------------------------------------------
from bubble_methods import get_num_steps
import ResNet as net
#-----------------------------------------------------
torch.cuda.empty_cache()
#=========================================================
# Input Arguments
#=========================================================
with open("parameters.yml", 'r') as stream:
    D = yaml.safe_load(stream)

for key in D:
    globals()[str(key)] = D[key]
    print('{}: {}'.format(str(key), D[key]))
    # transforms key-names from dictionary into global variables, then assigns them the dictionary-values
#=========================================================
arch = [n_input]
for j in range(n_layer):
    arch.append(n_neuron)
arch.append(n_output)
print("ResNet Architecture: {}".format(arch))
#print('PRESS [c] TO CONTINUE. PRESS [q] TO QUIT.')
#pdb.set_trace()
#---------------------------------------
max_epoch = 100000     # the maximum training epoch for each batch size
#=========================================================
# Directories and Paths
#=========================================================
n_steps = get_num_steps(dt, model_steps, step_sizes[-1], period_min, n_periods)
data_folder = f"data_dt={dt}_n-steps={n_steps}_m-steps={model_steps}_delta={step_sizes}_n-waves={n_wave}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_train+val+test={n_train}+{n_val}+{n_test}"
data_dir = os.path.join(os.getcwd(), 'data', data_folder)
model_folder = f"models_dt={dt}_steps={n_steps}_m-steps={model_steps}_delta={step_sizes}_n-waves={n_wave}_period={period_min}-{period_max}_amp={amp_min}-{amp_max}_{n_lr}-lr={learn_rate_min}-{learn_rate_max}_n-batches={n_batch}_nresnet={n_input}+{n_layer}x{n_neuron}+{n_output}"
model_dir = os.path.join(os.getcwd(), 'models', model_folder)
if not os.path.exists(data_dir):
    print("current directory:")
    print(os.getcwd())
    sys.exit("Cannot find folder ../data/{data_folder} in current directory".format(data_folder))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#----------------------------------------------------------
parameter_source = os.path.abspath(os.path.join(os.getcwd(), "parameters.yml"))
parameter_destination = os.path.abspath(os.path.join(model_dir, "parameters.yml"))
copyfile(parameter_source, parameter_destination)

#=========================================================
# Start Training
#=========================================================
for k in range(len(step_sizes)):
    #=========================================================
    # Load Data, each with step_size = 2^k
    #=========================================================
    step_size = step_sizes[k]
    train_data = np.load(os.path.join(data_dir, f'train_D{step_size}.npy'))
    val_data = np.load(os.path.join(data_dir, f'val_D{step_size}.npy'))
    test_data = np.load(os.path.join(data_dir, f'test.npy'))
    n_train = train_data.shape[0]
    n_val = val_data.shape[0]
    n_test = test_data.shape[0]
    n_steps = test_data.shape[1] - 1
    batch_size = np.int64(np.round(n_val/n_batch))
    #=========================================================
    # Train Models, each with step_size = 2^k
    #=========================================================
    print("=======================")
    # create dataset object
    dataset = net.DataSet(train_data, val_data, test_data, dt, step_size, model_steps)
    model_name = f'model_D{step_size}.pt'
    # create/load model object
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(os.path.join(model_dir, model_name), map_location=device)
        model.device = device
    except:
        print(f'TRAIN: {model_name} ...')
        model = net.ResNet(arch=arch, dt=dt, step_size=step_size)
    # training
    print(f'training samples: {n_train}')
    print(f'device: {device}')
    model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr_list=np.linspace(learn_rate_max, learn_rate_min, n_lr),
                    model_path=os.path.join(model_dir, model_name))