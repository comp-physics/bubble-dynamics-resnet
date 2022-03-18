import torch
import numpy as np
import scipy.interpolate
from utils import DataSet


class NNBlock(torch.nn.Module):
    def __init__(self, arch, activation=torch.nn.ReLU()):
        """
        :param arch: architecture of the nn_block
        :param activation: activation function
        """
        super(NNBlock, self).__init__()

        # param
        self.n_layers = len(arch)-1
        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # network arch
        for i in range(self.n_layers):
            self.add_module('Linear_{}'.format(i), torch.nn.Linear(arch[i], arch[i+1]).to(self.device))

    def forward(self, x):
        """
        :param x: input of nn
        :return: output of nn
        """
        for i in range(self.n_layers - 1):
            x = self.activation(self._modules['Linear_{}'.format(i)](x))
        # no nonlinear activations in the last layer
        x = self._modules['Linear_{}'.format(self.n_layers - 1)](x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, arch, dt, step_size, activation=torch.nn.ReLU()):
        """
        :param arch: a list that provides the architecture; e.g. [ 3, 128, 128, 128, 2 ]
        :param dt: time step unit
        :param step_size: forward step size
        :param activation: activation function in neural network
        """
        super(ResNet, self).__init__()

        # check consistencies
        assert isinstance(arch, list)
        assert arch[0] >= arch[-1]  # (Scott Sims) originally "==", but changed for 3-inputs to 2-outputs

        # param
        self.n_inputs = arch[0]
        self.n_outputs = arch[-1]

        # data
        self.dt = dt
        self.step_size = step_size

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # layer
        self.activation = activation
        self.add_module('increment', NNBlock(arch, activation=activation))

    def check_data_info(self, dataset):
        """
        :param: dataset: a dataset object
        :return: None
        """
        assert self.n_inputs == dataset.n_inputs
        assert self.dt == dataset.dt
        assert self.step_size == dataset.step_size

    def forward(self, x_init):
        """
        :param x_init: array of shape batch_size x n_outputs (Scott Sims)
        :return: next step prediction of shape batch_size x n_inputs
        """
        return x_init[:, 0:self.n_outputs] + self._modules['increment'](x_init)

    def uni_scale_forecast(self, x_init, n_steps, y_known=None):
        """
        :param x_init: array of shape n_test x n_output
        :param n_steps: number of steps forward in terms of dt
        :param y_known: array of shape n_test x n_steps x (n_inputs - n_outputs)
        :return: predictions of shape n_test x n_steps x n_outputs and the steps
        """
        if y_known is None:
            assert (self.n_inputs == self.n_outputs)
        else:
            assert (self.n_inputs > self.n_outputs)
            assert y_known.shape[0] == x_init.shape[0]
            assert y_known.shape[1] > n_steps
            assert y_known.shape[2] > 0

        steps = list()
        preds = list()
        sample_steps = range(n_steps)      # [ 0, 1, ..., (n-1) ] indexes smallest time-steps [ 0dt, 1dt, ... , (n-1)dt ]

        # forward predictions
        if y_known is None:
            x_prev = x_init
        else:
            x_prev = torch.column_stack((x_init, y_known[:, 0, :]))
        #---------------------------------------------------------
        cur_step = self.step_size - 1      # k := NN step_size multiplier dT = k * dt
        while cur_step < n_steps + self.step_size:
            if y_known is None:  # (Scott Sims) adapted for when n_inputs > n_outputs
                x_next = self.forward(x_prev)  # x(i) = x(i-1) + f( x(i-1) )
            else:
                x_next = torch.column_stack((self.forward(x_prev), y_known[:, cur_step, :]))
            steps.append(cur_step)         # creates a list of indexes [k, 2k, ... , n] for times [k*dt, 2k*dt, ... , n*dt]
            preds.append(x_next[:, :self.n_outputs])           # creates a list of vectors { x(i) } = [x(1), x(2), ... , x(n/k)]
            cur_step += self.step_size     # updates NN step_size: i*k
            x_prev = x_next

        # include the initial frame
        steps.insert(0, 0)
        preds.insert(0, torch.tensor(x_init).float().to(self.device))

        # interpolations
        preds = torch.stack(preds, 2).detach().numpy()
        #preds = preds[:, :, 0:self.n_outputs]
        cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
        y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()
        return y_preds

    def train_net(self, dataset, max_epoch, batch_size, w=1.0, lr_max=1e-3, lr_min=1e-4, model_path=None, min_loss=1e-8, record=False, record_period=100):
        """
        :param dataset: a dataset object
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param w: l2 error weight
        :param lr: learning rate
        :param model_path: path to save the model
        :param record: directs train_net() to return a record of loss-function values with some frequency
        :param record_period: how often loss-function values are recorded (unit = epochs) 
        :return: None
        """
        #-----------------------------------------------------
        # (Scott Sims)
        if(record == True):
            machine_epsilon = np.finfo(np.float64).eps
            n_record = 0
            max_record = np.ceil( max_epoch/record_period)
            record_loss = np.zeros( [max_record, self.n_inputs] )
        #-----------------------------------------------------
        # check consistency
        self.check_data_info(dataset)

        # training
        lr_exp_max = np.round(np.log10(lr_min, decimals=1) # expected to be negative
        lr_exp_min = np.round(np.log10(lr_max, decimals=1) # expoected to be negative
        num_exp = np.int(1+np.round(np.abs(lr_exp_max - lr_exp_min))) # number of different learning rates
        best_loss = 1e+5
        count_no_gain = 0
        for j in range(num_exp):
        # ========== initialize learning rate ================    
            lr = 10.0**(lr_exp_max-j)
            print("=========================")
            print(f"learning rate = {lr}")
            print("=========================")
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            epoch = 0
            while epoch < max_epoch:
                epoch += 1
                # ================= prepare data ==================
                n_samples = dataset.n_train
                new_idxs = torch.randperm(n_samples)
                batch_x = dataset.train_x[new_idxs[:batch_size], :]
                batch_ys = dataset.train_ys[new_idxs[:batch_size], :, :]
                # =============== calculate losses ================
                train_loss = self.calculate_loss(batch_x, batch_ys, w=w)
                val_loss = self.calculate_loss(dataset.val_x, dataset.val_ys, w=w)
                # ================ early stopping =================
                if best_loss <= min_loss:
                    print('--> model has reached an accuracy of 1e-8! Finished training!')
                    break
                # =================== backward ====================
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # =================== log =========================        
                if epoch % 1000 == 0:
                    print('epoch {}, training loss {}, validation loss {}'.format(epoch, train_loss.item(), val_loss.item()))
                    if val_loss.item() < best_loss:
                        best_loss = val_loss.item()
                        count_no_gain = 0
                        if model_path is not None:
                            print('(--> new model saved @ epoch {})'.format(epoch))
                            torch.save(self, model_path)
                    else:
                        count_no_gain += 1 # counts how many thousand epochs with no improvement in loss
                    #-------------------------------
                    if count_no_gain >= int(np.round(0.5*max_epoch/1000) # this number will be in thousands of epochs
                        print('No improvement for many epochs. Trying next learning rate') 
                        break
                    
                # =================== record ======================
                # (Scott Sims)
                if (record == True) and (epoch % record_period == 0):
                    n_record += 1
                    record_loss[n_record, :] = np.array( [epoch, val_loss.item() , train_loss.item()] )
                #--------------------------------------------------
                    # if to save at the end
             # ====================== end while loop ====================
             if val_loss.item() < best_loss and model_path is not None:
                 print('--> new model saved @ epoch {}'.format(epoch))
                 torch.save(self, model_path)

        #------------------------------------------------------
        # (Scott Sims)
        if (record == True):
            n_record += 1
            record_loss[n_record, :] = np.array( [ epoch, val_loss.item(), train_loss.item() ] )
            return record_loss[range(n_record),:]
        #------------------------------------------------------

    def calculate_loss(self, x_init, ys, w=1.0):
        """
        :param x: x batch, array of size batch_size x n_inputs
        :param ys: ys batch, array of size batch_size x n_steps x n_inputs
        :return: overall loss
        """
        batch_size, n_steps, n_inputs = ys.size()
        assert n_inputs == self.n_inputs

        if (n_inputs == self.n_outputs):
            y_known = None
        elif (n_inputs > self.n_outputs):
            y_known = ys[:, :, self.n_outputs:]
        else:
            assert n_inputs >= self.n_outputs  # should be FALSE which will terminate execution

        # forward (recurrence)
        y_preds = torch.zeros(batch_size, n_steps, n_inputs).float().to(self.device)
        y_prev = x_init
        for t in range(n_steps):
            if y_known is None:  # (Scott Sims) adapted for when n_inputs > n_outputs
                y_next = self.forward(y_prev)  # x(i) = x(i-1) + f( x(i-1) )
            else:
                y_next = torch.column_stack((self.forward(y_prev), y_known[:, t, :])) # ENSURE THAT t IS ACCRUATE BY CHECKING utils.py
            y_preds[:, t, :] = y_next
            y_prev = y_next

        # compute loss
        criterion = torch.nn.MSELoss(reduction='none')
        loss = w * criterion(y_preds[:, :, :self.n_outputs], ys[:, :, :self.n_outputs]).mean() + (1-w) * criterion(y_preds[:, :, :self.n_outputs], ys[:, :, :self.n_outputs]).max()
        return loss


def multi_scale_forecast(x_init, n_steps, models):
    """
    :param x_init: initial state torch array of shape n_test x n_inputs
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_inputs
    
    This function is not used in the paper for low efficiency,
    we suggest to use vectorized_multi_scale_forecast() below.
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_extended_steps = n_steps + min(step_sizes)
    sample_steps = range(1, n_steps+1)

    steps = list()
    preds = list()
    steps.insert(0, 0)
    preds.insert(0, torch.tensor(x_init).float().to(device))
    for model in models:
        tmp_steps = list()
        tmp_preds = list()
        for j in range(len(steps)):
            if j < len(steps) - 1:
                end_step = steps[j+1]
            else:
                end_step = n_extended_steps
            # starting point
            cur_step = steps[j]
            cur_x = preds[j]
            tmp_steps.append(cur_step)
            tmp_preds.append(cur_x)
            while True:
                step_size = model.step_size
                cur_step += step_size
                if cur_step >= end_step:
                    break
                cur_x = model(cur_x)
                tmp_steps.append(cur_step)
                tmp_preds.append(cur_x)
        # update new predictions
        steps = tmp_steps
        preds = tmp_preds

    # interpolation
    preds = torch.stack(preds, 2).detach().numpy()
    cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
    y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()
    return y_preds

def vectorized_multi_scale_forecast(x_init, n_steps, models, y_known=None, key=False):
    """
    :param x_init: initial state torch array of shape n_test x n_outputs
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :param y_known:
    :param key (optional): directs function to return a 2nd object, 'model_key', 
        a list with an model-index for each time-point
    :return: a torch array of size n_test x n_steps x n_inputs (tensor)
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # we assume models are sorted by their step sizes (decreasing order)
    n_test, n_inputs = x_init.shape                         # n_test = number of x(0) values to test; n_inputs = dimension of each x0
    if y_known is not None:
        assert (n_steps+1) == y_known.shape[1]
        n_known = y_known.shape[2]
        n_outputs = n_inputs - n_known
    #-------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    indices = list()
    extended_n_steps = n_steps + models[0].step_size
    preds = torch.zeros(n_test, extended_n_steps + 1, n_inputs).float().to(device)
    
    #-------------------------
    # (Scott Sims)
    model_idx = int(0)
    if(key == True):
        # model_key = [0]*(extended_n_steps+1)
        model_key = torch.zeros(extended_n_steps+1, dtype=torch.int8).to(device)
    #-------------------------
    # vectorized simulation
    indices.append(0)
    preds[:, 0, :] = x_init
    total_step_sizes = n_steps
    for model in models:                                              # for each model (largest 'step_size' first)
        n_forward = int(total_step_sizes/model.step_size)             # pick how many steps forward (rounded down)
        y_prev = preds[:, indices, :].reshape(-1, n_inputs)              # initialize y_prev to the end of last prediction
        indices_lists = [indices]                                     # initialize indices_lists (indices = 0)
        model_idx += int(1)                                           # (Scott Sims) used when optional argument 'key' == True
        for t in range(n_forward):                                    # for t-steps forward
            shifted_indices = [x + (t + 1) * model.step_size for x in indices] # shift 'indices' forward 1 step_size
            indices_lists.append(shifted_indices)                     # add shifted 'indices' to 'indices_lists'
            if y_known is None:  # (Scott Sims) adapted for when n_inputs > n_outputs
                y_next = model(y_prev)  # y(i) = y(i-1) + f( y(i-1) )
            else:
                y_next = torch.column_stack((model(y_prev), y_known[:, shifted_indices, :].reshape(-1, n_known)))   # y(i) = y(i-1) + f( y(i-1) )
            #-------------------------
            # (Scott Sims)
            if( key == True ):
                for x in shifted_indices:
                    model_key[x] = model_idx                 # update model indices
            #-------------------------
            preds[:, shifted_indices, :] = y_next.reshape(n_test, -1, n_inputs) # store prediction y(i)
            y_prev = y_next                                           # prepare for next iteration (i+1)
        indices = [val for tup in zip(*indices_lists) for val in tup] # indices = values in tuple, for tuples in indices_list
        total_step_sizes = model.step_size - 1                        # reduce total_step_sizes for next model (finer)
        # NOTE: about zip(*list): "Without *, you're doing zip( [[1,2,3],[4,5,6]] ). With *, you're doing zip([1,2,3], [4,5,6])."

    # simulate the tails
    last_idx = indices[-1]
    y_prev = preds[:, last_idx, :]
    last_model = models[-1]
    while last_idx < n_steps:
        last_idx += last_model.step_size
        if y_known is None:  # (Scott Sims) adapted for when n_inputs > n_outputs
            y_next = last_model(y_prev)  # y(i) = y(i-1) + f( y(i-1) )
        else:
            y_next = torch.column_stack((last_model(y_prev), y_known[:, last_idx, :]))  # y(i) = y(i-1) + f( y(i-1) )
        preds[:, last_idx, :] = y_next
        indices.append(last_idx)
        y_prev = y_next
        #-------------------------
        # (Scott Sims)
        if( key == True ):
            model_key[last_idx] = model_idx                   # update model indices
        #-------------------------

    # interpolations
    sample_steps = range(1, n_steps+1)
    if y_known is None:
        valid_preds = preds[:, indices, :].detach().numpy() 
    else:
        valid_preds = preds[:, indices, :n_outputs].detach().numpy()  # (Scott Sims) modified by parameter 'n_outputs'
    cs = scipy.interpolate.interp1d(indices, valid_preds, kind='linear', axis=1)
    y_preds = torch.tensor( cs(sample_steps) ).float()

    #-------------------------
    # (Scott Sims) 
    # https://www.kite.com/python/answers/how-to-access-multiple-indices-of-a-list-in-python
    if( key == True ):
        # model_key = list( map(model_key.__getitem__, sample_steps) ) # used initially when model_key was a list
        return y_preds, model_key[sample_steps]
    else:
        return y_preds    
    # https://note.nkmk.me/en/python-function-return-multiple-values/



    #-------------------------
    # (Scott Sims) ARCHIVED CODE UNUSED
    #tensor_indices = torch.tensor( model_indices ).float()
    #  reshape vector before transforming into matrix
    #tensor_indices = tensor_indices.reshape(1, len(sample_steps), 1)
    #  tile vector into matrix
    #tensor_indices = torch.tile( tensor_indices, (1,1,3) )
    #  concatenate matrix onto prediction tensor
    #y_preds = torch.cat( (y_preds, tensor_indices) , axis=0 ) 
    #-------------------------




