import torch
import random
import numpy as np


class DataSet:
    def __init__(self, train_data, val_data, test_data, dt, step_size, n_forward):
        """
        :param train_data: array of shape n_train x train_steps x input_dim
                            where train_steps = max_step x (n_steps + 1)
        :param val_data: array of shape n_val x val_steps x input_dim
        :param test_data: array of shape n_test x test_steps x input_dim
        :param dt: the unit time step
        :param step_size: an integer indicating the step sizes
        :param n_forward: number of steps forward
        """
        n_train, train_steps, n_inputs = train_data.shape
        n_val, val_steps, _ = val_data.shape
        n_test, test_steps, _ = test_data.shape
        #(Scott Sims) replaced redundant calculations with 'num_steps'
        num_steps = step_size*n_forward + 1
        assert num_steps <= train_steps and num_steps <= val_steps

        # params
        self.dt = dt
        self.n_inputs = n_inputs #(Scott Sims) updated n_dim to n_inputs
        #self.n_outputs = n_outputs #(Scott Sims) update may not be needed
        self.step_size = step_size
        self.n_forward = n_forward
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # data
        t_idx = 0
        y_start_idx = t_idx + step_size
        y_end_idx = t_idx + num_steps
        self.train_x = torch.tensor(train_data[:, t_idx, :]).float().to(self.device)
        self.train_ys = torch.tensor(train_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
        self.val_x = torch.tensor(val_data[:, t_idx, :]).float().to(self.device)
        self.val_ys = torch.tensor(val_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
        self.test_x = torch.tensor(test_data[:, 0, :]).float().to(self.device)
        self.test_ys = torch.tensor(test_data[:, 1:, :]).float().to(self.device)
