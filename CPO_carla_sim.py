from argparse import ArgumentParser
import matplotlib.pyplot as plt
#from envs.ramp_merge import Vehicle
import gym
import torch
from yaml import load
from cpo import CPO
from memory import Memory
from models import build_diag_gauss_policy, build_mlp
#from simulators import SinglePathSimulator
from torch_utils.torch_utils import get_device
import torch.nn as nn
from torch.nn import Linear, LogSoftmax, Module, Parameter, Sequential, Tanh
from torch.distributions.normal import Normal
from torch.distributions import Independent
import seaborn as sns
import numpy as np
from carla_sim_merge import SinglePathSimulator

state_dim = 5
action_dim = 1

n_episodes = 1
env_name = 'ramp_merge'
model_name = 'cpo'
n_trajectories = 1
trajectory_len = 1000
policy_dims = [120]
vf_dims = [80]
cf_dims = [80]
max_constraint_val = 0.1


deltat=0.01
speed_limit = 35
feasible_set = (8,-15)
# bias_red_cost = config['bias_red_cost']
device = get_device()



class DiagGaussianLayer(Module):
    '''
    Implements a layer that outputs a Gaussian distribution with a diagonal
    covariance matrix

    Attributes
    ----------
    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix

    Methods
    -------
    __call__(mean)
        takes as input a mean vector and outputs a Gaussian distribution with
        diagonal covariance matrix defined by log_std

        '''

    def __init__(self, output_dim=None, log_std=None):
        Module.__init__(self)

        self.log_std = log_std

        if log_std is None:
            self.log_std = Parameter(torch.zeros(output_dim), requires_grad=True)

    def __call__(self, mean):
        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)

        return normal_dist


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, speed_limit=speed_limit, feasible_set=feasible_set,
                 deltat=deltat):
        super().__init__()
        layer_sizes = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

            if i != len(layer_sizes) - 2:
                layers.append(Tanh())
        layers.append(Tanh())
        self.layer = Sequential(*layers)
        self.gaussian = DiagGaussianLayer(output_dim=output_dim)
        self.speed_limit = speed_limit
        self.u_max, self.u_min = feasible_set
        self.deltat = deltat

    def forward(self, x):
        u_max, u_min = self.compute_action_space(x)
        x = self.layer(x)
        x = ((u_max - u_min) / 2) * x + (u_max + u_min) / 2
        #         x = 8*x
        try:
            x = self.gaussian(x)
        except:
            print(x)

        return x

    def compute_action_space(self, x):

        v = x[:, -2]
        upper = (self.speed_limit - v) / self.deltat
        upper[upper > self.u_max] = self.u_max
        upper[upper < self.u_min] = self.u_min
        u_max = upper
        lower = (0 - v) / self.deltat
        lower[lower < self.u_min] = self.u_min
        lower[lower > self.u_max] = self.u_max
        u_min = lower
        return u_max.reshape(-1, 1), u_min.reshape(-1, 1)

def build_policy(input_dim, hidden_dims, output_dim):
    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

        if i != len(layer_sizes) - 2:
            layers.append(Tanh())
    layers.append(Tanh())
    return Sequential(*layers)

# policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)

policy = Net(state_dim, policy_dims, action_dim)
value_fun = build_mlp(state_dim + 1, vf_dims, 1)
cost_fun = build_mlp(state_dim + 1, cf_dims, 1)

policy.to(device)
value_fun.to(device)
cost_fun.to(device)

#policy.load_state_dict(torch.load('infeasible_initial.pth'))

x1,y1,v1,x2,y2,v2 = (0,-90,20,12.5,-90,20)
#ego =Vehicle(x1,y1,v1)
#host = Vehicle(x2,y2,v2)



simulator = SinglePathSimulator(env_name, policy, n_trajectories, trajectory_len)
cpo = CPO(policy, value_fun, cost_fun, simulator, model_name=model_name,
           max_constraint_val=max_constraint_val)

print(f'Training policy {model_name} on {env_name} environment...\n')

actions = cpo.train(10)

plt.plot(np.asarray([i.numpy()[-2]for i in cpo.simulator.traj[10].observations]))

import pandas as pd
pd.Series(cpo.violation).to_csv('cpo_violation.csv')
pd.Series(cpo.mean_rewards).to_csv('cpo_mean_reward.csv')

cpo.policy(torch.tensor([[-90,12.5,-90,0,30]]).to(device)).log_prob(torch.tensor([0]).to(device))

plt.figure(figsize=(10,7))
plt.plot(cpo.mean_rewards,label='CPO')
plt.legend()
plt.xlabel('The Number of Epoch',fontsize=15)
plt.ylabel('Average Return',fontsize=15)
plt.grid()
plt.savefig('plot_rewards.jpg')


plt.figure(figsize=(10,7))
plt.plot(cpo.violation,label='CPO')
plt.legend()
plt.xlabel('The Number of Epoch',fontsize=20)
plt.ylabel('Eposide Constraint Violation Distance',fontsize=20)
plt.grid()

