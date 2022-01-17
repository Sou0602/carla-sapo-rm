# SAPO-RM
implementation of SAPO-RM
SAMP-RM is a CBF based safe-RL algorithm for a simple 1-d stochastic ramp merging scenario. 
The code is largely based on [a pytorch implementation of Constrained Policy Optimization](https://github.com/ajlangley/cpo-pytorch). 
This repo contains the whole repo of [CPO](https://github.com/ajlangley/cpo-pytorch) and the implementation of SAPO-RM. 
All results of Probabilistic Safe Reinforcement Learning using Control Barrier Function for Autonomous Vehicle Ramp Merging Control can be reproduced from this repo. 
## code structure
The SAPO_RM.py is the implementation of SAPO-RM. CPO.py is the implementation of CPO. You can get the result of CPO and SAPO-RM by directly run Experiment_CPO.ipynb and Experiment_SAPO_RM.ipynb respectively. 
infeasible_initial_best.pth is a infeasible initial policy used to test the performance of CPO and SAPO-RM. 
plot.ipynb used to gather and visulize the result of the experiments. 

## Added CARLA Compatibility
CPO_CARLA_Sim.py and SAPO_CARLA_Sim.py
