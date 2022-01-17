import numpy as np
import copy


class Vehicle(object):
    def __init__(self, x_pos, y_pos, velocity):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.velocity = velocity
    
    def set_info(self, x_pos, y_pos, velocity):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.velocity = velocity
    

class RampMergeEnv(object):
    '''
    A uncertain environment for a simple ramp merging scenario that contains 1 ego vehicle and 1 host vehicle. 
    The host vihecle has constant velocity. 
    '''
    def __init__(self, ego:Vehicle, host:Vehicle, uncertainty = 0.01,expected_a = 10,deltat:float = 0.01 ,safethreshold=8, speedlimit=35):
        '''
        initialize the environment
        param:
        ego: a vehicle object for ego vehicle
        host: a vehicle object for host vehicle
        uncertainty: the sigma parameter for the gaussian noice in the environment dynamics
        expected_a : expected acceleration of the ego vehicle
        deltat: the time interval for each time step. 
        safethreshold: the safe distance between the ego and host vehicle
        speedlimit: speedlimit for the environment 
        '''
        self.mergepoint= (0,70)
        self.host = host
        self.ego = ego
        self.deltat = deltat
        self.angle = np.arctan(12.5/90)
        self.safethreshold = safethreshold
        self.expected_a = expected_a
        self.sigma = uncertainty
        self.init_ego = copy.deepcopy(ego)
        self.init_host = copy.deepcopy(host)
        self.counter = 0
        self.speedlimit = speedlimit

    def step(self, action):
        '''
        forward the environment of one time step:
        param:
        action: the action chosen by the controller
        return:
        obs: next state
        reward: the reward for current s,a pairs
        done: Boolean represent if the current state is the terminal state 
        info: dictionary contains constraint cost and speed limit of the environment
        '''

        velocity = self.ego.velocity+np.random.normal(0,self.sigma)
        newx = self.ego.x_pos
        newy = self.ego.y_pos + velocity*self.deltat + 0.5*action*self.deltat**2
        velocity += action
        self.ego.set_info(newx,newy,velocity)


        if self.host.x_pos > 0:
                velocity = self.host.velocity +np.random.normal(0, self.sigma)
                newx = self.host.x_pos - velocity*np.sin(self.angle)*self.deltat
                newy = self.host.y_pos + velocity*np.cos(self.angle)*self.deltat
                self.host.set_info(newx,newy,velocity)
        else:
            newx = self.host.x_pos
            newy = self.host.y_pos + self.host.velocity*self.deltat
            velocity = self.host.velocity
        self.host.set_info(newx,newy,velocity)



        obs = (self.ego.y_pos.item() ,self.host.x_pos ,self.host.y_pos, self.ego.velocity.item(), self.host.velocity)
        reward = self.reward(obs,action)
        done = True if self._max_episode_steps and self.counter >= self._max_episode_steps else False
        info = {'constraint_cost':self.cost(obs,action),'speed_limit':self.speedlimit}
        self.counter += 1

        return obs, reward, done, info
        
    def reset(self):
        '''
        reset the environment to the intial state
        return: numpy.array the initial state s
        
        '''
        self.counter = 0
        self.ego = copy.deepcopy(self.init_ego)
        self.host = copy.deepcopy(self.init_host)
        s = (self.ego.y_pos, self.host.x_pos , self.host.y_pos, self.ego.velocity, self.host.velocity)
        return s
    
    def reward(self,s,a):
        '''
        compute the reward
        param:
        s: the environment state
        a: the chosen action
        return
        reward: numpy.array, the dimension depends on the input dimension
        '''
        return - (a - self.expected_a)**2

    def cost(self,s,a):
        '''
        compute the cost for cost based algorithms
        param:
        s: the environment state
        a: the chosen action
        '''
        return (s[0]-s[2])**2 + (s[1] - 0)**2 < self.safethreshold**2
    
    def next_step(self, state, action):
        '''
        used for bootstrap. Get next state without random noise and without change the environment, 
        param:
        state: environment state
        action: the chosen action
        return:
        obs: next state
        '''
        ego_velocity = state[-2]
        ego_newy = state[0] + ego_velocity*self.deltat + 0.5*action*self.deltat**2
        ego_velocity += action



        if state[1] > 0:
            host_velocity = state[-1]
            host_newx = state[1] - host_velocity*np.sin(self.angle)*self.deltat
            host_newy = state[2] + host_velocity*np.cos(self.angle)*self.deltat
        else:
            host_newx = state[1]
            host_newy = state[2] + host_velocity*self.deltat
            



        obs = (ego_newy ,host_newx ,host_newy, ego_velocity, host_velocity)

        return obs

