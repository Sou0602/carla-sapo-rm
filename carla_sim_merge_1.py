#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function
from torch_utils.torch_utils import get_device
from autoassign import autoassign
import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref
from itertools import chain
import torch
import copy

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.basic_agent_lat import BasicAgent as basicagent_lat

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def make_env(env_name):
    #     if env_name == 'ant_gather':
    #         return PointGather(**env_args)
    #     elif env_name == 'point_gather':
    #         return PointGatherEnv(**env_args)
    if env_name == 'ramp_merge':
        return RampMergeEnv()
    else:
        pass


class Simulator:
    @autoassign(exclude=('env_name'))
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, obs_filter=None):
        #self.env = np.asarray([make_env(env_name) for i in range(n_trajectories)])
        self.n_trajectories = n_trajectories

        #for env in self.env:
        #    env._max_episode_steps = trajectory_len

        self.device = get_device()


class SinglePathSimulator:
    def __init__(self, env_name,env, clock,policy, n_trajectories, trajectory_len, state_filter=None,sync = False):
        Simulator.__init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter)
        self.traj = []
        self.max_ep_steps = trajectory_len
        self.env = env
        self.sync = sync
        self.clock = clock
        self.env._max_episode_steps = self.max_ep_steps
        ## start and connect to client

    def run_sim(self):
        def compute_violation(trajectories):
            val = 64 - ((trajectories[:, :, 0] - trajectories[:, :, 2]) ** 2 + (trajectories[:, :, 1] - 0) ** 2)
            val[val < 0] = 0
            return val.sum(1).mean(0)

        self.policy.eval()

        with torch.no_grad():
            trajectories = np.asarray([Trajectory() for i in range(self.n_trajectories)])
            continue_mask = np.ones(self.n_trajectories)
            '''''
            for env, trajectory in zip(self.env, trajectories):
                obs = torch.tensor(env.reset()).float()

                # Maybe batch this operation later
                if self.obs_filter:
                    obs = self.obs_filter(obs)

                trajectory.observations.append(obs)
            '''''

            if self.n_trajectories == 1:
                env = self.env
                trajectory = trajectories[0]
                obs = torch.tensor(env.reset()).float()

                # Maybe batch this operation later
                if self.obs_filter:
                    obs = self.obs_filter(obs)

                trajectory.observations.append(obs)


            ##For one trajectory
            while not trajectory.done:

                self.clock.tick()
                if self.sync:
                    self.env.world.world.tick()
                else:
                    self.env.world.world.wait_for_tick()

                self.env.world.tick(self.clock)
                #self.env.world.render(display)
                #pygame.display.flip()

                #continue_indices = np.where(continue_mask)
                trajs_to_update = trajectories
                continuing_envs = self.env

                policy_input = torch.stack([torch.tensor(trajectory.observations[-1]).to(self.device)
                                            for trajectory in trajs_to_update])
                try:
                    action_dists = self.policy(policy_input)
                    actions = action_dists.sample()
                    actions = actions.cpu()
                except:
                    print('policy input')
                    print(policy_input)
                    print('action dists', actions_dists, sep='\n')

                #for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                if self.n_trajectories == 1:

                    env = continuing_envs
                    action = actions[0]
                    trajectory = trajs_to_update[0]

                    obs, reward, trajectory.done, info = env.step(action.numpy())
                    obs = torch.tensor(obs).float()
                    reward = torch.tensor(reward, dtype=torch.float)
                    cost = torch.tensor(info['constraint_cost'], dtype=torch.float)

                    if self.obs_filter:
                        obs = self.obs_filter(obs)

                    trajectory.actions.append(action)
                    trajectory.rewards.append(reward)
                    trajectory.costs.append(cost)

                    if not trajectory.done:
                        trajectory.observations.append(obs)

                continue_mask = np.asarray([1 - trajectory.done for trajectory in trajectories])

                self.traj.append(trajectories[0])
                maen = np.asarray([[j.numpy() for j in i.observations] for i in trajectories])
                violation = compute_violation(maen)
                memory = Memory(trajectories)


        return memory, violation


####################################################################################################################



class Trajectory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.done = False

    def __len__(self):
        return len(self.observations)


class Memory:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def sample(self):
        observations = torch.cat([torch.stack(trajectory.observations) for trajectory in self.trajectories])
        actions = torch.cat([torch.stack(trajectory.actions) for trajectory in self.trajectories])
        rewards = torch.cat([torch.tensor(trajectory.rewards) for trajectory in self.trajectories])
        costs = torch.cat([torch.tensor(trajectory.costs) for trajectory in self.trajectories])

        return observations, actions, rewards, costs

    def __getitem__(self, i):
        return self.trajectories[i]

    def sample_t(self, i):
        observations = torch.cat([torch.stack([trajectory.observations[i]]) for trajectory in self.trajectories])
        next_obs = torch.cat([torch.stack([trajectory.observations[i + 1]]) for trajectory in self.trajectories])
        actions = torch.cat([torch.stack([trajectory.actions[i]]) for trajectory in self.trajectories])
        rewards = torch.cat([torch.tensor([trajectory.rewards[i]]) for trajectory in self.trajectories])
        costs = torch.cat([torch.tensor([trajectory.costs[i]]) for trajectory in self.trajectories])
        return observations, next_obs, actions, rewards, costs

########################################################################################################################

class RampMergeEnv(object):
    '''
    A uncertain environment for a simple ramp merging scenario that contains 1 ego vehicle and 1 host vehicle.
    The host vihecle has constant velocity.
    '''

    def __init__(self, ego, host, world, uncertainty=0.01, expected_a=0.5, deltat: float = 0.01,
                 safethreshold=8, speedlimit=35):
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
        self.mergepoint = (0, 70)
        self.host = host
        self.ego = ego
        self.deltat = deltat
        self.angle = np.arctan(12.5 / 90)
        self.safethreshold = safethreshold
        self.expected_a = expected_a
        self.sigma = uncertainty
        self.counter = 0
        self.speedlimit = speedlimit
        self.world = world
        self.ego_y_pos = 0
        self.host_x_pos = 0
        self.host_y_pos = 0
        self.ego_velocity = 0
        self.host_velocity = 0

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

        '''''
        velocity = self.ego.velocity + np.random.normal(0, self.sigma)
        newx = self.ego.x_pos
        newy = self.ego.y_pos + velocity * self.deltat + 0.5 * action * self.deltat ** 2
        velocity += action
        self.ego.set_info(newx, newy, velocity)
        '''''

        # Action is the control input:
        if action > 1:
            action = 1
        elif action < -1:
            action = -1
        if isinstance(action,int):
            self.long_control = action
        else:
            self.long_control = action[0].item()
        #print(self.long_control)
        self.control = self.ego.run_step(self.long_control)
        self.control1 = self.host.run_step()
        self.world.player.apply_control(self.control)
        self.world.player1.apply_control(self.control1)
        '''''
        if self.host.x_pos > 0:
            velocity = self.host.velocity + np.random.normal(0, self.sigma)
            newx = self.host.x_pos - velocity * np.sin(self.angle) * self.deltat
            newy = self.host.y_pos + velocity * np.cos(self.angle) * self.deltat
            self.host.set_info(newx, newy, velocity)
        else:
            newx = self.host.x_pos
            newy = self.host.y_pos + self.host.velocity * self.deltat
            velocity = self.host.velocity
        self.host.set_info(newx, newy, velocity)
        '''''
        ##Carla terms and check
        player_loc = self.world.player.get_transform().location
        player1_loc = self.world.player1.get_transform().location
        player_vel = self.world.player.get_velocity()
        player1_vel = self.world.player1.get_velocity()
        #print(player_loc)
        self.ego_y_pos = player_loc.y
        self.host_x_pos = player1_loc.x
        self.host_y_pos = player1_loc.y
        self.ego_velocity = np.sqrt(player_vel.x ** 2 + player_vel.y ** 2) * 3.6
        self.host_velocity = np.sqrt(player1_vel.x ** 2 + player1_vel.y ** 2) * 3.6

        #print("Host_Ego_Dist:" ,np.linalg.norm([self.host_x_pos - player_loc.x, self.host_y_pos - self.ego_y_pos]))
        print("Ego_Goal_Dist:" , np.linalg.norm([player_loc.x - self.world.map.get_spawn_points()[416].location.x,
                                                 player_loc.y - self.world.map.get_spawn_points()[416].location.y]),
                "Host_Goal_Dist:" ,np.linalg.norm([player1_loc.x - self.world.map.get_spawn_points()[416].location.x,
                                                 player1_loc.y - self.world.map.get_spawn_points()[416].location.y]),"Counter:",self.counter)

        obs =  (self.ego_y_pos ,self.host_x_pos ,self.host_y_pos, self.ego_velocity, self.host_velocity)
        reward = self.reward(obs, action)
        done = True if self.counter >= self._max_episode_steps else False
        info = {'constraint_cost': self.cost(obs, action), 'speed_limit': self.speedlimit}
        done = done or self.ego.done() or np.linalg.norm([player_loc.x - self.world.map.get_spawn_points()[416].location.x,
                                                 player_loc.y - self.world.map.get_spawn_points()[416].location.y])  < 1
        self.counter += 1

        if self.host.done():
            self.world.reset_host()
            self.host = BasicAgent(self.world.player1)
            spawn_points = self.world.map.get_spawn_points()
            self.host.set_destination(spawn_points[416].location)

        return obs, reward, done, info

    def reset(self):
        '''
        reset the environment to the intial state
        return: numpy.array the initial state s
        '''

        ## destroy all the actors and respawn to initial positions
        self.world.reset_env()
        self.world.camera_manager.recording = False
        self.counter = 0
        self.ego = basicagent_lat(self.world.player)
        self.host = BasicAgent(self.world.player1)
        spawn_points = self.world.map.get_spawn_points()
        self.ego.set_destination(spawn_points[416].location)
        self.host.set_destination(spawn_points[416].location)
        player_loc = self.world.player.get_transform().location
        player1_loc = self.world.player1.get_transform().location
        player_vel = self.world.player.get_velocity()
        player1_vel = self.world.player1.get_velocity()
        self.ego_y_pos =  player_loc.y
        self.host_x_pos = player1_loc.x
        self.host_y_pos = player1_loc.y
        self.ego_velocity = np.sqrt(player_vel.x**2 + player_vel.y**2) * 3.6
        self.host_velocity = np.sqrt(player1_vel.x**2 + player1_vel.y**2) * 3.6
        s = (self.ego_y_pos, self.host_x_pos, self.host_y_pos, self.ego_velocity, self.host_velocity)
        return s

    def reward(self, s, a):
        '''
        compute the reward
        param:
        s: the environment state
        a: the chosen action
        return
        reward: numpy.array, the dimension depends on the input dimension
        '''
        #print(a)
        spawn = self.world.map.get_spawn_points()
        dest = spawn[416]
        dx = dest.location.x
        dy = dest.location.y
        player_loc = self.world.player.get_transform().location
        ex = player_loc.x
        ey = s[0]
        return - (a - self.expected_a) ** 2 - 4*(ex - dx)**2 - 4*(ey - dy)**2
        #return - 4*(ex - dx)**2 - 4*(ey - dy)**2

    def cost(self, s, a):
        '''
        compute the cost for cost based algorithms
        param:
        s: the environment state
        a: the chosen action
        '''
        return (s[0] - s[2]) ** 2 + (s[1] - 0) ** 2 < self.safethreshold ** 2

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
        ego_newy = state[0] + ego_velocity * self.deltat + 0.5 * action * self.deltat ** 2
        ego_velocity += action

        if state[1] > 0:
            host_velocity = state[-1]
            host_newx = state[1] - host_velocity * np.sin(self.angle) * self.deltat
            host_newy = state[2] + host_velocity * np.cos(self.angle) * self.deltat
        else:
            host_newx = state[1]
            host_newy = state[2] + host_velocity * self.deltat

        obs = (ego_newy, host_newx, host_newy, ego_velocity, host_velocity)

        return obs

########################################################################################################################

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

'''
def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(1280, 720)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        if args.agent == "Basic":
            agent = basicagent_lat(world.player)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)


        agent1 = BasicAgent(world.player1)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        #destination = random.choice(spawn_points).location
        destination = spawn_points[189].location
        destination1 = spawn_points[416].location
        agent.set_destination(destination)
        agent1.set_destination(destination1)
        clock = pygame.time.Clock()
        ##Define environment and simulator

        while True:
            # Use this block in the simulator version
            ##################################################
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            ###############################################

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            throttle_input = 0.25
            control = agent.run_step(throttle_input)
            control.manual_gear_shift = False
            world.player.apply_control(control)
            player_loc = world.player.get_transform().location
            player_vel = world.player.get_velocity()
            player_control = world.player.get_control()
            print(player_control.throttle, player_control.steer,player_vel)


            control1 = agent1.run_step()
            control1.manual_gear_shift = False
            world.player1.apply_control(control1)
            player1_loc = world.player1.get_transform().location
            player1_vel = world.player.get_velocity()
            player1_control = world.player.get_control()


    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Basic")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
'''




