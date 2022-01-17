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
from carla_sim_merge_1 import SinglePathSimulator,RampMergeEnv

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
#######################################################################################################################


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


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.player1 = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.collision_sensor1 = None
        self.lane_invasion_sensor1 = None
        self.gnss_sensor1 = None
        self.camera_manager1 = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = 'vehicle.*'
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        #blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        veh_blueprint = self.world.get_blueprint_library().filter('model3')[0]
        veh_blueprint.set_attribute('role_name', 'hero')
        if veh_blueprint.has_attribute('color'):
            color = random.choice(veh_blueprint.get_attribute('color').recommended_values)
            veh_blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.spawn_actor(veh_blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point =spawn_points[430] #430
            self.player = self.world.spawn_actor(veh_blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        # Spawn the player1
        if self.player1 is not None:
            spawn_point1 = self.player1.get_transform()
            spawn_point1.location.z += 2.0
            spawn_point1.rotation.roll = 0.0
            spawn_point1.rotation.pitch = 0.0
            self.destroy()
            self.player1 = self.world.spawn_actor(veh_blueprint, spawn_point1)
            self.modify_vehicle_physics(self.player1)
        while self.player1 is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point1 = spawn_points[357]
            self.player1 = self.world.spawn_actor(veh_blueprint, spawn_point1)
            self.modify_vehicle_physics(self.player1)

        self.world.tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        ## Set up sensors for actor2
        self.collision_sensor1 = CollisionSensor(self.player1, self.hud)
        self.lane_invasion_sensor1 = LaneInvasionSensor(self.player1, self.hud)
        self.gnss_sensor1 = GnssSensor(self.player1)
        self.camera_manager1 = CameraManager(self.player1, self.hud)
        self.camera_manager1.transform_index = cam_pos_id
        self.camera_manager1.set_sensor(cam_index, notify=False)
        actor_type1 = get_actor_display_name(self.player1)
        self.hud.notification(actor_type1)

    def reset_env(self):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        self.destroy()
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        veh_blueprint = self.world.get_blueprint_library().filter('model3')[0]
        veh_blueprint.set_attribute('role_name', 'hero')
        if veh_blueprint.has_attribute('color'):
            color = random.choice(veh_blueprint.get_attribute('color').recommended_values)
            veh_blueprint.set_attribute('color', color)


        spawn_points = self.map.get_spawn_points()
        # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_point = spawn_points[430] #430
        self.player = self.world.spawn_actor(veh_blueprint, spawn_point)
        self.modify_vehicle_physics(self.player)

        spawn_point1 = spawn_points[357]
        goal1 = spawn_points[416]
        sp1_xyz = spawn_point1.location
        sp1_yaw = spawn_point1.rotation.yaw
        g1_xyz = goal1.location
        g1_yaw = goal1.rotation.yaw
        m = (g1_xyz.y - sp1_xyz.y)/(g1_xyz.x - sp1_xyz.x)
        theta = np.arctan(m)
        d = -20 + 40 * np.random.rand(1)[0]
        sp1 = [sp1_xyz.x + d*np.cos(theta) , sp1_xyz.y + d*np.sin(theta) , 0.3 ]
        sp1_l = carla.Vector3D(x=sp1[0],y = sp1[1],z=sp1[2])
        yaw1 = sp1_yaw + ((g1_yaw - sp1_yaw)/(g1_xyz.x - sp1_xyz.x))*(d*np.cos(theta))
        sp1_r = carla.Rotation(pitch = 0, yaw = yaw1 , roll = 0)
        sp_transform = carla.Transform(location = sp1_l,rotation = sp1_r)
        self.player1 = self.world.spawn_actor(veh_blueprint, sp_transform)
        self.modify_vehicle_physics(self.player1)

        self.world.tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.recording = False
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        ## Set up sensors for actor2
        self.collision_sensor1 = CollisionSensor(self.player1, self.hud)
        self.lane_invasion_sensor1 = LaneInvasionSensor(self.player1, self.hud)
        self.gnss_sensor1 = GnssSensor(self.player1)
        self.camera_manager1 = CameraManager(self.player1, self.hud)
        self.camera_manager1.transform_index = cam_pos_id
        self.camera_manager1.set_sensor(cam_index, notify=False)
        actor_type1 = get_actor_display_name(self.player1)
        self.hud.notification(actor_type1)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player,self.camera_manager1.sensor,
            self.collision_sensor1.sensor,
            self.lane_invasion_sensor1.sensor,
            self.gnss_sensor1.sensor,
            self.player1]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def destroy_host(self):
        actors = [
            self.camera_manager1.sensor,
            self.collision_sensor1.sensor,
            self.lane_invasion_sensor1.sensor,
            self.gnss_sensor1.sensor,
            self.player1]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def reset_host(self):
        # Keep same camera config if the camera manager exists.
        self.destroy_host()
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        veh_blueprint = self.world.get_blueprint_library().filter('model3')[0]
        veh_blueprint.set_attribute('role_name', 'hero')
        if veh_blueprint.has_attribute('color'):
            color = random.choice(veh_blueprint.get_attribute('color').recommended_values)
            veh_blueprint.set_attribute('color', color)

        spawn_points = self.map.get_spawn_points()
        # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_point1 = spawn_points[357]
        goal1 = spawn_points[416]
        sp1_xyz = spawn_point1.location
        sp1_yaw = spawn_point1.rotation.yaw
        g1_xyz = goal1.location
        g1_yaw = goal1.rotation.yaw
        m = (g1_xyz.y - sp1_xyz.y)/(g1_xyz.x - sp1_xyz.x)
        theta = np.arctan(m)
        d = -20 + 40* np.random.randn(1)[0]
        sp1 = [sp1_xyz.x + d*np.cos(theta) , sp1_xyz.y + d*np.sin(theta) , 0.3 ]
        sp1_l = carla.Vector3D(x=sp1[0],y = sp1[1],z=sp1[2])
        yaw1 = sp1_yaw + ((g1_yaw - sp1_yaw)/(g1_xyz.x - sp1_xyz.x))*(d*np.cos(theta))
        sp1_r = carla.Rotation(pitch = 0, yaw = yaw1 , roll = 0)
        sp_transform = carla.Transform(location = sp1_l,rotation = sp1_r)

        self.player1 = self.world.spawn_actor(veh_blueprint, sp_transform)
        self.modify_vehicle_physics(self.player1)

        self.world.tick()

        self.collision_sensor1 = CollisionSensor(self.player1, self.hud)
        self.lane_invasion_sensor1 = LaneInvasionSensor(self.player1, self.hud)
        self.gnss_sensor1 = GnssSensor(self.player1)
        self.camera_manager1 = CameraManager(self.player1, self.hud)
        self.camera_manager1.transform_index = cam_pos_id
        self.camera_manager1.set_sensor(cam_index, notify=False)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        #raise NotImplementedError
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

def make_env(env_name):
    #     if env_name == 'ant_gather':
    #         return PointGather(**env_args)
    #     elif env_name == 'point_gather':
    #         return PointGatherEnv(**env_args)
    if env_name == 'ramp_merge':
        return RampMergeEnv()
    else:
        pass
########################################################################################################################
pygame.init()
pygame.font.init()
world = None

try:

    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)

    traffic_manager = client.get_trafficmanager()
    sim_world = client.get_world()

    settings = sim_world.get_settings()
  #  settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    sim_world.apply_settings(settings)

  #  traffic_manager.set_synchronous_mode(True)

    display = pygame.display.set_mode(
        (1280, 720),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    hud = HUD(1280, 720)
    actors = client.get_world().get_actors()

    for a in actors:
        print(a.type_id)
        if 'vehicle' in a.type_id:
            a.destroy()

    world = World(client.get_world(), hud)
    controller = KeyboardControl(world)

    agent = basicagent_lat(world.player)
    agent1 = BasicAgent(world.player1)

    # Set the agent destination
    spawn_points = world.map.get_spawn_points()
    # destination = random.choice(spawn_points).location
    destination = spawn_points[416].location
    destination1 = spawn_points[416].location
    agent.set_destination(destination)
    agent1.set_destination(destination1)
    clock = pygame.time.Clock()

    env = RampMergeEnv(agent, agent1, world)
    ########################################################################################################################
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


    ####################################################################################################################



    simulator = SinglePathSimulator(env_name,env, clock,policy, n_trajectories, trajectory_len)
    cpo = CPO(policy, value_fun, cost_fun, simulator, model_name=model_name,
               max_constraint_val=max_constraint_val)

    print(f'Training policy {model_name} on {env_name} environment...\n')

    actions = cpo.train(200)

    plt.plot(np.asarray([i.numpy()[-2]for i in cpo.simulator.traj[200].observations]))
    plt.savefig('traj.jpg')
    plt.close()
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
    plt.close()

    plt.figure(figsize=(10,7))
    plt.plot(cpo.violation,label='CPO')
    plt.legend()
    plt.xlabel('The Number of Epoch',fontsize=20)
    plt.ylabel('Eposide Constraint Violation Distance',fontsize=20)
    plt.grid()
    plt.savefig('Episode_Constraint_Violation.jpg')
finally:
    if world is not None:
        world.destroy()
    pygame.quit()