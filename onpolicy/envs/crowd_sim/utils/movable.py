import numpy as np
from numpy.linalg import norm
import abc
import logging
from onpolicy.envs.crowd_sim.crowd_nav.policy.policy_factory import policy_factory
from onpolicy.envs.crowd_sim.utils.action import ActionXY, ActionRot
from onpolicy.envs.crowd_sim.utils.state import ObservableState, FullState

from onpolicy.envs.crowd_sim.utils.entity import Entity
from onpolicy.envs.crowd_sim.utils.state import JointState
import random

class Movable(Entity):
    def __init__(self, config, subconfig):
        """
        Base class for robot and human. Have the physical attributes of an agent.
        """
        super(Movable, self).__init__()
        self.movable = True

        if subconfig.randomize_policy:
            if random.random()<0.5:
                subconfig.policy = 'orca'
            else:
                subconfig.policy = 'social_force'

        if subconfig.randomize_policy_parameter:
            if subconfig.policy == 'orca':
                config.orca.neighbor_dist = np.random.uniform(config.orca.neighbor_dist - config.orca.neighbor_dist_interval, config.orca.neighbor_dist + config.orca.neighbor_dist_interval)
                config.orca.safety_space = np.random.uniform(config.orca.safety_space - config.orca.safety_space_interval, config.orca.safety_space + config.orca.safety_space_interval)
                config.orca.time_horizon = np.random.uniform(config.orca.time_horizon - config.orca.time_horizon_interval, config.orca.time_horizon + config.orca.time_horizon_interval)
                config.orca.time_horizon_obst = np.random.uniform(config.orca.time_horizon_obst - config.orca.time_horizon_obst_interval, config.orca.time_horizon_obst + config.orca.time_horizon_obst_interval)
            elif subconfig.policy == 'social_force':
                config.sf.A = np.random.uniform(config.sf.A - config.sf.A_interval, config.sf.A + config.sf.A_interval)
                config.sf.B = np.random.uniform(config.sf.B - config.sf.B_interval, config.sf.B + config.sf.B_interval)
                config.sf.KI = np.random.uniform(config.sf.KI - config.sf.KI_interval, config.sf.KI + config.sf.KI_interval)

        # # randomize neighbor_dist of ORCA
        # if subconfig.randomize_attributes:
        #     config.orca.neighbor_dist = np.random.uniform(5, 10)
        self.policy = policy_factory[subconfig.policy](config)
        self.sensor = subconfig.sensor
        self.FOV = np.pi * subconfig.FOV

        self.v_pref_interval = subconfig.v_pref_interval
        self.radius_interval = subconfig.radius_interval

        self.kinematics = 'holonomic'

        self.gx = None
        self.gy = None

        self.sensor_range = 0

        self.time_step = config.env.time_step
        self.policy.time_step = config.env.time_step

    # def print_info(self):
    #     logging.info('Agent is ' and has {} kinematic constraint'.format(
    #         'visible' if self.visible else 'invisible', self.kinematics))

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(self.v_pref - self.v_pref_interval, self.v_pref + self.v_pref_interval)
        self.radius = np.random.uniform(self.radius - self.radius_interval, self.radius + self.radius_interval)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.radius = radius
        self.v_pref = v_pref

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_full_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    def get_full_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.theta]
        # return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref]

    def get_goal_position(self):
        return self.gx, self.gy

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy
        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        # unicycle
        else:
            # naive dynamics
            # theta = self.theta + action.r * delta_t # if action.r is w
            # # theta = self.theta + action.r # if action.r is delta theta
            # px = self.px + np.cos(theta) * action.v * delta_t
            # py = self.py + np.sin(theta) * action.v * delta_t

            # differential drive
            epsilon = 0.0001
            if abs(action.r) < epsilon:
                R = 0
            else:
                w = action.r/delta_t # action.r is delta theta
                R = action.v/w

            px = self.px - R * np.sin(self.theta) + R * np.sin(self.theta + action.r)
            py = self.py + R * np.cos(self.theta) - R * np.cos(self.theta + action.r)

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def one_step_lookahead(self, pos, action):
        px, py = pos
        self.check_validity(action)
        new_px = px + action.vx * self.time_step
        new_py = py + action.vy * self.time_step
        new_vx = action.vx
        new_vy = action.vy
        return [new_px, new_py, new_vx, new_vy]

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

class Robot(Movable):
    def __init__(self, config):
        subconfig = config.robot
        super(Robot, self).__init__(config, subconfig)
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius
        self.sensor_range = subconfig.sensor_range
        self.kinematics = subconfig.kinematics
        self.collide = subconfig.collide
        self.sensor_range_robot = subconfig.sensor_range_robot
        self.sensor_range_human = subconfig.sensor_range_human

    def action_callback(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def actWithJointState(self, ob):
        action = self.policy.predict(ob)
        return action

class Human(Movable):
    def __init__(self, config):
        subconfig = config.human
        super(Human, self).__init__(config, subconfig)
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius
        self.sensor_range = subconfig.sensor_range
        self.kinematics = 'holonomic'
        self.collide = subconfig.collide

    def action_callback(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def actWithJointState(self, ob):
        action = self.policy.predict(ob)
        return action

class Camera(Movable):
    def __init__(self, config):
        subconfig = config.cam
        super(Camera, self).__init__(config, subconfig)
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius
        self.sensor_range = subconfig.sensor_range
        self.kinematics = 'holonomic'
        self.collide = subconfig.collide

    def action_callback(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def actWithJointState(self, ob):
        action = self.policy.predict(ob)
        return action