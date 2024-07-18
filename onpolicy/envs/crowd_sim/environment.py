import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from numpy import arctan2
import math

import logging
import gym
import numpy as np
import rvo2
import random
import copy

from numpy.linalg import norm

from onpolicy.envs.crowd_sim.utils.info import *
from onpolicy.envs.crowd_sim.crowd_nav.policy.orca import ORCA
from onpolicy.envs.crowd_sim.utils.state import *
from onpolicy.envs.crowd_sim.utils.action import ActionRot, ActionXY
from onpolicy.envs.crowd_sim.utils.recorder import Recoder

from onpolicy.envs.crowd_sim.utils.state import JointState

import inspect

def get_class_that_defined_method(meth):
    for cls in inspect.getmro(meth.im_class):
        if meth.__name__ in cls.__dict__: 
            return cls
    return None

# update bounds to center around agent
cam_range = 2

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'visualize']
    }

    def __init__(self, world, reset_callback=None, reward_done_info_callback=None,
                 observation_callback=None, reset_agent=None):

        self.world = world
        self.reset_callback = reset_callback
        self.reward_done_info_callback = reward_done_info_callback
        self.observation_callback = observation_callback

        self.humans = world.humans
        
        self.time_limit = None
        self.time_step = None
        self.global_time = None
        self.step_counter = 0


        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.action_space = None
        self.observation_space = None

        #seed
        self.thisSeed = None # the seed will be set when the env is created
        #nenv
        self.nenv = None # the number of env will be set when the env is created.
        # Because the human crossing cases are controlled by random seed, we will calculate unique random seed for each
        # parallel env.

        self.phase = None # set the phase to be train, val or test
        self.test_case = None # the test case ID, which will be used to calculate a seed to generate a human crossing case

        self.render_axis = None

        # to receive data from gst pred model
        self.gst_out_traj = None

        self.last_left = 0.
        self.last_right = 0.

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need
        num_spatial_edges = world.max_human_num + world.num_agents if world.robot_in_traj_robot else world.max_human_num

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        obs_dim = 0
        for i in range(self.world.num_agents):
            high = np.inf * np.ones([2, ])
            self.action_space.append(gym.spaces.Box(-high, high, dtype=np.float32))
            obs_dim = self.observation_callback(i, self.world).shape

            # share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=obs_dim, dtype=np.float32))  # [-inf,inf]

        share_obs_dim = (self.world.num_agents,) + obs_dim  

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=share_obs_dim, dtype=np.float32) for _ in range(self.world.num_agents)]

        self.reset_agent = reset_agent



    def configure(self, args, config):
        """ read the config to the environment variables """

        self.config = config

        self.time_limit = args.episode_length * config.env.time_step
        self.time_step = config.env.time_step # sans doute mieux dans world
        self.randomize_attributes = config.env.randomize_attributes

        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': self.config.env.val_size,
                          'test': self.config.env.test_size}
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.arena_size = config.sim.arena_size

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('Circle width: {}'.format(self.circle_radius))

        # for sim2real dynamics check
        self.record=config.sim2real.record
        self.load_act=config.sim2real.load_act
        self.ROSStepInterval=config.sim2real.ROSStepInterval
        self.fixed_time_interval=config.sim2real.fixed_time_interval
        self.use_fixed_time_interval = config.sim2real.use_fixed_time_interval
        if self.record:
            self.episodeRecoder=Recoder()
            self.load_act=config.sim2real.load_act
            if self.load_act:
                self.episodeRecoder.loadActions()
        # use dummy robot and human states or use detected states from sensors
        self.use_dummy_detect = config.sim2real.use_dummy_detect


        self.action_type=config.action_space.kinematics

        self.thisSeed = args.seed
        self.nenv = args.n_rollout_threads


    # With set_action
    def smooth_action(self, action):
        """ mimic the dynamics of Turtlebot2i for sim2real """
        # if action.r is delta theta
        w = action.r / self.time_step
        # if action.r is w
        # w = action.r
        beta = 0.1
        left = (2 * action.v - 0.23 * w) / (2 * 0.035)
        right = (2 * action.v + 0.23 * w) / (2 * 0.035)

        left = np.clip(left, -17.5, 17.5)
        right = np.clip(right, -17.5, 17.5)

        # print('Before: left:', left, 'right:', right)
        if self.phase == 'test':
            left = (1. - beta) * self.last_left + beta * left
            right = (1. - beta) * self.last_right + beta * right

        self.last_left = copy.deepcopy(left)
        self.last_right = copy.deepcopy(right)

        # subtract a noisy amount of delay from wheel speeds to simulate the delay in tb2
        # do this in the last step because this happens after we send action commands to tb2
        if left > 0:
            adjust_left = left - np.random.normal(loc=1.8, scale=0.15)
            left = max(0., adjust_left)
        else:
            adjust_left = left + np.random.normal(loc=1.8, scale=0.15)
            left = min(0., adjust_left)

        if right > 0:
            adjust_right = right - np.random.normal(loc=1.8, scale=0.15)
            right = max(0., adjust_right)
        else:
            adjust_right = right + np.random.normal(loc=1.8, scale=0.15)
            right = min(0., adjust_right)

        if self.record:
            self.episodeRecoder.wheelVelList.append([left, right])
        # print('After: left:', left, 'right:', right)

        v = 0.035 / 2 * (left + right)
        r = 0.035 / 0.23 * (right - left) * self.time_step
        return ActionRot(v, r)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)


    def step(self, action_n):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        self._set_action(action_n)
        reward, done, episode_info = self._get_reward_done_info(action_n)
        self.world.step(self.phase, self.global_time)

        if self.record:
            self.episodeRecoder.actionList.append(list(action))
            self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
            self.episodeRecoder.orientationList.append(self.robot.theta)

            if done:
                self.episodeRecoder.robot_goal.append([self.robot.gx, self.robot.gy])
                self.episodeRecoder.saveEpisode(self.case_counter['test'])

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter = self.step_counter+1

        bad_transition = True if str(episode_info[0])=='Timeout' else False
        info={'info': episode_info, 
              'bad_transition': bad_transition,
              'reward': []}

        ob = []
        for i in range(self.world.num_agents):
            ob.append(self._get_obs(i))


        for i in range(self.world.num_agents):
            if isinstance(episode_info[i], Collision):
                self.reset_callback(self.world, i)
            info['reward'].append(math.sqrt(pow(self.world.agents[i].px - self.world.agents[i].gx,2) + pow(self.world.agents[i].py - self.world.agents[i].gy,2)))

        return ob, reward, done, info

    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        :return:
        """
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        
        self.reset_callback(self.world)

        for agent in self.world.agents + self.humans + self.world.cams:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get current observation
        ob = []
        for i in range(self.world.num_agents):
            ob.append(self._get_obs(i))

        return ob

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    def _get_reward_done_info(self, action, phase='train', danger_zone='circle'):
        if self.global_time >= self.time_limit - 1:
            reward = [[0]]*self.world.num_agents
            _, done, _ = reward, done, info = self.reward_done_info_callback(action, self.world, phase, danger_zone)
            info = [Timeout()]*self.world.num_agents
        elif self.reward_done_info_callback is None:
            return 0.0, True, Nothing()
        else:
            reward, done, info = self.reward_done_info_callback(action, self.world, phase, danger_zone)
        
        return reward, done, info

    def talk2Env(self, data):
        """
        Call this function when you want extra information to send to/recv from the env
        :param data: data that is sent from gst_predictor network to the env, it has 2 parts:
        output predicted traj and output masks
        :return: True means received
        """
        self.gst_out_traj=data
        return True

    # set env action for a particular agent
    def _set_action(self, actions):
        # world.robot.set_velocity(action)
        for j, robot in enumerate(self.world.agents):
            if robot.policy.name in ['ORCA', 'social_force']:
                # assemble observation for orca: px, py, vx, vy, r
                # include all observable humans from t to t+t_pred
                _, _, human_visibility = self.world.get_visible_entities(j,'agent')
                # [self.predict_steps + 1, self.human_num, 4]
                human_states = copy.deepcopy(self.world.calc_human_future_traj(method='truth'))
                # append the radius, convert it to [human_num*(self.predict_steps+1), 5] by treating each predicted pos as a new human

                human_states = np.concatenate((human_states.reshape((-1, 4)),
                                            np.tile(self.world.cur_states[:, -1], self.world.predict_steps+1).reshape((-1, 1))),
                                            axis=1)

                # get orca action
                action = self.world.agents[j].action_callback(human_states.tolist())
            else:
                action = self.world.agents[j].policy.clip_action(actions[j], robot.v_pref)

            if self.world.agents[j].kinematics == 'unicycle':
                self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.world.agents[j].v_pref, self.world.agents[j].v_pref)
                action = ActionRot(self.desiredVelocity[0], action.r)

                # if action.r is delta theta
                if self.record:
                    self.episodeRecoder.unsmoothed_actions.append(list(action[j]))

                action = self.smooth_action(action)
            
            self.world.agents[j].action = action


    def render(self, ax, mode='visualize'):
        """ Render the current status of the environment using matplotlib """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        if mode=='human':
            return

        self.world.update_visibility()

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        artists=[]
        # compute orientation in each step and add arrow to show the direction

        arrowStartEnd=[]
        
        for j, robot in enumerate(self.world.agents):
            # add goal
            goal=mlines.Line2D([robot.gx], [robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal)
            artists.append(goal)

            # add robot
            robotX,robotY=robot.get_position()

            robot_circle=plt.Circle((robotX,robotY), robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot_circle)
            artists.append(robot_circle)

            radius = robot.radius
            robot_theta = robot.theta if robot.kinematics == 'unicycle' else np.arctan2(robot.vy, robot.vx)
            arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

            if self.world.robot_fov < 2 * np.pi:
                FOVAng = self.world.robot_fov / 2
                FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
                FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')

                robotX, robotY = robot.get_position()

                startPointX = robotX
                startPointY = robotY
                endPointX = robotX + radius * np.cos(robot_theta)
                endPointY = robotY + radius * np.sin(robot_theta)

                # transform the vector back to self.world frame origin, apply rotation matrix, and get end point of FOVLine
                # the start point of the FOVLine is the center of the robot
                FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / robot.radius)
                FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
                FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
                FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / robot.radius)
                FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
                FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

                ax.add_artist(FOVLine1)
                ax.add_artist(FOVLine2)
                artists.append(FOVLine1)
                artists.append(FOVLine2)

            # add an arc of robot's sensor range
            sensor_range = plt.Circle(robot.get_position(), robot.sensor_range + robot.radius+ self.world.config.human.radius, fill=False, linestyle='--')
            ax.add_artist(sensor_range)
            artists.append(sensor_range)

        for i, human in enumerate(self.world.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        for j, robot in enumerate(self.world.agents):
            
            potential_line = mlines.Line2D([0,0], [0,0], linestyle='--')
            
            robotX, robotY = robot.get_position()

            startPointX = robotX
            startPointY = robotY            

            endPointX = robot.gx
            endPointY = robot.gy

            potential_line.set_xdata(np.array([startPointX, endPointX]))
            potential_line.set_ydata(np.array([startPointY, endPointY]))

            ax.add_artist(potential_line)
            artists.append(potential_line)

            plt.text(robot.px - 0.1, robot.py - 0.1, j, color='black', fontsize=12)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.world.humans]
        obs_circles = [plt.Circle(obs.get_position(), obs.radius, fill=True, linewidth=1.5) for obs in self.world.obs]
        cams_circles = [plt.Circle(cam.get_position(), cam.radius, fill=True, linewidth=1.5, color='tab:red') for cam in self.world.cams]
        sensor_cams_circles = [plt.Circle(cam.get_position(), cam.radius + cam.sensor_range + self.world.config.cam.radius, fill=False, linestyle='--') for cam in self.world.cams]

        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        for i in range(len(self.world.obs)):
            ax.add_artist(obs_circles[i])
            artists.append(obs_circles[i])

        for i in range(len(self.world.cams)):
            ax.add_artist(cams_circles[i])
            artists.append(cams_circles[i])
            ax.add_artist(sensor_cams_circles[i])
            artists.append(sensor_cams_circles[i])

        

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5
        # plot current human states
        for i in range(len(self.world.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            plt.text(self.world.humans[i].px - 0.1, self.world.humans[i].py - 0.1, i, color='black', fontsize=12)

        if self.world.pred_method == 'inferred':
            # plot predicted human positions
            for i in range(len(self.world.humans)):
                # add future predicted positions of each human
                if self.gst_out_traj is not None:
                    for j in range(self.world.predict_steps):
                        circle = plt.Circle(self.gst_out_traj[i, (2 * j):(2 * j + 2)] + np.array([robotX, robotY]),
                                            self.world.config.human.radius, fill=False, color='tab:orange', linewidth=1.5)
                        ax.add_artist(circle)
                        artists.append(circle)

        if self.world.future_traj is not None:
            for r in range(self.world.num_agents):
                ids, num_human_in_view, mask = self.world.get_visible_entities(r, 'agent')
                for j in range(self.world.max_human_num):
                    if mask[j]:
                        for i in range(self.world.predict_steps):
                            circle = plt.Circle(self.world.future_traj[i, j,:2],
                                                self.world.config.human.radius, fill=False, color='tab:orange', linewidth=1.5)
                            ax.add_artist(circle)
                            artists.append(circle)  
                for j in range(self.world.num_agents):
                    if mask[j + self.world.entities_num]:
                        for i in range(self.world.predict_steps):
                            circle = plt.Circle(self.world.future_traj[i, self.world.entities_num + j,:2],
                                                self.world.config.human.radius, fill=False, color='tab:green', linewidth=1.5)
                            ax.add_artist(circle)
                            artists.append(circle)                                           

        # save plot for slide show
        if self.config.save_slides:
            import os
            folder_path = os.path.join(self.config.save_path, str(self.rand_seed)+'pred')
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok = True)
            plt.savefig(os.path.join(folder_path, str(self.step_counter)+'.png'), dpi=300)

        
        plt.pause(0.1)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        
        for t in ax.texts:
            t.set_visible(False)