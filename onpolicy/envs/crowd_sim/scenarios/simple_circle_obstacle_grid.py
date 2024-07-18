import numpy as np
from numpy.linalg import norm

from onpolicy.envs.crowd_sim.core import World
from onpolicy.envs.crowd_sim.scenario import BaseScenario


from onpolicy.envs.crowd_sim.utils.movable import Robot, Human, Camera
from onpolicy.envs.crowd_sim.utils.obstacle import Obstacle

from onpolicy.envs.crowd_sim.utils.info import *
from onpolicy.envs.crowd_sim.crowd_nav.policy.orca import ORCA
from onpolicy.envs.crowd_sim.utils.state import *
from onpolicy.envs.crowd_sim.utils.action import ActionRot, ActionXY
from onpolicy.envs.crowd_sim.utils.recorder import Recoder

from onpolicy.envs.crowd_sim.utils.state import JointState

import numpy as np
import rvo2
import random
import copy
import math

class Scenario(BaseScenario):
    def make_world(self, args, config):
        self.config = config
        self.args = args
        world = World()

        # args for reward function (replace it by config maybe)
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        world.randomize_attributes = config.env.randomize_attributes
        world.discomfort_dist = config.reward.discomfort_dist
        self.grid_cell = args.grid_cell

        # Communication
        self.communication = config.sim.can_communicate
        self.comm_cam_dist = config.sim.comm_cam_dist

        # Config world
        world.config = config
        world.size_map = config.sim.circle_radius
        self.size_map = world.size_map

        # configure randomized goal changing of humans midway through episode
        world.random_goal_changing = config.human.random_goal_changing
        if world.random_goal_changing:
            world.goal_change_chance = config.human.goal_change_chance
        # configure randomized goal changing of humans after reaching their respective goals
        world.end_goal_changing = config.human.end_goal_changing
        if world.end_goal_changing:
            world.end_goal_change_chance = config.human.end_goal_change_chance
        world.update_human_goal = self.update_human_goal
        world.generate_humans = self.generate_humans

        # Humans
        world.humans = []
        world.human_num = config.sim.human_num
        world.human_fov = np.pi * config.human.FOV
        world.predict_steps = args.predict_steps
        world.human_num_range = config.sim.human_num_range

        assert world.human_num > world.human_num_range
        if self.config.action_space.kinematics == 'holonomic':
            world.max_human_num = world.human_num + world.human_num_range
            world.min_human_num = world.human_num - world.human_num_range
        else:
            world.min_human_num = 1
            world.max_human_num = 5

        # dummy humans, used if any human is not in view of other agents
        world.dummy_human = Human(self.config)
        # if a human is not in view, set its state to (px = 100, py = 100, vx = 0, vy = 0, theta = 0, radius = 0)
        world.dummy_human.set(7, 7, 7, 7, 0, 0, 0) # (7, 7, 7, 7, 0, 0, 0)
        world.dummy_human.time_step = config.env.time_step

        # prediction period / control (or simulation) period
        world.pred_interval = int(config.data.pred_timestep // config.env.time_step)
        world.buffer_len = world.predict_steps * world.pred_interval
        world.pred_method = config.sim.predict_method
        world.time_step = config.env.time_step

        # Robot
        world.num_agents = args.num_agents
        world.agents = []
        # Distance from goal for each agents
        world.multi_potential = [0]*world.num_agents
        world.robot_fov = np.pi * config.robot.FOV
        # dummy robot, used if any robot is not in view of other agents
        world.dummy_robot = Robot(self.config)
        world.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        world.dummy_robot.time_step = config.env.time_step
        world.dummy_robot.kinematics = 'holonomic'
        world.dummy_robot.policy = ORCA(config)

        # Obstacles
        world.obs = []
        world.obstacles = config.sim.obstacles
        world.obstacle_radius = config.obstacle.max_radius

        # Cameras
        world.cams = []
        world.num_cameras = config.sim.num_cameras
        world.cam_fov = config.cam.FOV
        world.cam_sensor_range = config.cam.sensor_range
        world.camera_fov = np.pi * config.cam.FOV

        # robot seen by humans (TODO: destroy)
        world.robot_in_traj_human = config.robot.robot_in_traj_human
        # robot seen by robot
        world.robot_in_traj_robot = config.robot.robot_in_traj_robot

        self.reset_world(world)
        return world

    def generate_obstacles(self, world):
        world.obstacles = [] if world.obstacles == [] else world.size_map * np.array(self.config.sim.obstacles)

        if world.obstacles != []:
            dist = world.obstacles[:,:2] - world.obstacles[:,2:]
            dist = np.linalg.norm(dist, axis=1)
            world.radius_obstacle = self.config.obstacle.max_radius
            world.num_obs = (dist/world.radius_obstacle).astype(int)
            tot_obs_num = world.num_obs.sum()
        else:
            tot_obs_num = 0

        world.tot_obs_num = tot_obs_num

        world.obs = []
        for i, row in enumerate(world.obstacles):
            obs_x = np.linspace(row[0], row[2], int(world.num_obs[i]))
            obs_y = np.linspace(row[1], row[3], int(world.num_obs[i]))
            obs_x += world.radius_obstacle/2
            obs_y += world.radius_obstacle/2
            for k in range(int(world.num_obs[i])):
                world.obs.append(Obstacle(self.config))
                world.obs[-1].set(obs_x[k], obs_y[k], 0, 0, world.radius_obstacle)

    def generate_cameras(self, world):
        """
        Calls generate_circle_crossing_human function to generate a certain number of random humans
        :param human_num: the total number of humans to be generated
        :return: None
        """
        # initial min separation distance to avoid danger penalty at beginning
        for i in range(world.num_cameras):
            world.cams.append(self.generate_camera(world))

    def generate_camera(self, world):
        cam = Camera(self.config)

        if world.randomize_attributes:
            cam.sample_random_attributes()

        angle = np.random.random() * np.pi * 2
        v_pref = 1.0 if cam.v_pref == 0 else cam.v_pref
        px_noise = np.random.uniform(0, 1) * v_pref
        py_noise = np.random.uniform(0, 1) * v_pref
        px = world.size_map * np.cos(angle) + px_noise
        py = world.size_map * np.sin(angle) + py_noise
        cam.set(px, py, -px, -py, 0, 0, 0)

        return cam

    def generate_humans(self, world, human_num):
        """
        Calls generate_circle_crossing_human function to generate a certain number of random humans
        :param human_num: the total number of humans to be generated
        :return: None
        """
        for i in range(human_num):
            world.humans.append(self.generate_human(world))

    def generate_human(self, world):
        human = Human(self.config)

        while True:
            if world.randomize_attributes:
                human.sample_random_attributes()

            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 2
            px_noise = np.random.uniform(0, 1) * noise_range
            py_noise = np.random.uniform(0, 1) * noise_range

            # v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            # px_noise = (np.random.uniform() - 0.5) * v_pref
            # py_noise = (np.random.uniform() - 0.5) * v_pref
            
            px = world.size_map * np.cos(angle) + px_noise
            py = world.size_map * np.sin(angle) + py_noise
            gx = -px
            gy = -py
            collide = False

            for i, other_human in enumerate(world.humans):
                if human.kinematics == 'unicycle':
                    min_dist = world.size_map / 2
                else:
                    min_dist = human.radius + other_human.radius + self.discomfort_dist
                if norm((px - other_human.px, py - other_human.py)) < min_dist or \
                    norm((gx - other_human.gx, gy - other_human.gy)) < min_dist:
                    collide = True
                    break

            for i, obs in enumerate(world.obs):
                min_dist = human.radius + obs.radius + self.discomfort_dist
                if norm((px - obs.px, py - obs.py)) < min_dist or \
                    norm((gx - obs.px, gy - obs.py)) < min_dist:
                    collide = True
                    break

            for i, agent in enumerate(world.agents):
                if human.kinematics == 'unicycle':
                    min_dist = world.size_map / 2
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                    collide = True
                    break

            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)

        return human

    def update_human_goal(self, world, human):
        # Update human's goals randomly
        angle = np.random.random() * np.pi * 2
        # add some noise to simulate all the possible cases robot could meet with human
        v_pref = 1.0 if human.v_pref == 0 else human.v_pref
        gx_noise = (np.random.random() - 0.5) * v_pref
        gy_noise = (np.random.random() - 0.5) * v_pref
        gx = world.size_map * np.cos(angle) + gx_noise
        gy = world.size_map * np.sin(angle) + gy_noise

        # Give human new goal
        human.gx = gx
        human.gy = gy

    def generate_agents(self, world):
        """
        Calls generate_circle_crossing_human function to generate a certain number of random humans
        :param human_num: the total number of humans to be generated
        :return: None
        """
        # initial min separation distance to avoid danger penalty at beginning
        for i in range(world.num_agents):
            world.agents.append(self.generate_agent(world))

            rob_goal_vec = np.array([world.agents[i].gx, world.agents[i].gy]) - np.array([world.agents[i].px, world.agents[i].py])
            angle = np.arctan2(rob_goal_vec[1], rob_goal_vec[0]) - world.agents[i].theta
            if angle > np.pi:
                angle = angle - 2 * np.pi
            elif angle < -np.pi:
                angle = angle + 2 * np.pi
            # initialize potential
            world.multi_potential[i] = -abs(np.linalg.norm(rob_goal_vec))

    def generate_agent(self, world):
        agent = Robot(self.config)
        if agent.kinematics == 'unicycle':
            while True:
                angle = np.random.uniform(0, np.pi * 2)
                px = world.size_map * np.cos(angle)
                py = world.size_map * np.sin(angle)
                gx, gy = np.random.uniform(-world.size_map, world.size_map, 2)

                for i, other_agent in enumerate(world.agents):
                    if agent.kinematics == 'unicycle':
                        min_dist = world.size_map / 2
                    else:
                        min_dist = agent.radius + other_agent.radius + self.discomfort_dist
                    if norm((px - other_agent.px, py - other_agent.py)) < min_dist or \
                       norm((gx - other_agent.gx, gy - other_agent.gy)) < min_dist:
                        collide = True
                        break

                for i, obs in enumerate(world.obs):
                    min_dist = agent.radius + obs.radius + self.discomfort_dist
                    if norm((px - obs.px, py - obs.py)) < min_dist or \
                       norm((gx - obs.px, gy - obs.py)) < min_dist:
                        collide = True
                        break

                for i, human in enumerate(world.humans):
                    if agent.kinematics == 'unicycle':
                        min_dist = world.size_map / 2
                    else:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                       norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break

                if not collide:
                    break

            agent.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2*np.pi))

        else:
            while True:
                px, py, gx, gy = np.random.uniform(-world.size_map, world.size_map, 4)
                collide = False
                
                if np.linalg.norm([px - gx, py - gy]) < 6:
                    continue

                for i, other_agent in enumerate(world.agents):
                    if agent.kinematics == 'unicycle':
                        min_dist = world.size_map / 2
                    else:
                        min_dist = agent.radius + other_agent.radius + self.discomfort_dist
                    if norm((px - other_agent.px, py - other_agent.py)) < min_dist or \
                       norm((gx - other_agent.gx, gy - other_agent.gy)) < min_dist:
                        collide = True
                        break

                for i, obs in enumerate(world.obs):
                    min_dist = agent.radius + obs.radius + self.discomfort_dist
                    if norm((px - obs.px, py - obs.py)) < min_dist or \
                       norm((gx - obs.px, gy - obs.py)) < min_dist:
                        collide = True
                        break

                for i, human in enumerate(world.humans):
                    if agent.kinematics == 'unicycle':
                        min_dist = world.size_map / 2
                    else:
                        min_dist = agent.radius + human.radius + self.discomfort_dist
                    if norm((px - human.px, py - human.py)) < min_dist or \
                       norm((gx - human.gx, gy - human.gy)) < min_dist:
                        collide = True
                        break

                if not collide:
                    break

            agent.set(px, py, gx, gy, 0, 0, np.pi/2)

        return agent

    def generate_entities(self, world, human_num=None):
        if human_num is None:
            human_num = world.human_num

        # total number of obstacles (point obstacles)
        world.tot_obs_num = 0
        # total number of entities (agents not included)
        world.entities_num = 0

        self.generate_obstacles(world)
        self.generate_cameras(world)
        self.generate_agents(world)
        self.generate_humans(world, world.human_num)

        world.entities_num = world.max_human_num + world.num_cameras + world.tot_obs_num

        # set human ids
        for i in range(world.human_num):
            world.humans[i].id = i

    def reset_world(self, world, reset_agent=-1):
        """
        Reset the environment
        :return:
        """
        if reset_agent==-1:
            # list of agents
            world.agents = []
            # list of humans
            world.humans = []
            # list of cams
            world.cams = []
            # list of obstacles
            world.obs = []
            # initialize a list to store observed humans' IDs
            world.observed_human_ids = []

            self.generate_entities(world)

            size_predict = world.buffer_len if world.pred_method == 'truth' else world.predict_steps
            # future trajectories of all entities : num_step_predict x num_entities x 4
            world.future_traj = np.zeros((size_predict + 1, world.entities_num + world.num_agents, 4))
            world.cur_states = np.zeros((world.entities_num + world.num_agents, 5))
            
            world.visible_mask = [False]*(world.entities_num + world.num_agents)
            world.collide_mask = [False]*(world.entities_num + world.num_agents)
            
            world.update_last_human_states()
            world.update_visibility()
        else:
            self.reset_agent(world, reset_agent)

    def reset_agent(self, world, j):
        new_robot = self.generate_agent(world)
        world.agents[j] = new_robot

    def reward(self, action_n, world, phase='train', danger_zone='circle'):
        # Collisions detected, distance between agents and other entities
        collision, min_human_dist = world.is_collided()
        # Agents who reached the goals, distance between agents and goals 
        goal_reached, goal_dist = world.is_goal_reached()
        # Danger warning for agents to close to humans
        danger_cond = world.is_robot_in_danger(min_human_dist, danger_zone, phase, self.discomfort_dist)
        # Danger warning for agents to close to future humans trajectories
        danger_traj, reward_robot_in_traj = world.reward_robot_in_traj(self.collision_penalty)

        reward_n = []
        done_n = [False]*world.num_agents
        episode_info_n = [Nothing() for i in range(world.num_agents)]

        for j, robot in enumerate(world.agents):
            reward = 0
            if collision[j]:
                reward += self.collision_penalty
                done_n[j] = True
                episode_info_n[j] = Collision()
            elif goal_reached[j]:
                reward += self.success_reward
                done_n[j] = True
                episode_info_n[j] = ReachGoal()

            elif danger_cond[j] or danger_traj[j]:
                # print('DANGER=============', danger_cond[j], danger_traj[j])
                reward += (min_human_dist[j] - self.discomfort_dist) * self.discomfort_penalty_factor * world.time_step
                done_n[j] = False
                episode_info_n[j] = Danger(min_human_dist[j])
                # print(reward)
            else:
                if robot.kinematics == 'holonomic':
                    pot_factor = 2
                else:
                    pot_factor = 3
                potential_cur = np.linalg.norm(
                    np.array([robot.px, robot.py]) - np.array(robot.get_goal_position()))

                reward = pot_factor * (-abs(potential_cur) - world.multi_potential[j])
                done_n[j] = False
                episode_info_n[j] = Nothing()

                world.multi_potential[j] = -abs(potential_cur)

            
            reward += reward_robot_in_traj[j]
            # print(reward)

            # if the robot is near collision/arrival, it should be able to turn a large angle
            if robot.kinematics == 'unicycle':
                # add a rotational penalty
                r_spin = -5 * action_n[j].r**2
                # add a penalty for going backwards
                if action_n[j].v < 0:
                    r_back = -2 * abs(action_n[j].v)
                else:
                    r_back = 0.
                reward = reward + r_spin + r_back
            
            reward_n.append([reward])
        return reward_n, done_n, episode_info_n

    def convert_to_grid_cell(self, a):
        """4 levels of grid cells, the biggest takes the size of the map"""
        a0 = np.array(a)/float(self.size_map)*np.pi
        a1 = np.array(a)/float(self.size_map)*np.pi*math.sqrt(2)
        a2 = np.array(a)/float(self.size_map)*np.pi*2
        a3 = np.array(a)/float(self.size_map)*np.pi*2*math.sqrt(2)

        return np.array([(math.cos(a0[0])+math.cos(a0[0]/2.+math.sqrt(3)/2.*a0[1])+math.cos(a0[0]/2.-math.sqrt(3)/2.*a0[1]))/3.,
                         (math.cos(a1[0])+math.cos(a1[0]/2.+math.sqrt(3)/2.*a1[1])+math.cos(a1[0]/2.-math.sqrt(3)/2.*a1[1]))/3.,
                         (math.cos(a2[0])+math.cos(a2[0]/2.+math.sqrt(3)/2.*a2[1])+math.cos(a2[0]/2.-math.sqrt(3)/2.*a2[1]))/3.,
                         (math.cos(a3[0])+math.cos(a3[0]/2.+math.sqrt(3)/2.*a3[1])+math.cos(a3[0]/2.-math.sqrt(3)/2.*a3[1]))/3.,]
                        )

    def observation(self, id, world):
        """Generate observation for reset and step functions"""
        h, mh, o, c, a = world.index_in_cur_states()

        world.prev_human_pos = copy.deepcopy(world.cur_states)
        world.update_last_human_states()
        
        robot = world.agents[id]

        # Visiblity
        world.update_visibility()
        _, _, visible_entities = world.get_visible_entities(id, 'agent')
        if self.communication:
            visible_entities = self.human_seen_by_robot(world, id)
        if not world.robot_in_traj_robot:
            visible_entities[-world.num_agents:] = False
        if self.args.archi_name == "DSRNNpp":
            visible_entities[c+id] = False
        else:
            visible_entities[c+id] = True
        num_visibles = visible_entities.sum()
        new_mask = np.array(visible_entities)==1

        # Trajectories and conversion
        predicted_states = world.calc_human_future_traj(method=world.pred_method)
        if self.grid_cell:
            dim_pos = int(4*(world.predict_steps+1))
        else:
            dim_pos = int(2*(world.predict_steps+1))
        spatial_edges = np.ones((a, dim_pos)) * np.inf
        pred_pos = np.transpose(predicted_states[:, :, :2], (1, 0, 2))
        if self.grid_cell:
            pred_pos = np.apply_along_axis(self.convert_to_grid_cell, -1, pred_pos)
        else:
            pred_pos = pred_pos - np.array([robot.px, robot.py])
        spatial_edges[new_mask] = (pred_pos).reshape((a, -1))[new_mask]

        # sort all humans by distance (invisible humans will be in the end automatically)
        if self.args.archi_name == "DSRNNpp":
            spatial_edges[-1] = np.inf
            sorted_spatial_edges = np.array(sorted(spatial_edges, key=lambda x: np.linalg.norm(x[:2])))
            sorted_spatial_edges[np.isinf(sorted_spatial_edges)] = 15

            nv = num_visibles if num_visibles > 0 else 1
            first_line_obs = robot.get_full_state_list_noV() + [robot.vx, robot.vy, nv, a-c-id]
            visi = world.visibility.copy()

            # dim_pos + 3 = dim_pos + visi (1) + category of nodes (2)
            ob = np.empty((2*a+1, max(a, dim_pos+3)))
            ob[:a, 0] = visible_entities
            ob[:a, 1:(dim_pos+1)] = sorted_spatial_edges
            one_hot = np.ones((a, 2))
            one_hot[:, 1] = 0
            one_hot[-world.num_agents:] = [0,1]
            ob[:a, (dim_pos+1):(dim_pos+3)] = one_hot
            ob[a:(2*a), :a] = visi
            ob[2*a, :len(first_line_obs)] = first_line_obs

            return ob

        # States of the agent
        nv = num_visibles if num_visibles > 0 else 1
        first_line_obs = robot.get_full_state_list_noV() + [robot.vx, robot.vy, a, a-c-id]
        if self.grid_cell:
            first_line_obs = list(self.convert_to_grid_cell(first_line_obs[:2])) + [first_line_obs[2]] + list(self.convert_to_grid_cell(first_line_obs[3:5])) + first_line_obs[5:]

        visi = world.visibility.copy()

        for j in range(visi.shape[0]):
            if not(visible_entities[j]) and j!= c + id:
                visi[j,:] = 0
                visi[:,j] = 0
            if not world.robot_in_traj_human and j>=c:
                visi[:mh,j] = 0
            if not world.robot_in_traj_robot and j>=c and j!= c+id:
                visi[c:,j] = 0
            visi[j,j] = 0

        spatial_edges[np.isinf(spatial_edges)] = 15

        # Positions of all entities
        ob = np.empty((2*a+1, max(a, dim_pos+3)))
        ob[:a, 0] = visible_entities
        ob[:a, 1:(dim_pos+1)] = spatial_edges
        # Encoding of the types
        one_hot = np.ones((a, 2))
        one_hot[:, 1] = 0
        one_hot[-world.num_agents:] = [0,1]
        ob[:a, (dim_pos+1):(dim_pos+3)] = one_hot
        # Visibilities
        ob[a:(2*a), :a] = visi
        ob[2*a, :len(first_line_obs)] = first_line_obs

        return ob

    def can_communicate(self, world, ind_1, ind_2):
        if np.linalg.norm(world.cur_states[ind_1, :2] - world.cur_states[ind_2, :2]) < self.comm_cam_dist:
            return True
        else:
            return False

    def human_seen_by_robot(self, world, ind_robot):
        h, mh, o, c, a = world.index_in_cur_states()
        visible_humans_id, num_visibles, visible_mask = world.get_visible_entities(ind_robot, 'agent')

        visible_mask_tot = visible_mask

        for ind_cam, cam in enumerate(world.cams):
            if self.can_communicate(world, ind_robot, o + ind_cam):
                visible_humans_id, num_visibles, visible_mask = world.get_visible_entities(ind_cam, 'cam')
                visible_mask_tot += visible_mask
                visible_mask_tot[ind_cam] = True if cam.visible else False
        for ind_robot2, robot in enumerate(world.agents):
            if self.can_communicate(world, ind_robot, c + ind_robot2):
                visible_humans_id, num_visibles, visible_mask = world.get_visible_entities(ind_robot2, 'agent')
                visible_mask_tot += visible_mask
                visible_mask_tot[ind_cam] = True if robot.visible else False
        
        return visible_mask_tot>0
