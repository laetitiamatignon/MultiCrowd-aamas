import numpy as np
from numpy.linalg import norm

from onpolicy.envs.crowd_sim.utils.info import *
from onpolicy.envs.crowd_sim.crowd_nav.policy.orca import ORCA
from onpolicy.envs.crowd_sim.utils.state import *
from onpolicy.envs.crowd_sim.utils.action import ActionRot, ActionXY
from onpolicy.envs.crowd_sim.utils.recorder import Recoder

from onpolicy.envs.crowd_sim.utils.state import JointState


# multi-agent world
class World(object):
    def __init__(self):
        self.size_map = None
        self.time_step = None
        self.world_step = 0
        self.predict_steps = None
        self.pred_interval = None
        self.buffer_len = None
        self.entities_num = None

        self.humans = None
        self.dummy_human = None
        self.human_fov = None
        self.human_num = None
        self.observed_human_ids = None
        self.random_goal_changing = None
        self.goal_change_chance = None
        self.end_goal_changing = None
        self.end_goal_change_chance = None
        self.human_num_range = None
        self.max_human_num = None
        self.min_human_num = None
        
        self.obs = None
        self.tot_obs_num = None
        self.radius_obstacles = None
        
        # limit FOV
        self.num_agents = None
        self.agents = None
        self.robot_fov = None
        self.dummy_robot = None
        self.multi_potential = None

        self.cams = None
        self.cam_fov = None
        self.cam_sensor_range = None
        
        self.pred_method = None

        # NUMPY STYLE
        # np.array : number_human+number_agents x number_human+number_agents
        self.visibility = None
        # np.array : number_human+number_agents x 2
        self.speed = None
        # np.array : number_human+number_agents x 2
        self.position = None
        # np.array : number_human+number_agents x 1
        self.radius = None
        # np.array : number_agents x 1
        self.goal = None
        # np.array : number_human+number_agents x 5
        self.cur_states = None
        # np.array : new calc_future_traj number_human+(number_agents(optionnel)) x 4
        self.future_traj = None
        self.prev_human_pos = None
        # If robot consider other robots as humans
        self.robot_in_traj_human = None

        self.visible_mask = []
        self.collide_mask = []

    def step(self, phase, global_time):
        self.world_step += 1

        self.update_visibility()
        human_actions = self.get_human_actions()
        camera_actions = self.get_camera_actions()

        # apply action and update all entities
        for j, robot in enumerate(self.agents):
            self.agents[j].step(robot.action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        for i, camera_action in enumerate(camera_actions):
            self.cams[i].step(camera_action)

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.human_num_range > 0 and global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - self.min_human_num
                else:
                    max_remove_num = min(self.human_num - self.min_human_num, (self.human_num - 1) - max(self.observed_human_ids))
                remove_num = np.random.randint(low=0, high=max_remove_num + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                self.cur_states[self.human_num:self.max_human_num] = [15, 15, 0, 0, 0.3]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i >= self.max_human_num:
                            break
                        self.generate_humans(self, 1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num

        assert self.min_human_num <= self.human_num <= self.max_human_num

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if global_time % 5 == 0:
                for human in self.humans:
                    if human.v_pref == 0:
                        continue
                    self.update_human_goal(self, human)

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if  human.v_pref != 0 and norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    if np.random.random() <= self.end_goal_change_chance:
                        self.update_human_goal(self, human)

    # update last human states for invisible objects by guessing the trajectory
    def pred_invisible_traj(self, ob_state, reset):
        if reset:
            return np.array([15., 15., 0., 0., 0.3])
        else:
            return np.array([ob_state.px + ob_state.vx*self.time_step, ob_state.py + ob_state.vy*self.time_step, ob_state.vx, ob_state.vy, ob_state.r])

    def update_last_human_states(self, visible_mask=None, reset=False):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        h, mh, o, c, a = self.index_in_cur_states()
        
        for i in range(h):
            self.cur_states[i,:] = np.array(self.humans[i].get_observable_state_list())
            self.visible_mask[i] = self.humans[i].visible
            self.collide_mask[i] = self.humans[i].collide
        for i in range(h, mh):
            self.cur_states[i,:] = np.array(self.dummy_human.get_observable_state_list())
            self.visible_mask[i] = False
            self.collide_mask[i] = False
        for i in range(mh, o):
            self.cur_states[i,:] = np.array(self.obs[i - mh].get_observable_state_list())
            self.visible_mask[i] = self.obs[i - mh].visible
            self.collide_mask[i] = self.obs[i - mh].collide
        for i in range(o, c):
            self.cur_states[i,:] = np.array(self.cams[i - o].get_observable_state_list())
            self.visible_mask[i] = self.cams[i - o].visible
            self.collide_mask[i] = self.cams[i - o].collide
        for i in range(c, a):
            self.cur_states[i,:] = np.array(self.agents[i - c].get_observable_state_list())
            # because only human need to know robot pos
            self.visible_mask[i] = self.agents[i - c].visible
            self.collide_mask[i] = self.agents[i - c].collide

    def calc_human_future_traj(self, method='const_vel'):
        h, mh, o, c, a = self.index_in_cur_states()

        self.future_traj[:, :, :] = self.cur_states[:, :-1]
        if method=='const_vel':
            self.future_traj[:, :, 2:4] = self.prev_human_pos[:, 2:4]

        if method == 'truth':
            for i in range(1, self.buffer_len + 1):
                observable_states = []
                # observable states, for each human, remove human from obs and act
                ob_h = [np.concatenate((self.future_traj[i - 1, k], [self.humans[k].radius])) for k in range(len(self.humans)) if self.humans[k].visible]
                ob_o = [np.concatenate((self.future_traj[i - 1, mh + k], [self.obs[k].radius])) for k in range(len(self.obs)) if self.obs[k].visible]
                ob_c = [np.concatenate((self.future_traj[i - 1, o + k], [self.cams[k].radius])) for k in range(len(self.cams)) if self.cams[k].visible]
                ob_a = [np.concatenate((self.future_traj[i - 1, c + k], [self.agents[k].radius])) for k in range(len(self.agents)) if self.agents[k].visible]

                observable_states = ob_h + ob_o + ob_c + ob_a

                for j in range(len(self.humans)):
                    if self.humans[j].visible:
                        full_state = np.concatenate(
                            (self.future_traj[i - 1, j], self.humans[j].get_full_state_list()[4:]))

                        observable_states.pop(j)
                        # use joint states to get actions from the states in the last step (i-1)
                        action = self.humans[j].actWithJointState(JointState(full_state, observable_states))
                        observable_states.insert(j, np.concatenate((self.future_traj[i - 1, j], [self.humans[j].radius])))

                        # step all humans with action
                        self.future_traj[i, j] = self.humans[j].one_step_lookahead(
                            self.future_traj[i - 1, j, :2], action)

                for j in range(len(self.obs)):
                    if self.obs[j].visible:
                        self.future_traj[i, mh + j] = self.future_traj[i-1, mh + j]

                for j in range(len(self.cams)):
                    if self.cams[j].visible:
                        full_state = np.concatenate(
                            (self.future_traj[i - 1, o + j], self.cams[j].get_full_state_list()[4:]))

                        observable_states.pop(o + j)
                        # use joint states to get actions from the states in the last step (i-1)
                        action = self.cams[j].actWithJointState(JointState(full_state, observable_states))
                        observable_states.insert(j, np.concatenate((self.future_traj[i - 1,o + j], [self.cams[j].radius])))

                        # step all humans with action
                        self.future_traj[i, o + j] = self.cams[j].one_step_lookahead(
                            self.future_traj[i - 1, o + j, :2], action)

                for j in range(len(self.agents)):
                    if self.agents[j].visible and self.robot_in_traj_human:
                        action = ActionXY(*self.future_traj[i-1, c + j, 2:])
                        self.future_traj[i, c + j] = self.agents[j].one_step_lookahead(self.future_traj[i-1, c + j, :2], action)

            self.future_traj = self.future_traj[::self.pred_interval]
        elif method == 'const_vel':
            self.future_traj[0, :, 2:4] = self.prev_human_pos[:, 2:4]
            self.future_traj = np.tile(self.future_traj[0].reshape(1, self.entities_num + self.num_agents, 4), (self.predict_steps+1, 1, 1))
            pred_timestep = np.tile(np.arange(0, self.predict_steps+1, dtype=float).reshape((self.predict_steps+1, 1, 1)) * self.time_step * self.pred_interval, [1, self.entities_num + self.num_agents, 2])
            pred_disp = pred_timestep * self.future_traj[:, :, 2:]
            self.future_traj[:, :, :2] = self.future_traj[:, :, :2] + pred_disp
        else:
            raise NotImplementedError

        return self.future_traj

    def detect_visible_idx(idx_1, idx_2):
        return self.visibility[idx_1, idx_2]

    # update all visibilities at once
    def update_visibility(self, custom_fov=None, custom_sensor_range=None):
        visible_mask = np.tile(np.array(self.visible_mask), (len(self.visible_mask),1))

        position = self.cur_states[:,:2]
        fov = np.ones(self.entities_num+self.num_agents) if custom_fov else np.array([self.human_fov]*self.entities_num + [self.robot_fov]*self.num_agents)
        sensor = np.array([custom_sensor_range]*(self.entities_num+self.num_agents)) if custom_sensor_range else np.array([np.inf]*self.max_human_num + [0]*self.tot_obs_num + [self.cam_sensor_range]*self.num_cameras + [agent.sensor_range for agent in self.agents])
        sensor = np.tile(sensor, (sensor.shape[0],1))
        sensor[:-len(self.agents), :-len(self.agents)] = self.agents[0].sensor_range_robot

        if np.count_nonzero(self.cur_states[:, 2:4]):
            v_fov = self.cur_states[:,2:4] / np.linalg.norm(self.cur_states[:,2:4], axis=1)[:,None]
            v_12 = position - position[:,np.newaxis]
            v_12 = v_12 / np.linalg.norm(v_12, axis=2)[:,:,None]
            v_12[np.isnan(v_12)] = 0

            offset = np.arccos(np.clip((v_12*v_fov[:,np.newaxis]).sum(axis=2), a_min=-1, a_max=1))
            fov_limit = np.transpose(np.tile(fov/2, (len(self.visible_mask))).reshape(len(self.visible_mask), len(self.visible_mask)))
            visibility_angle = np.abs(offset) <= fov_limit
        else:
            visibility_angle = np.ones([self.entities_num + self.num_agents, self.entities_num + self.num_agents]).astype(int)

        # Sensor visibility
        dist = np.linalg.norm(position[:, np.newaxis] - position, axis=2)
        dist = dist - self.cur_states[:,-1] - self.cur_states[:,-1].reshape((-1,1))

        visibility_range = np.transpose(dist <= sensor)

        self.visibility = (visibility_angle & visibility_range & visible_mask)

        return self.visibility

    def get_visible_entities(self, id, type_ent):
        id = self.get_index_in_cur_matrix(id, type_ent)
        humans_in_view = np.where(self.visibility[id, :])
        num_humans_in_view = self.visibility[id, :].sum()
        human_ids = self.visibility[id]
        return humans_in_view, num_humans_in_view, human_ids

    def is_collided(self):
        h, mh, o, c, a = self.index_in_cur_states()

        dist = np.linalg.norm(self.cur_states[c:,:2][:, np.newaxis] - self.cur_states[:,:2], axis=2)
        for j in range(self.num_agents):
            dist[j, c + j] = np.inf
        dist = dist - self.cur_states[c:,-1].reshape((-1,1)) - self.cur_states[:,-1]
        dmin = dist[:, self.collide_mask].min(axis=1)
        dmin[dmin<0] = float('inf')
        
        return list((dist[:, self.collide_mask] < 0).any(axis=1)), list(dmin)

    def is_goal_reached(self):
        goal_reached = []
        dist_robot_goal = []
        for j, robot in enumerate(self.agents):
            if robot.kinematics == 'unicycle':
                goal_radius = 0.6
            else:
                goal_radius = robot.radius
            dist_robot_goal.append(norm(np.array(robot.get_position()) - np.array(robot.get_goal_position())))
            goal_reached.append(dist_robot_goal[-1] < goal_radius)
        return goal_reached, dist_robot_goal

    def is_robot_in_danger(self, min_human_dist, danger_zone, phase, discomfort_dist):
        danger_cond = []
        for j, robot in enumerate(self.agents):
            if danger_zone == 'circle' or phase == 'train':
                danger_cond.append(min_human_dist[j] < discomfort_dist)
            else:
                # if the robot collides with future states, give it a collision penalty
                relative_pos = self.human_future_traj[1:, :, :2] - np.array([robot.px, robot.py])
                relative_dist = np.linalg.norm(relative_pos, axis=-1)
                collision_idx = relative_dist < robot.radius + self.r
                danger_cond.append(np.any(collision_idx))
        return danger_cond

    def reward_robot_in_traj(self, collision_penalty):
        h, mh, o, c, a = self.index_in_cur_states()
        position = self.cur_states[:,:2]
        robot_position = self.cur_states[c:,:2]
        radius = self.cur_states[:,-1]
        env_radius = radius[:c]
        robot_radius = radius[c:]

        # relative_pos : pred x num_obs x num_agents x pos
        size_future_traj = self.future_traj.shape

        pos_robot = self.cur_states[-self.num_agents:, :2]
        futu_traj = np.tile(self.future_traj[1:, :, :2], self.num_agents).reshape((size_future_traj[0]-1, size_future_traj[1], self.num_agents, 2))

        relative_pos = futu_traj - pos_robot
        radius_h_r = np.tile(radius[:, np.newaxis] + robot_radius, size_future_traj[0]-1).reshape((size_future_traj[0]-1, size_future_traj[1], self.num_agents))

        collision_idx = np.linalg.norm(relative_pos, axis=-1) < radius_h_r
        collision_idx[:, -self.num_agents:, :] = np.multiply(collision_idx[:, -self.num_agents:, :], np.repeat(-1*(np.eye(self.num_agents) - 1)[None, :], size_future_traj[0]-1, axis=0).reshape(size_future_traj[0]-1, self.num_agents, self.num_agents))

        coefficients = 2. ** np.tile(np.arange(2, size_future_traj[0]-1 + 2).reshape((size_future_traj[0]-1, 1)), self.num_agents)  # 4, 8, 16, 32
        collision_penalties = collision_penalty / coefficients

        reward_future = collision_idx * collision_penalties.reshape((size_future_traj[0]-1, 1, self.num_agents)) # [predict_steps, human_num]
        reward_future = np.min(reward_future.reshape([-1,self.num_agents]), axis=0)

        return np.any(collision_idx.reshape(((size_future_traj[0]-1)* size_future_traj[1], self.num_agents)),axis=0), reward_future

    def get_human_actions(self):
        # step all humans
        human_actions = []

        if(self.human_fov/np.pi==2.):
            # observable states, for each human, remove human from obs and act
            ob_h = [human.get_observable_state() for i, human in enumerate(self.humans) if human.visible]
            ob_a = [agent.get_observable_state() for i, agent in enumerate(self.agents) if agent.visible and self.robot_in_traj_human]
            ob_o = [ob.get_observable_state() for i, ob in enumerate(self.obs) if ob.visible]
            ob_c = [cam.get_observable_state() for i, cam in enumerate(self.cams) if cam.visible]

            ob = ob_h + ob_o + ob_c + ob_a
            
            for i, human in enumerate(self.humans):
                ob.pop(i)
                human_actions.append(human.action_callback(ob))
                ob.insert(i, human.get_observable_state())
        else:
            # observable states, for each human, remove human from obs and act
            ob_h = [human.get_observable_state() for i, human in enumerate(self.humans)]
            ob_a = [agent.get_observable_state() for i, agent in enumerate(self.agents) if self.robot_in_traj_human]
            ob_o = [ob.get_observable_state() for i, ob in enumerate(self.obs)]
            ob_c = [cam.get_observable_state() for i, cam in enumerate(self.cams)]

            ob = ob_h + ob_o + ob_c + ob_a

            for i, human in enumerate(self.humans):
                ob.pop(i)
                ob_save = ob
                _, _, visible_mask = self.get_visible_entities(i, 'human')
                human_actions.append(human.action_callback([ob[i] for i in range(len(ob)) if visible_mask[i]]))
                ob.insert(i, human.get_observable_state())

        return human_actions

    def get_camera_actions(self):
        # step all humans
        camera_actions = []  # a list of all humans' actions
        h, mh, o, c, a = self.index_in_cur_states()

        if(self.cam_fov/np.pi==2.):
            # observable states, for each human, remove human from obs and act
            ob_h = [human.get_observable_state() for i, human in enumerate(self.humans) if human.visible]
            ob_a = [agent.get_observable_state() for i, agent in enumerate(self.agents) if agent.visible and self.robot_in_traj_human]
            ob_o = [ob.get_observable_state() for i, ob in enumerate(self.obs) if ob.visible]
            ob_c = [cam.get_observable_state() for i, cam in enumerate(self.cams) if cam.visible]

            ob = ob_h + ob_o + ob_c + ob_a
            
            for i, cam in enumerate(self.cams):
                ob.pop(o + i)
                camera_actions.append(cam.action_callback(ob))
                ob.insert(o + i, cam.get_observable_state())
        else:
            # observable states, for each human, remove human from obs and act
            ob_h = [human.get_observable_state() for i, human in enumerate(self.humans)]
            ob_a = [agent.get_observable_state() for i, agent in enumerate(self.agents) if self.robot_in_traj_human]
            ob_o = [ob.get_observable_state() for i, ob in enumerate(self.obs)]
            ob_c = [cam.get_observable_state() for i, cam in enumerate(self.cams)]

            ob = ob_h + ob_o + ob_c + ob_a

            for i, cam in enumerate(self.cams):
                ob.pop(o + i)
                ob_save = ob
                _, _, visible_mask = self.get_visible_entities(i, 'cam')
                camera_actions.append(cam.action_callback([ob[i] for i in range(len(ob)) if visible_mask[i]]))
                ob.insert(o + i, cam.get_observable_state())

        return camera_actions

    def index_in_cur_states(self):
        human_idx = self.human_num
        max_human_idx = self.max_human_num
        obs_idx = len(self.obs) + max_human_idx
        cam_idx = len(self.cams) + obs_idx
        ag_idx = len(self.agents) + cam_idx

        return human_idx, max_human_idx, obs_idx, cam_idx, ag_idx

    def get_index_in_cur_matrix(self, num, type_ent='agent'):
        h, m, o, c, a = self.index_in_cur_states()
        if type_ent=='agent':
            return c + num
        if type_ent=='human':
            return num
        if type_ent=='cam':
            return o + num
        if type_ent=='obs':
            return m + num
        return -1
                
