import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.algorithms.utils.temperature_scheduler import Temp_Scheduler 
from onpolicy.envs.crowd_sim.utils.info import *
import matplotlib.pyplot as plt 
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class CrowdRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(CrowdRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        temperature_scheduler = Temp_Scheduler(episodes, 5, 0.05, temp_min=0.03)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            tau = temperature_scheduler.step()

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step, tau)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute(tau)
            train_infos = self.train(tau)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("{}/{} episodes, total num timesteps {}/{}, FPS {}, mean/min/max rew {}/{}/{}."
                        .format(episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                                np.mean(self.buffer.rewards),
                                np.min(self.buffer.rewards),
                                np.max(self.buffer.rewards)))

                if self.env_name == "CROWD":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = rewards[0][agent_id]

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step, tau):
        # Sample actions
        with torch.no_grad():
            self.trainer.prep_rollout()
            value, action, action_log_prob, rnn_states, rnn_states_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                np.concatenate(self.buffer.obs[step]),
                                np.concatenate(self.buffer.rnn_states[step]),
                                np.concatenate(self.buffer.rnn_states_critic[step]),
                                np.concatenate(self.buffer.masks[step]),
                                tau)
        
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))

        rnn_states = np.concatenate(np.split(_t2n(rnn_states), self.n_rollout_threads, axis=1))
        rnn_states_critic = np.array(np.split(rnn_states_critic, self.n_rollout_threads, axis=0))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.hidden_size), dtype=np.float32)

        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # if infos[0]['bad_transition']:
        #     bad_masks = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # else:
        #     bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                0.05,
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        tau = 0.003

        winning_goal_pos = []
        time_step_win_end = []

        eval_episode_rewards = []

        success_times = []
        collision_times = []
        timeout_times = []

        success = 0
        collision = 0
        timeout = 0
        too_close_ratios = []
        dist_intrusion = []

        collision_cases = []
        timeout_cases = []

        all_path_len = []

        end_dist_goal_tot = []
        min_dist_goal_tot = []

        for episode in range(self.all_args.render_episodes):
            done = False
            stepCounter = 0
            episode_rew = 0
            global_time = 0.0
            path_len = 0
            too_close = 0.
            max_dist_goal = 0
            end_dist_goal = 0
            min_dist_goal = [-1] * self.num_agents
            ended_agent = [0] * self.num_agents
            obs = envs.reset()
            last_pos = [obs[0, a, -1, :2] for a in range(self.num_agents)]

            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            elif self.all_args.visualize_traj:
                envs.render('visualize', self.ax)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()
                stepCounter = stepCounter + 1
                
                self.trainer.prep_rollout()

                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    tau,
                                                    deterministic=True)
                
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions)
                episode_rewards.append(rewards)

                if not dones.all():
                    for a in range(self.num_agents):
                        if ended_agent[a]:
                            continue

                        path_len = path_len + np.linalg.norm(obs[0, a, -1, :2] - last_pos[a])

                        if infos[0]['reward'][a] < min_dist_goal[a] or min_dist_goal[a] < 0:
                            min_dist_goal[a] = infos[0]['reward'][a]
                        if isinstance(infos[0]['info'][a], ReachGoal):
                            success += 1
                            success_times.append(global_time)
                            ended_agent[a] = 1
                            winning_goal_pos.append(obs[0, a, -1, 3:5])
                            time_step_win_end.append(int((float(step)/150.)*100.))
                        if isinstance(infos[0]['info'][a], Danger):
                            too_close = too_close + 1
                            dist_intrusion.append(infos[0]['info'][a].min_dist)
                        elif isinstance(infos[0]['info'][a], Collision):
                            collision += 1
                            collision_cases.append(episode)
                            collision_times.append(global_time)
                            ended_agent[a] = 1
                        episode_rew += rewards[0][a]
                else:
                    break

                last_pos = [obs[0, a, -1, :2] for a in range(self.num_agents)]
                rnn_states = rnn_states[0]

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                elif self.all_args.visualize_traj:
                    envs.render('visualize', self.ax)

                if not sum(ended_agent) == self.num_agents:
                    global_time = envs.envs[0].global_time
                else:
                    break

            eval_episode_rewards.append(np.mean(episode_rew))

            all_path_len.append(path_len/float(self.num_agents))
            too_close_ratios.append(too_close/(stepCounter*self.num_agents)*100)

            for i in range(len(infos[0]['info'])):
                if ended_agent[i]:
                    continue

                if isinstance(infos[0]['info'][i], ReachGoal):
                    success += 1
                    success_times.append(global_time)
                    winning_goal_pos.append(obs[0, a, -1, 3:5])
                    time_step_win_end.append(int((float(step)/150.)*100.))
                elif isinstance(infos[0]['info'][i], Collision):
                    collision += 1
                    collision_cases.append(episode)
                    collision_times.append(global_time)
                elif isinstance(infos[0]['info'][i], Timeout):
                    timeout += 1
                    timeout_cases.append(episode)
                    timeout_times.append(global_time)
                elif isinstance(infos[0]['info'][i], Nothing):
                    pass
                else:
                    raise ValueError('Invalid end signal from environment')

            end_dist_goal_tot.append(np.mean([infos[0]['reward'][i] for i in range(len(infos[0]['info']))]))
            min_dist_goal_tot.append(np.mean(min_dist_goal))

        num_traj_tot = success + collision + timeout

        # all episodes end
        success_rate = success / num_traj_tot
        collision_rate = collision / num_traj_tot
        timeout_rate = timeout / num_traj_tot

        avg_nav_time = sum(success_times) / len(
            success_times) if success_times else envs.envs[0].time_limit  # baseEnv.env.time_limit

        mean_end_goal = np.mean(winning_goal_pos, axis=0)

        x = []
        y = []
        for i in range(len(winning_goal_pos)):
            x.append(winning_goal_pos[i][0])
            y.append(winning_goal_pos[i][1])



        plt.close('all')
        plt.scatter(x, y, c=time_step_win_end, cmap='viridis')
        plt.colorbar()

        # plt.plot(x,y,'o')

        # for i in range(len(time_step_win_end)):
        #     plt.annotate(time_step_win_end[i], (x[i], y[i] + 0.2))


        plt.show()
        plt.savefig('books_read.png')


        # logging
        print(
            '=================\nNavigation:\n testing success rate: {:.2f}\n collision rate (per agent, per episode): {:.2f}\n timeout rate: {:.2f}\n '
            'nav time: {:.2f}\n path length: {:.2f}\n average intrusion ratio: {:.2f}%\n '
            'average minimal distance during intrusions: {:.2f}\n average minimum distance to goal: {:.2f}\n average end distance to goal: {:.2f}\n average position of reached goal: {:.2f},{:.2f}\n'.
                format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                    np.mean(too_close_ratios), np.mean(dist_intrusion), np.mean(min_dist_goal_tot), np.mean(end_dist_goal_tot), mean_end_goal[0], mean_end_goal[1]))

        # print('=================\nCases:')
        # print(' collision: ' + ' '.join([str(x) for x in collision_cases]))
        # print(' timeout: ' + ' '.join([str(x) for x in timeout_cases]))
        print("\nEvaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))

            
