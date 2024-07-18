import torch
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.GHR import GHR
from onpolicy.algorithms.utils.GHGR import GHGR
from onpolicy.algorithms.utils.LHGR import LHGR
from onpolicy.algorithms.utils.DSRNNpp import selfAttn_merge_SRNN
from onpolicy.algorithms.utils.distributions import Bernoulli, Categorical, DiagGaussian

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if args.archi_name == "GHR":
            self.network = GHR(self.obs_space, args, self.device)
        elif args.archi_name == "GHGR":
            self.network = GHGR(self.obs_space, args, self.device)
        elif args.archi_name == "LHGR":
            self.network = LHGR(self.obs_space, args, self.device)
        elif args.archi_name == "DSRNNpp" or args.archi_name == "DSRNNoff":
            self.network = selfAttn_merge_SRNN(self.obs_space, args, self.device)
        else:
            print('Par defaut')
            self.network = selfAttn_merge_SRNN(self.obs_space, args, self.device)
        self.args = args

        if act_space.__class__.__name__ == "Discrete":
            num_outputs = act_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif act_space.__class__.__name__ == "Box":
            num_outputs = act_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        elif act_space.__class__.__name__ == "MultiBinary":
            num_outputs = act_space.shape[0]
            self.dist = Bernoulli(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.dist.to(device)
        self.optimizer = torch.optim.Adam([{'params' : self.network.parameters()}, {'params': self.dist.parameters()}],
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.network, episode, episodes, self.lr)
        update_linear_schedule(self.dist, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, tau, available_actions=None,
                    deterministic=False):

        values, actor_features, rnn_states_actor = self.network(obs, rnn_states_actor, masks, tau, infer=True)
        dist = self.dist(actor_features)

        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()

        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, tau):

        values, _, _ = self.network(cent_obs, rnn_states_critic, masks, tau, infer=True)

        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, tau,
                         available_actions=None, active_masks=None):
        values, actor_features, _ = self.network(obs, rnn_states_actor, masks, tau)

        dist = self.dist(actor_features)


        action_log_probs = dist.log_probs(torch.from_numpy(action).to(self.device))
        dist_entropy = dist.entropy().mean()

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, tau, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        _ , actor_features, rnn_states_actor = self.network(obs, rnn_states_actor, masks, tau, infer=True)
        dist = self.dist(actor_features)


        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()
        

        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return actions, rnn_states_actor

