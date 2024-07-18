import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from onpolicy.algorithms.utils.util import init, check
import math
from torch.nn.functional import softmax
from onpolicy.algorithms.utils.rnn import EndRNN
from onpolicy.algorithms.utils.edge_selector import edge_selector
from onpolicy.algorithms.utils.mha import multi_head_attention
from onpolicy.algorithms.utils.gat import GAT
from onpolicy.algorithms.utils.gumbel_softmax import edge_sampler
from onpolicy.algorithms.utils.LC import LC

class GHR(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space, args, device=torch.device("cpu"), infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(GHR, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args = args

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args.tpdv = self.tpdv
        self.device = device

        self.predict_steps = args.predict_steps
        self.grid_cell = args.grid_cell

        self.num_entities = obs_space.shape[-1]
        self.num_agents = args.num_agents

        self.episode_length = args.episode_length
        self.seq_length = args.seq_length
        self.nenv = args.n_training_threads
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.GHR_human_node_rnn_size
        self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
        self.output_size = args.GHR_human_node_output_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        if self.grid_cell:
            robot_size = 13
        else:
            robot_size = 9
            
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear = init_(nn.Linear(self.output_size,2))

        if self.grid_cell:
            self.dim_pos = int(4*(self.predict_steps+1))
        else:
            self.dim_pos = int(2*(self.predict_steps+1))

        self.input_entities_size = self.dim_pos + args.label_entity if args.label_entity > 1 else self.dim_pos
        self.LC = LC(self.input_entities_size, args, 'GHR')

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), 
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), 
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), 
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), 
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(args)

        self.to(device)

    def forward(self, inputs, rnn_hxs, masks, tau=1., infer=False):
        inputs = check(inputs).to(**self.tpdv)
        rnn_hxs = check(rnn_hxs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.args.n_rollout_threads * self.args.num_agents
            nagent = self.args.num_agents
            hard = False
        else:
            # seq_length = 30
            # nenv = 8
            seq_length = self.args.data_chunk_length
            nenv = (self.args.n_rollout_threads * self.args.episode_length * self.args.num_agents // self.args.data_chunk_length) // self.args.num_mini_batch
            nagent = self.args.num_agents
            hard = False

        num_visible_entities = int(inputs[0,-1,9].item())
        if self.grid_cell:
            robot_node = reshapeT(inputs[:,-1,:11].unsqueeze(-2), seq_length, nenv)
            temporal_edges = reshapeT(inputs[:,-1,11:13].unsqueeze(-2), seq_length, nenv)
        else:
            robot_node = reshapeT(inputs[:,-1,:7].unsqueeze(-2), seq_length, nenv)
            temporal_edges = reshapeT(inputs[:,-1,7:9].unsqueeze(-2), seq_length, nenv)
        id_robot = inputs[:, -1, 10].long()
        spatial_edges = reshapeT(inputs[:, :num_visible_entities, 1:(self.input_entities_size+1)], seq_length, nenv)
        visibility = inputs[:, num_visible_entities:2*num_visible_entities, :num_visible_entities]

        for i in range(num_visible_entities):
            visibility[:, i, i] = 0
        for i in range(self.num_agents):
            visibility[:, -(i+1), -(i+1)] = 1

        hidden_states_node_RNNs = reshapeT(rnn_hxs, 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        hidden_attn = self.LC(spatial_edges.reshape(nenv*seq_length, num_visible_entities, -1), visibility, id_robot, tau)
        hidden_attn_weighted = hidden_attn.view(seq_length, nenv, -1).unsqueeze(-2)

        # Do a forward pass through GRU
        outputs, h_nodes \
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)

        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs = all_hidden_states_node_RNNs

        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        rnn_hxs.squeeze(1)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))
