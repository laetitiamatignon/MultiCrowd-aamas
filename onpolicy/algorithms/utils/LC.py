import torch.nn as nn
import torch
import numpy as np
from onpolicy.algorithms.utils.util import init, check
import math
from torch.nn.functional import softmax
from onpolicy.algorithms.utils.rnn import EndRNN
from onpolicy.algorithms.utils.edge_selector import edge_selector
from onpolicy.algorithms.utils.mha import multi_head_attention
from onpolicy.algorithms.utils.gat import GAT
from onpolicy.algorithms.utils.gumbel_softmax import edge_sampler

class LC(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, input_entities_size, args, device=torch.device("cpu")):
        super(LC, self).__init__()
        # self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args.tpdv = self.tpdv
        self.device = device

        self.episode_length = args.episode_length
        self.seq_length = args.seq_length
        self.nenv = args.n_training_threads
        self.nminibatch = args.num_mini_batch

        if args.archi_name == 'GHR':
            # Store required sizes
            self.human_node_rnn_size = args.GHR_human_node_rnn_size
            self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
            self.output_size = args.GHR_human_node_output_size
            self.edge_selector_embedding_size = args.GHR_edge_selector_embedding_size
            self.edge_selector_emb_size = args.GHR_edge_selector_emb_size
            self.edge_selector_num_head = args.GHR_edge_selector_num_head
            self.mha_emb_size = args.GHR_mha_emb_size
            self.mha_num_head = args.GHR_mha_num_head
        elif args.archi_name == 'GHGR':
            # Store required sizes
            self.human_node_rnn_size = args.GHGR_human_node_rnn_size
            self.human_human_edge_rnn_size = args.GHGR_human_human_edge_rnn_size
            self.output_size = args.GHGR_human_node_output_size
            self.edge_selector_embedding_size = args.GHGR_edge_selector_embedding_size
            self.edge_selector_emb_size = args.GHGR_edge_selector_emb_size
            self.edge_selector_num_head = args.GHGR_edge_selector_num_head
            self.mha_emb_size = args.GHGR_mha_emb_size
            self.mha_num_head = args.GHGR_mha_num_head
        elif args.archi_name == 'LHGR':
            # Store required sizes
            self.human_node_rnn_size = args.LHGR_human_node_rnn_size
            self.human_human_edge_rnn_size = args.LHGR_human_human_edge_rnn_size
            self.output_size = args.LHGR_human_node_output_size
            self.edge_selector_embedding_size = args.LHGR_edge_selector_embedding_size
            self.edge_selector_emb_size = args.LHGR_edge_selector_emb_size
            self.edge_selector_num_head = args.LHGR_edge_selector_num_head
            self.mha_emb_size = args.LHGR_mha_emb_size
            self.mha_num_head = args.LHGR_mha_num_head
        else:
            # Store required sizes
            self.human_node_rnn_size = args.GHR_human_node_rnn_size
            self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
            self.output_size = args.GHR_human_node_output_size
            self.edge_selector_embedding_size = args.GHR_edge_selector_embedding_size
            self.edge_selector_emb_size = args.GRH_edge_selector_emb_size
            self.edge_selector_num_head = args.GRH_edge_selector_num_head
            self.mha_emb_size = args.GRH_mha_emb_size
            self.mha_num_head = args.GRH_mha_num_head

        self.num_agents = args.num_agents

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        self.edge_selector = edge_selector(input_entities_size, self.edge_selector_emb_size, self.mha_emb_size, self.edge_selector_num_head)
        self.linear = nn.Linear(1, 1)
        self.gat = GAT(1, [self.mha_num_head], [self.mha_emb_size, self.mha_emb_size], add_skip_connection=False, bias=False, dropout=0.25, log_attention_weights=False)


    # input : [batch_size, num_entities, num_feature]
    # visibility : [batch_size, num_entities, num_entities]
    def forward(self, input, visibility, id_robot, tau=1.):
        # EDGE_SELECTOR
        clustered_entities, clustered_head_entities, weight_clustered_entities = self.edge_selector(input, visibility)

        # GUMBEL_SOFTMAX
        A = self.linear(weight_clustered_entities.unsqueeze(-1)).squeeze(-1)
        self.hard = False

        edge_multinomial = softmax(A, dim=-1)
        new_mask = []
        for _ in range(self.edge_selector_num_head):
            new_mask.append(visibility)
        new_mask = torch.stack(new_mask, dim=1)
        edge_multinomial = edge_multinomial * new_mask
        edge_multinomial = edge_multinomial / (edge_multinomial.sum(-1).unsqueeze(-1) + 1e-10)
        sampled_edges = edge_sampler(edge_multinomial, tau=tau, hard=self.hard)
        sampled_edges = sampled_edges.sum(1)

        # GAT
        attn_agents = self.gat(clustered_entities, sampled_edges * visibility)

        # For each env, there is an analysis of num_agents + num_humans entities
        if id_robot is not None:
            hidden_attn_weighted = attn_agents[0][torch.arange(attn_agents[0].size(0)), -id_robot]
        else:
            hidden_attn_weighted = attn_agents[0]

        return hidden_attn_weighted

