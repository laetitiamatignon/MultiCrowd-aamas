import torch.nn.functional as F
import torch.nn as nn
import torch

class RNNBase(nn.Module):
    """
    The class for RNN with done masks
    """
    # edge: True -> edge RNN, False -> node RNN
    def __init__(self, args, edge):
        super(RNNBase, self).__init__()
        self.args = args

        if args.archi_name == 'GHR':
            self.human_human_edge_embedding_size = args.GHR_human_human_edge_embedding_size
            self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
            self.human_node_embedding_size = args.GHR_human_node_embedding_size
            self.human_node_rnn_size = args.GHR_human_node_rnn_size
        elif args.archi_name == 'GHGR':
            self.human_human_edge_embedding_size = args.GHR_human_human_edge_embedding_size
            self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
            self.human_node_embedding_size = args.GHR_human_node_embedding_size
            self.human_node_rnn_size = args.GHR_human_node_rnn_size
        elif args.archi_name == 'LHGR':
            self.human_human_edge_embedding_size = args.GHR_human_human_edge_embedding_size
            self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
            self.human_node_embedding_size = args.GHR_human_node_embedding_size
            self.human_node_rnn_size = args.GHR_human_node_rnn_size
        else:
            self.human_human_edge_embedding_size = args.GHR_human_human_edge_embedding_size
            self.human_human_edge_rnn_size = args.GHR_human_human_edge_rnn_size
            self.human_node_embedding_size = args.GHR_human_node_embedding_size
            self.human_node_rnn_size = args.GHR_human_node_rnn_size

        # if this is an edge RNN
        if edge:
            self.gru = nn.GRU(self.human_human_edge_embedding_size, self.human_human_edge_rnn_size)
        # if this is a node RNN
        else:
            self.gru = nn.GRU(self.human_node_embedding_size*2, self.human_node_rnn_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    # x: [seq_len, nenv, 6 or 30 or 36, ?]
    # hxs: [1, nenv, human_num, ?]
    # masks: [1, nenv, 1]
    def _forward_gru(self, x, hxs, masks):
        # for acting model, input shape[0] == hidden state shape[0]
        if x.size(0) == hxs.size(0):
            # use env dimension as batch
            # [1, 12, 6, ?] -> [1, 12*6, ?] or [30, 6, 6, ?] -> [30, 6*6, ?]
            seq_len, nenv, agent_num, _ = x.size()
            x = x.view(seq_len, nenv*agent_num, -1)
            mask_agent_num = masks.size()[-1]
            hxs_times_masks = hxs * (masks.view(seq_len, nenv, mask_agent_num, 1))
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv*agent_num, -1)

            x, hxs = self.gru(x, hxs_times_masks) # we already unsqueezed the inputs in SRNN forward function
            x = x.view(seq_len, nenv, agent_num, -1)
            hxs = hxs.view(seq_len, nenv, agent_num, -1)

        # during update, input shape[0] * nsteps (30) = hidden state shape[0]
        else:
            # N: nenv, T: seq_len, agent_num: node num or edge num
            T, N, agent_num, _ = x.size()

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # for the [29, num_env] boolean array, if any entry in the second axis (num_env) is True -> True
            # to make it [29, 1], then select the indices of True entries
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # x and hxs have 4 dimensions, merge the 2nd and 3rd dimension
                x_in = x[start_idx:end_idx]
                x_in = x_in.view(x_in.size(0), x_in.size(1)*x_in.size(2), x_in.size(3))
                hxs = hxs.view(hxs.size(0), N, agent_num, -1)
                hxs = hxs * (masks[start_idx].view(1, -1, 1, 1))
                hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), hxs.size(3))
                rnn_scores, hxs = self.gru(x_in, hxs)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T, N, agent_num, -1)
            hxs = hxs.view(1, N, agent_num, -1)

        return x, hxs



class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        if args.archi_name == 'GHR':
            self.rnn_size = args.GHR_human_node_rnn_size
            self.output_size = args.GHR_human_node_output_size
            self.embedding_size = args.GHR_human_node_embedding_size
            self.input_size = args.GHR_human_node_input_size
            self.edge_rnn_size = args.GHR_human_human_edge_rnn_size

        else:
            self.rnn_size = args.GHR_human_node_rnn_size
            self.output_size = args.GHR_human_node_output_size
            self.embedding_size = args.GHR_human_node_embedding_size
            self.input_size = args.GHR_human_node_input_size
            self.edge_rnn_size = args.GHR_human_human_edge_rnn_size

        # Store required sizes


        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        # self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)
        self.edge_attention_embed = nn.Linear(256, self.embedding_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)


        return outputs, h_new
