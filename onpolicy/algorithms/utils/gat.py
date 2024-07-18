import torch.nn.functional as F
import torch.nn as nn
import torch

import math

class GAT(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.
    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.
    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.
    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=False, bias=False,
                 dropout=0.6, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                input_size=num_features_per_layer[i],  # consequence of concatenation
                embed_size=num_features_per_layer[i+1],
                num_head=num_heads_per_layer[i+1],
                # concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                # activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data, attn_mask):
        res = self.gat_net([data, attn_mask])
        return res


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """
    def __init__(self, input_size, embed_size, num_head, concat=False, activation=None,
                 dropout_prob=0.6, add_skip_connection=False, bias=False, log_attention_weights=False):

        super().__init__()

        self.num_attention_heads = num_head
        self.attention_head_size = int(embed_size / num_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.input_size = input_size
        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, embed_size), nn.ReLU()
                                            )

        self.query = nn.Linear(embed_size, self.all_head_size)
        self.key = nn.Linear(embed_size, self.all_head_size)
        self.value = nn.Linear(embed_size, self.all_head_size)

        self.dense = nn.Linear(embed_size, embed_size)



        # self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        # self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        # self.activation = activation
        # # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        # self.dropout = nn.Dropout(p=dropout_prob)

        # self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        # self.attention_weights = None  # for later visualization purposes, I cache the weights here

        #self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        return

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, data):
        # attn_mask : [env*seq_lenght, num_human, num_human]

        input_traj = data[0]
        attn_mask = data[1]
        new_mask = []
        for _ in range(self.num_attention_heads):
            new_mask.append(attn_mask)
        new_mask = torch.stack(new_mask, dim=1)

        embed_input = self.embedding_layer(input_traj)

        mixed_query_layer = self.query(embed_input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(embed_input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(embed_input)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer   = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        
        attention_scores.masked_fill_(new_mask<0.1, -1e10)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer_2 = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer_2)
        attn = attention_probs.permute(0, 2, 1, 3).contiguous().sum(-2)/self.num_attention_heads 

        return [output, attn_mask, attn]