import torch.nn.functional as F
import torch.nn as nn
import torch

class multi_head_attention(nn.Module):
    def __init__(self, input_size, embed_size, num_head):
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_traj, attn_mask):
        # print(attn_mask)
        # attn_mask : [env*seq_lenght, num_human, num_human]
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
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores.masked_fill_(new_mask<0.1, -1e10)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer_2 = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer_2)

        return output, attention_probs

