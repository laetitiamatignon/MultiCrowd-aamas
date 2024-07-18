import torch
from torch.autograd import Variable
import torch.nn.functional as F

def edge_sampler(edge_multinomial, tau=1., hard=False):
    """
    Sample from edge_multinomial using gumbel softmax for differentiable search.
    """
    logits = torch.log(edge_multinomial+1e-10)
    sampled_edges = gumbel_softmax(logits, tau=tau, hard=hard, eps=1e-10)
    return sampled_edges

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)

def sample_gumbel(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return - torch.log(eps - torch.log(uniform + eps))