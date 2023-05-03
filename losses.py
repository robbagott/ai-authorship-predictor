import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a_embed, p_embed, n_embed):
        p_dist = (a_embed - p_embed).pow(2).sum(1).sqrt()
        n_dist = (a_embed - n_embed).pow(2).sum(1).sqrt()
        print(a_embed.count_nonzero(), p_embed.count_nonzero(), n_embed.count_nonzero(), p_dist - n_dist)
        loss = torch.relu(p_dist - n_dist + self.alpha)
        return loss.sum()

def triplet_acc(alpha):
    def get_triplet_acc(a_embed, p_embed, n_embed):
        p_dist = (a_embed - p_embed).pow(2).sum(1).sqrt()
        n_dist = (a_embed - n_embed).pow(2).sum(1).sqrt()
        return (p_dist + alpha < n_dist)
    return get_triplet_acc


class ContrastLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity()
    
    def forward(self, a_embed, p_embed, n_embed):
        print(a_embed.count_nonzero(), p_embed.count_nonzero(), n_embed.count_nonzero())
        p_cosine = self.cos(a_embed, p_embed)
        n_cosine = self.cos(a_embed, n_embed)
        return -torch.log(torch.exp(p_cosine / self.temp) / torch.exp(n_cosine / self.temp)).sum()

def contrast_acc(temp):
    cos = nn.CosineSimilarity()
    def get_contrast_acc(a_embed, p_embed, n_embed):
        p_cosine = cos(a_embed, p_embed)
        n_cosine = cos(a_embed, n_embed)
        print((p_cosine < n_cosine).sum())
        return (p_cosine < n_cosine)
    return get_contrast_acc
