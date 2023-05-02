import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a_embed, p_embed, n_embed):
        p_dist = (a_embed - p_embed).pow(2).sum(2).sqrt()
        n_dist = (a_embed - n_embed).pow(2).sum(2).sqrt()
        loss = torch.relu(p_dist - n_dist + self.alpha)
        return loss.sum()

def triplet_acc(alpha):
    return lambda a_embed, p_embed, n_embed: torch.cdist(a_embed, p_embed) + alpha < torch.cdist(a_embed, n_embed)

class ConstrastLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
    
    def forward(self, a_embed, p_embed, n_embed):
        p_cosine = nn.CosineSimilarity(a_embed, p_embed)
        n_cosine = nn.CosineSimilarity(a_embed, n_embed)
        return -torch.log(torch.exp(p_cosine / self.temp) / torch.exp(n_cosine / self.temp))
