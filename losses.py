import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a_embed, p_embed, n_embed):
        return torch.max(torch.cdist(a_embed, p_embed) - torch.cdist(a_embed, n_embed) + self.alpha, 0)

def triplet_acc(a_embed, p_embed, n_embed, alpha):
    return torch.cdist(a_embed, p_embed) + alpha < torch.cdist(a_embed, n_embed)

class ConstrastLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
    
    def forward(self, a_embed, p_embed, n_embed):
        p_cosine = nn.CosineSimilarity(a_embed, p_embed)
        n_cosine = nn.CosineSimilarity(a_embed, n_embed)
        return -torch.log(torch.exp(p_cosine / self.temp) / torch.exp(n_cosine / self.temp))
