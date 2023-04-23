import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a_embed, p_embed, n_embed):
        return torch.max(torch.cdist(a_embed, p_embed) - torch.cdist(a_embed, n_embed) + self.alpha, 0)

