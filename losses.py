import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a_embed, p_embed, n_embed):
        p_dist = (a_embed - p_embed).pow(2).sum(1).sqrt()
        n_dist = (a_embed - n_embed).pow(2).sum(1).sqrt()
        print(f"\n{((n_embed == p_embed) | (n_embed == 0)).sum().item()}/{n_embed.size().numel()} are equal or zero")
        loss = torch.relu(p_dist - n_dist + self.alpha)
        loss = torch.where(loss > 0, loss, 0)
        return loss.sum()

def triplet_acc():
    def get_triplet_acc(a_embed, p_embed, n_embed):
        p_dist = (a_embed - p_embed).pow(2).sum(1).sqrt()
        n_dist = (a_embed - n_embed).pow(2).sum(1).sqrt()
        return (p_dist < n_dist)
    return get_triplet_acc


class ContrastLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity()
    
    def forward(self, a_embed, p_embed, n_embed):
        print(f"\n{((n_embed == p_embed) | (n_embed == 0)).sum().item()}/{n_embed.size().numel()} are equal or zero")
        p_cosine = self.cos(a_embed, p_embed)
        n_cosine = self.cos(a_embed, n_embed)

        numer = torch.exp(p_cosine / self.temp).sum()
        denom = torch.exp(torch.concat([p_cosine, n_cosine]) / self.temp).sum()
        loss = -torch.log(numer / denom).sum()

        numer = torch.exp(n_cosine / self.temp).sum()
        loss += -torch.log(numer / denom).sum()
        return loss

def contrast_acc():
    cos = nn.CosineSimilarity()
    def get_contrast_acc(a_embed, p_embed, n_embed):
        p_cosine = cos(a_embed, p_embed)
        n_cosine = cos(a_embed, n_embed)
        return (p_cosine > n_cosine)
    return get_contrast_acc

# NCA-based loss from hard negative paper.
class NcaHnLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity()
    
    def forward(self, a_embed, p_embed, n_embed):
        print(f"\n{((n_embed == p_embed) | (n_embed == 0)).sum().item()}/{n_embed.size().numel()} are equal or zero")
        p_cosine = self.cos(a_embed, p_embed)
        n_cosine = self.cos(a_embed, n_embed)
        normal_loss = -torch.log(torch.exp(p_cosine) / (torch.exp(p_cosine) + torch.exp(n_cosine)))
        return torch.where(n_cosine > p_cosine, n_cosine, normal_loss).sum()

# Margin-based loss from hard negative paper.
class MarginHnLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.cos = nn.CosineSimilarity()
        self.alpha = alpha
    
    def forward(self, a_embed, p_embed, n_embed):
        print(f"\n{((n_embed == p_embed) | (n_embed == 0)).sum().item()}/{n_embed.size().numel()} are equal or zero")
        p_dist = (a_embed - p_embed).pow(2).sum(1).sqrt()
        n_dist = (a_embed - n_embed).pow(2).sum(1).sqrt()
        normal_loss = torch.relu(p_dist - n_dist + self.alpha)
        return torch.where(n_dist < p_dist, n_dist, normal_loss).sum()

# Mixes contrast loss from whodunnit and loss from hard negative paper.
class MixedLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity()
    
    def forward(self, a_embed, p_embed, n_embed):
        print(f"\n{((n_embed == p_embed) | (n_embed == 0)).sum().item()}/{n_embed.size().numel()} are equal or zero")
        p_cosine = self.cos(a_embed, p_embed)
        n_cosine = self.cos(a_embed, n_embed)

        hn = (n_cosine > p_cosine)
        numer = torch.exp(p_cosine[~hn] / self.temp).sum()
        denom = torch.exp(torch.concat([p_cosine[~hn], n_cosine[~hn]]) / self.temp).sum()
        not_hn_loss = -torch.log(numer / denom).sum()
        numer = torch.exp(n_cosine[~hn] / self.temp).sum()
        not_hn_loss += -torch.log(numer / denom).sum()
        hn_loss = n_cosine[hn].sum()
        return hn_loss + not_hn_loss