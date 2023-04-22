import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pass
