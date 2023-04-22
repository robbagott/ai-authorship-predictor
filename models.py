# Define all model architectures here.

import torch.nn as nn
import transformers

class DebertaBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None # TODO

    def forward(self, input):
        return self.model(input)
