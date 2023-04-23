# Define all model architectures here.

import torch.nn as nn
import transformers

class DebertaBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.deberta = transformers.DebertaV2Model.from_pretrained('microsoft/deberta-v3-xsmall')

    def forward(self, input):
        return self.deberta(input)
