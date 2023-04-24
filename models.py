# Define all model architectures here.

import torch.nn as nn
import transformers

class DebertaBase(nn.Module):
    def __init__(self, model_name, embed_size, probe=True):
        super().__init__()
        self.deberta = transformers.DebertaV2Model.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU()
        )

        # Freeze deberta if a linear probe is requested. TODO: Check if we need to update optimizer for this.
        if probe:
            self.deberta.requires_grad_(False)

    def forward(self, input):
        return self.mlp(self.deberta(input))
