# Define all model architectures here.

import torch
import torch.nn as nn
import transformers

class DebertaBase(nn.Module):
    def __init__(self, model_name, embed_size, freeze=True):
        super().__init__()
        self.deberta = transformers.DebertaModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU()
        )

        # Freeze deberta if a freeze is requested. TODO: Check if we need to update optimizer for this.
        if freeze:
            self.deberta.requires_grad_(False)

    def forward(self, input):
        # Gets "pooled output" from deberta. Then runs this pooled output through mlp.
        # deberta_4 = torch.flatten(self.deberta(input).last_hidden_state[:, 0:4], start_dim=1)
        # return self.mlp(deberta_4)
        return self.mlp(self.deberta(input).last_hidden_state[:, 0])

class BertBase(nn.Module):
    def __init__(self, model_name, embed_size, freeze=True):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU()
        )

        # Freeze bert if a freeze is requested. TODO: Check if we need to update optimizer for this.
        if freeze:
            self.bert.requires_grad_(False)

    def forward(self, input):
        return self.mlp(self.bert(input).last_hidden_state)
