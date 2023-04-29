# Define all model architectures here.

import torch
import torch.nn as nn
import transformers

class DebertaBase(nn.Module):
    def __init__(self, model_name, embed_size, probe=True):
        super().__init__()
        self.tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_name)
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
        input = self.tokenizer(input, return_tensors="pt")
        return self.mlp(self.deberta(**input))

class BertBase(nn.Module):
    def __init__(self, model_name, embed_size, probe=True):
        super().__init__()
        self.tokenizer = transformers.BertTokenizerFast(model_name)
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU()
        )

        # Freeze bert if a linear probe is requested. TODO: Check if we need to update optimizer for this.
        if probe:
            self.bert.requires_grad_(False)

    def forward(self, input):
        input = self.tokenizer(input, return_tensors="pt")
        return self.mlp(self.bert(**input))

class KNN(nn.Module):
    def __init__(self, k, embeddings, labels):
        super().__init__()
        self.k = k
        self.embeddings = embeddings
        self.labels = labels

    def forward(self, input):
        # Compute differences between input and embeddings.
        diffs = self.embeddings - input
        dists = torch.linalg.vector_norm(diffs, ord=2, dim=1)

        # Get top k smallest distances.
        topk_ind = torch.topk(dists, self.k, largest=False).indices

        # Determine label
        topk_labels = self.labels[topk_ind]
        return torch.mode(topk_labels).values[0]
