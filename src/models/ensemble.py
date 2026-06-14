import torch
import torch.nn as nn
import numpy as np
from typing import List


class Ensemble(nn.Module):
    def __init__(self, models: List[nn.Module], method: str = 'soft'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        if self.method == 'soft':
            probs = torch.stack([torch.softmax(o, dim=-1) for o in outputs], dim=0)
            return probs.mean(dim=0)
        elif self.method == 'hard':
            votes = torch.stack([o.argmax(dim=-1) for o in outputs], dim=0)
            batch_size = votes.size(1)
            result = torch.zeros(batch_size, votes[0].max() + 1, device=x.device)
            for i in range(len(self.models)):
                result[range(batch_size), votes[i]] += 1
            return result.float() / len(self.models)
        return outputs[0]
