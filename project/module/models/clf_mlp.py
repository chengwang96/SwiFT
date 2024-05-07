import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, num_classes=2, num_tokens = 96):
        super(mlp, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        # torch.Size([16, 288, 2, 2, 2, 20])
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # torch.Size([16, 160, 288])
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        # torch.Size([16, 288, 1])
        x = torch.flatten(x, 1)
        # torch.Size([16, 288])
        x = self.head(x)
        # torch.Size([16, 1])
        return x
