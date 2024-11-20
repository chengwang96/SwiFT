import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=288, num_heads=8, num_layers=6, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size, 
                nhead=num_heads, 
                dim_feedforward=emb_size * forward_expansion, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, num_tokens=160, emb_size=288, num_heads=8, num_layers=6, forward_expansion=4, dropout=0.1):
        super().__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, emb_size))
        self.transformer_encoder = TransformerEncoder(emb_size=emb_size, num_heads=num_heads, num_layers=num_layers, forward_expansion=forward_expansion, dropout=dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_outputs)
        )

    def forward(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        B, L, C = x.shape
        assert L == self.pos_embed.shape[1] - 1, f"Expected input with {self.pos_embed.shape[1] - 1} tokens, but got {L} tokens."
        assert C == self.cls_token.shape[2], f"Expected input embedding size of {self.cls_token.shape[2]}, but got {C}."
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :L+1, :]
        x = self.transformer_encoder(x)
        cls_token_final = x[:, 0]
        x = self.mlp_head(cls_token_final)
        
        return x
