import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    
class CNNTransformerClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, model_dim=64, num_heads=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, model_dim//2, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(model_dim//2, model_dim, kernel_size=5, stride=2, padding=2)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        out = self.fc_out(x)
        return out