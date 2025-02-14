import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_model, 7)

    def forward(self, x, lengths):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        x = lstm_out[torch.arange(len(lengths)), lengths - 1, :]
        x = self.fc(x)
        return x
    
