# model.py
import torch, torch.nn as nn, math, config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]; return self.dropout(x)

class TransformerDQN(nn.Module):
    def __init__(self, n_features, n_actions):
        super(TransformerDQN, self).__init__()
        self.encoder = nn.Linear(n_features, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL, max_len=config.SEQUENCE_LENGTH)
        encoder_layers = nn.TransformerEncoderLayer(config.D_MODEL, config.N_HEAD, config.D_MODEL * 4, 0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.N_LAYERS)
        self.decoder = nn.Linear(config.D_MODEL, n_actions)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1; self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_(); self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src):
        src = self.encoder(src) * math.sqrt(src.shape[2])
        src = src.transpose(0, 1); src = self.pos_encoder(src); src = src.transpose(0, 1)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1); output = self.decoder(output)
        return output