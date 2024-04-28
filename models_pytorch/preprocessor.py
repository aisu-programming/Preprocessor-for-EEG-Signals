import torch
import torch.nn as nn
import torch.nn.functional as F





class PreTransformer(nn.Module):
    def __init__(self, channels, samples, num_heads, ffn_dim, num_layers, dropout=0.1):
        super(PreTransformer, self).__init__()
        self.channels = channels
        self.samples = samples
        self.d_model = 24
        self.positional_encoding = nn.Parameter(torch.randn(1, self.samples, self.d_model))
        self.input_projection = nn.Linear(channels, self.d_model)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.projection_layer = nn.Linear(self.d_model, channels)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # Change to (batch_size, samples, channels)
        x = self.input_projection(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.projection_layer(x)
        x = x.permute(0, 2, 1)  # Change back to (batch_size, channels, samples)
        return x
    


class PreLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(PreLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # Change to (batch_size, samples, channels)
        batch_size, seq_len, features = x.size()
        x: torch.Tensor = self.lstm(x)[0]              # (batch_size, seq_len, hidden_size)
        x = x.contiguous().view(-1, self.hidden_size)  # (batch_size*seq_len,  hidden_size)
        x = F.dropout(x, self.dropout)
        x: torch.Tensor = self.linear(x)               # (batch_size*seq_len,  features)
        x = x.view(batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # Change back to (batch_size, channels, samples)
        return x



def test():
    CHANNELS = 22
    SAMPLES = 257
    NUM_HEADS = 6
    FFN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.1
    model = PreTransformer(channels=CHANNELS, samples=SAMPLES,
                           num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                           num_layers=NUM_LAYERS, dropout=DROPOUT)
    input_tensor = torch.rand(10, CHANNELS, SAMPLES)
    output_tensor: torch.Tensor = model(input_tensor)
    assert output_tensor.shape == (10, CHANNELS, SAMPLES)
    model = PreLSTM(input_size=CHANNELS, hidden_size=64,
                    num_layers=NUM_LAYERS, dropout=DROPOUT)
    input_tensor = torch.rand(10, CHANNELS, SAMPLES)
    output_tensor: torch.Tensor = model(input_tensor)
    assert output_tensor.shape == (10, CHANNELS, SAMPLES)
    return





# Testing
if __name__ == "__main__":
    test()