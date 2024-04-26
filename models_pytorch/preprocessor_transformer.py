import torch
import torch.nn as nn



class EEGTransformer(nn.Module):
    def __init__(self, channels, samples, num_heads, ffn_dim, num_layers, dropout=0.1):
        super(EEGTransformer, self).__init__()
        self.channels = channels
        self.samples = samples
        self.d_model = 24

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.samples, self.d_model))

        # Input projection layer
        self.input_projection = nn.Linear(channels, self.d_model)

        # Transformer Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Projection layer to original channel dimension
        self.projection_layer = nn.Linear(self.d_model, channels)

    def forward(self, x: torch.Tensor):
        # Adjust input if necessary
        x = x.permute(0, 2, 1)  # Change to (batch_size, samples, channels)

        # Project input to the model dimension
        x = self.input_projection(x)

        # Apply positional encoding
        x = x + self.positional_encoding

        # Pass through Transformer
        x = self.transformer_encoder(x)

        # Project back to the original number of channels
        x = self.projection_layer(x)
        x = x.permute(0, 2, 1)  # Change back to (batch_size, channels, samples)

        return x



# Testing
if __name__ == "__main__":
    # Parameters for the model
    CHANNELS = 22
    SAMPLES = 257
    NUM_HEADS = 6
    FFN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.1

    # Create the model
    model = EEGTransformer(channels=CHANNELS, samples=SAMPLES,
                           num_heads=NUM_HEADS, ffn_dim=FFN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)
    print(model)

    # Example input tensor
    input_tensor = torch.rand(10, CHANNELS, SAMPLES)  # (batch_size, channels, samples)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should print: torch.Size([10, 22, 381])