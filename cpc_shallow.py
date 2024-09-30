import torch
import torch.nn as nn
import torch.optim as optim

class Shallow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shallow, self).__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x ** 2
        x = self.mean_pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CPCModel(nn.Module):
    def __init__(self, encoder, hidden_size=100, emb_size=100):
        super(CPCModel, self).__init__()
        self.encoder = encoder
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)  # Add classifier layer for downstream task (binary classification)

    def forward(self, x):
        # Encoding with the encoder
        batch_size, channels, time_steps = x.size()
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1, channels, time_steps)
        z = self.encoder(x)  # Shape: (batch_size, 100)

        # Pass through GRU (Context aggregation)
        z = z.unsqueeze(1)  # Reshape for GRU: (batch_size, seq_len, input_size)
        z, _ = self.gru(z)
        z = z.squeeze(1)  # Shape: (batch_size, hidden_size)

        # Classification Layer (for downstream binary classification task)
        out = self.classifier(z)
        return out



