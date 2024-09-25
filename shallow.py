import torch
import torch.nn as nn


class Shallow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1360, 100)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.temporal_conv(x) 
        x = self.spatial_conv(x)
        x = self.mean_pool(x) 
        x = x ** 2
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 1360)
        # Apply log transform
        x = torch.log(x + 1e-6)
        # Dropout layer
        x = self.dropout(x)
        x = self.fc(x) 
        return x


class ContrastiveNet(nn.Module):
    def __init__(self, emb_net, emb_size):
        super(ContrastiveNet, self).__init__()
        self.emb_net = emb_net
        self.clf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(emb_size, 1)
        )

    def forward(self, x1, x2):
        z1 = self.emb_net(x1)
        z2 = self.emb_net(x2)
        # diff = torch.abs(z1 - z2)
        # out = self.clf(diff)
        # return torch.sigmoid(out)
        return self.clf(torch.abs(z1 - z2)).flatten()