import torch
import torch.nn as nn


class Shallow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.input_conv = nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 25))
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19,1), stride=(1, 1))
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # if x.shape[1] != 19:
        #     x = x.transpose(1, 2)
        #x = self.input_conv(x)
        x = self.temporal_conv(x) 
        x = self.spatial_conv(x)
        x = x ** 2
        x = self.mean_pool(x) 
        #print(x.shape)
        #x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
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
        # print(f"x1 shape: {x1.shape}") 
        # print(f"x2 shape: {x2.shape}")
        z1 = self.emb_net(x1)
        z2 = self.emb_net(x2)
        # diff = torch.abs(z1 - z2)
        # out = self.clf(diff)
        # return torch.sigmoid(out)
        return self.clf(torch.abs(z1 - z2)).flatten()