import torch
import torch.nn as nn
from linformer import Linformer
from performer_pytorch import Performer
import torch.nn.functional as F


class Shallow_deep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.input_conv = nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 25))
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19,1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(1400, 100)

    def forward(self, x):
        # if x.shape[1] != 19:
        #     x = x.transpose(1, 2)
        #x = self.input_conv(x)
        x = self.temporal_conv(x) 
        x = self.bn1(x)

        x_temp = x.clone()

        x = self.spatial_conv(x)
        x = self.bn2(x)
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
        x_final = self.fc(x) 


        x_temp = self.mean_pool(x_temp)  
        x_temp = x_temp.reshape(x_temp.size(0), -1)
        x_temp = self.deep_supervision(x_temp)
        return x_final, x_temp


class ContrastiveNet_deep(nn.Module):
    def __init__(self, emb_net, emb_size):
        super(ContrastiveNet_deep, self).__init__()
        self.emb_net = emb_net
        self.clf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(emb_size, 1)
        )

    def forward(self, x1, x2):
        # print(f"x1 shape: {x1.shape}") 
        # print(f"x2 shape: {x2.shape}")
        z1, z1_temp = self.emb_net(x1)
        z2, z2_temp = self.emb_net(x2)


        score_main = self.clf(torch.abs(z1 - z2)).flatten()
        score_deep = self.clf(torch.abs(z1_temp - z2_temp)).flatten()
        # diff = torch.abs(z1 - z2)
        # out = self.clf(diff)
        # return torch.sigmoid(out)
        return score_main, score_deep


class Shallow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.input_conv = nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 25))
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19,1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # if x.shape[1] != 19:
        #     x = x.transpose(1, 2)
        #x = self.input_conv(x)
        x = self.temporal_conv(x) 
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
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




class ContrastiveNetB(nn.Module):
    def __init__(self, emb_net, emb_size):
        super(ContrastiveNetB, self).__init__()
        self.emb_net = emb_net
        self.clf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(emb_size, 1)
        )

        self.mlp_classifier = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.BatchNorm1d(128),       
            nn.LeakyReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),       
            nn.LeakyReLU(), 
            nn.Dropout(0.5),
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32),       
            nn.LeakyReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(32, 2)
        )

    def forward(self, x1, x2, anchor_labels):
        z1 = self.emb_net(x1)  
        z2 = self.emb_net(x2)
        contrastive_output = self.clf(torch.abs(z1 - z2)).flatten()

        anchor_output = self.mlp_classifier(z1)

        classification_loss = F.cross_entropy(anchor_output, anchor_labels)

        return contrastive_output, classification_loss


class Shallow_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super( ).__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)
  
        self.deep_supervision = nn.Linear(1400, 100)
        
        self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)

        x_temp = x.clone()

        residual = self.residual_conv(x)  

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)  
        
        x += residual
        
        x = self.mean_pool(x)
        x = x.reshape(x.size(0), -1)

        x = torch.log(x + 1e-6)
        
        x = self.dropout(x)
        x_final = self.fc(x)

        x_temp = self.mean_pool(x_temp)
        x_temp = x_temp.reshape(x_temp.size(0), -1)
        x_temp = self.deep_supervision(x_temp)

        return x_final, x_temp

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes=in_channels)
        
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x  
        #x = self.spatial_attention(x) * x  
        return x



class Shallow_deep_with_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(1400, 100)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention = CBAM(in_channels=40)

       # self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        #residual = self.residual_conv(x) 
        x_temp = x.clone() 
        x = self.attention(x)
   
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.attention(x)
       # x = x + residual
        x = x ** 2

        # x = self.attention(x)

        x = self.mean_pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x_final = self.fc(x)


        if self.training:
            x_temp = self.mean_pool2(x_temp)
            x_temp = x_temp.reshape(x_temp.size(0), -1)  
            x_temp = self.deep_supervision(x_temp)
            return (x_final+x_temp)/2
        else:
            return x_final




class Shallow_deep_with_selfattention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.down=nn.Linear(1,16)
        self.self_attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True, dropout=0.5) 
        self.up = nn.Linear(16,40)

        encoder_layer = nn.TransformerEncoderLayer(d_model=40, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(1400, 100)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))


        self.pool = nn.AvgPool2d(kernel_size=(8, 1), stride=(8, 1))
        # self.attention = CBAM(in_channels=40)

       # self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        # batch_size, channels, height, width = x.shape
        # # x = x.view(batch_size, channels, -1).permute(0, 2, 1) 
        # x = x.view(batch_size, height * width, channels)
        # x = self.down(x)
        # x, _ = self.self_attention(x, x, x) 
        # x = self.up(x)
        # # x = x.permute(0, 2, 1).view(batch_size, 40, height, width)
        # x = x.view(batch_size, height, width, 40).permute(0, 3, 1, 2)
        
    
        # x = self.temporal_conv(x)
        # x_temp = x.clone()  # Clone for deep supervision
        # x = self.bn2(x)

        # batch_size, channels, height, width = x.shape
        # x = x.view(batch_size, channels, height * width).permute(2, 0, 1)
        # x = self.transformer(x)
        # x = x.permute(1, 2, 0).view(batch_size, channels, height, width)
        x = self.spatial_conv(x)
        x_temp = x.clone() 
        x = self.bn1(x)
        x = self.pool(x)

        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).permute(2, 0, 1)
        
   
        x = self.transformer(x)
        
  
        x = x.permute(1, 2, 0).view(batch_size, channels, height, width)
        x = x ** 2
        # x = self.attention(x)

        x = self.mean_pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x_final = self.fc(x)

        if self.training:
            x_temp = self.mean_pool2(x_temp)
            x_temp = x_temp.reshape(x_temp.size(0), -1)
            x_temp = self.deep_supervision(x_temp)
            return (x_final + x_temp) / 2
        else:
            return x_final




 


class Shallow_deep_with_linformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.down = nn.Linear(40, 20)

        self.self_attention = Linformer(
            dim=40, seq_len=19*526, depth=1, heads=4, k=16,  
            dropout=0.1 
        )

        # self.self_attention = Performer(
        #     dim=16,  # Input embedding dimension
        #     depth=1,  # Number of layers, here 1 layer of Performer
        #     heads=2,  # Number of attention heads
        #     causal=False  
        # )





        self.up = nn.Linear(20, 40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1600, 100)
        self.dropout = nn.Dropout(p=0.3)
        self.deep_supervision = nn.Linear(1400, 100)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention = CBAM(in_channels=40)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x_temp = x.clone()  # Clone for deep supervision

        # x = self.spatial_conv(x)
        # x = self.bn2(x)

        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # reshape for attention
        # x = self.down(x)
        
        # Using Linformer instead of standard self-attention
        x = self.self_attention(x)
        
        # x = self.up(x)
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)

        x = x ** 2
        #x = self.attention(x)

        x = self.mean_pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x_final = self.fc(x)

        if self.training:
            x_temp = self.mean_pool2(x_temp)
            x_temp = x_temp.reshape(x_temp.size(0), -1)
            x_temp = self.deep_supervision(x_temp)
            return (x_final + x_temp) / 2
        else:
            return x_final





class Shallow_deep_with_performer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 75), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.down = nn.Linear(40, 20)

        # self.self_attention = Linformer(
        #     dim=20, seq_len=600, depth=1, heads=4, k=64,  
        #     dropout=0.1 
        # )

        self.self_attention = Performer(
            dim=20,  # Input embedding dimension
            depth=1,  # Number of layers, here 1 layer of Performer
            heads=4,  # Number of attention heads
            dim_head=5,
            causal=False  
        )





        self.up = nn.Linear(20, 40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(1400, 100)
        self.dropout = nn.Dropout(p=0.3)
        self.deep_supervision = nn.Linear(1400, 100)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention = CBAM(in_channels=40)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x_temp = x.clone()  