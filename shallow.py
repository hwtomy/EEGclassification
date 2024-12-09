import torch
import torch.nn as nn
from linformer import Linformer
from performer_pytorch import Performer
import torch.nn.functional as F
import pywt 
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, GlobalAttention, MeanAggregation, BatchNorm, global_mean_pool, GATConv
import numpy as np
import os
import sys
import pickle


class Shallow_deep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.input_conv = nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 25))
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 17))
        self.fc = nn.Linear(24000, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.mean_pool2 = nn.MaxPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention1 = CBAM(in_channels=160)
        self.deep_supervision = nn.Linear(25080, 128)
        self.bi_lstm = nn.LSTM(input_size=40, 
                               hidden_size=20, 
                               num_layers=2, 
                               dropout=0.5, 
                               bidirectional=True, 
                               batch_first=True)

    def forward(self, x):
        # if x.shape[1] != 19:
        #     x = x.transpose(1, 2)
        #x = self.input_conv(x)
        # x = self.temporal_conv(x) 
        # x = self.bn1(x)
        # x = self.wavelet_transform(x)
        # x_temp = x.clone()
        # x = self.attention1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        # x = self.attention1(x)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1) 
        x= x.permute(0, 2, 1)

        x, _ = self.bi_lstm(x)  


        # x_final = x[:, -1, :] 
        # x = x ** 2
        # x = self.mean_pool(x) 
        #print(x.shape)
        #x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        # Apply log transform
        # x = torch.log(x + 1e-6)
        # Dropout layer
        # x = self.dropout(x)
        x_final = self.fc(x)


        # if self.training:
        #     x_temp = self.mean_pool2(x_temp)
        #     x_temp = x_temp.reshape(x_temp.size(0), -1)  
        #     x_temp = self.deep_supervision(x_temp)
        #     return (x_final+x_temp)/2
        # else:
        #     return x_final
        return x_final
    
    def wavelet_transform(self, x):

        coeffs = pywt.wavedec2(x.detach().cpu().numpy(), wavelet='haar', level=1)
   
        cA, (cH, cV, cD) = coeffs



        cA = torch.tensor(cA, device=x.device)
        cH = torch.tensor(cH, device=x.device)
        cV = torch.tensor(cV, device=x.device)
        cD = torch.tensor(cD, device=x.device)

        x_wavelet = torch.cat((cA, cH, cV, cD), dim=1)
        # x_wavelet = torch.tensor(x_wavelet, device=x.device)
        return x_wavelet



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





class mlpf(nn.Module):
    def __init__(self):
        super(ContrastiveNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  
            nn.BatchNorm1d(64),       
            nn.LeakyReLU(), 
            nn.Dropout(0.5),  

            # nn.Linear(128, 64),       
            # nn.BatchNorm1d(64),  
            # nn.ReLU(),
            # nn.Dropout(0.5),  

            # nn.Linear(64, 32),       
            # nn.BatchNorm1d(32),  
            # nn.LeakyReLU(),
            # nn.Dropout(0.5), 

            # nn.Linear(32, 16),       
            # nn.BatchNorm1d(16),  
            # nn.LeakyReLU(),
            # nn.Dropout(0.5), 

            nn.Linear(64, 1),           
            nn.Sigmoid()                
        )
    
    def forward(self, x):
        return self.model(x)




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
        x = self.spatial_attention(x) * x  
        return x



class Shallow_deep_with_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))


        self.fc = nn.Linear(18432, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(25080, 256)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention1 = CBAM(in_channels=40)
        self.attention2 = CBAM(in_channels=64)
        self.attention3 = CBAM(in_channels=128)

       # self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        #residual = self.residual_conv(x) 
        x_temp = x.clone() 
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.attention1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.attention2(x)
        x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.bn4(x)
        # x = self.attention3(x)
        # x = self.pool3(x)
       # x = x + residual
        x = x ** 2

        # x = self.attention(x)

        # x = self.mean_pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x_final = self.fc(x)


        if self.training:
            # x_temp = self.mean_pool2(x_temp)
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
        x_temp = x.clone()  # Clone for deep supervision

        x = self.spatial_conv(x)
        x = self.bn2(x)

        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # reshape for attention
        x = self.down(x)
        
        # Using Linformer instead of standard self-attention
        x = self.self_attention(x)
        
        x = self.up(x)
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




class LinformerLayer(nn.Module):
    def __init__(self, seq_length, input_dim, num_heads, linformer_k):
        super(LinformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.linformer = Linformer(
            dim=input_dim,
            seq_len=seq_length,
            depth=1,
            heads=num_heads,
            k=linformer_k
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim),
        )

    def forward(self, x):
        x = self.norm1(x)
        residual = x
        attention_output = self.linformer(x)
        x = attention_output + residual
        x = self.norm2(x)
        residual = x
        mlp_output = self.mlp(x)
        x = mlp_output + residual
        return x

class ChannelLinformer(nn.Module):
    def __init__(self, seq_length, input_channels, conv_out_channels, embed_dim, num_heads, linformer_k, num_layers):
        super(ChannelConvLinformer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, conv_out_channels, kernel_size=1, padding=1)
            for _ in range(input_channels)
        ])
        self.norm1 = nn.LayerNorm(conv_out_channels)
        self.stacked_linformer = StackedLinformer(seq_length, conv_out_channels, num_heads, linformer_k, num_layers)
        self.norm2 = nn.LayerNorm(conv_out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_channels, 80),
            nn.GELU(),
            nn.Linear(80, embed_dim)
        )

    def forward(self, x):
        batch_size, feature_dim, seq_length, channels = x.shape  
        channel_outputs = []
        for i in range(channels):
            channel_input = x[:, :, :, i]
            channel_output = self.convs[i](channel_input)
            channel_output = channel_output.permute(0, 2, 1)
            channel_output = self.stacked_linformer(channel_output)
            channel_outputs.append(channel_output)
        fused_output = torch.mean(torch.stack(channel_outputs), dim=0)
        return fused_output

class StackedLinformer(nn.Module):
    def __init__(self, seq_length, input_dim, num_heads, linformer_k, num_layers):
        super(StackedLinformer, self).__init__()
        self.layers = nn.ModuleList([LinformerLayer(seq_length, input_dim, num_heads, linformer_k) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EEGClassificationModel(nn.Module):
    def __init__(self, seq_length=600, input_channels=1, conv_out_channels=40, embed_dim=40, num_heads=4, linformer_k=20, num_layers=8, num_classes=2):
        super(EEGClassificationModel, self).__init__()
        self.channel_processor = ChannelLinformer(
            seq_length=seq_length,
            input_channels=input_channels,
            conv_out_channels=conv_out_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            linformer_k=linformer_k,
            num_layers=num_layers
        )
        # self.mlp_fusion = nn.Sequential(
        #     nn.Linear(embed_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_classes)
        # )
        self.output_layer = nn.Linear(conv_out_channels * seq_length, 256)
    def forward(self, x):
        x = self.channel_processor(x)
        out = self.outlayer(x)
        return out


class Shallow_sep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.input_conv = nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 25))
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19,1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.conv = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(40)
        self.mean_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.fc = nn.Linear(24000, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(24000, 256)

    def forward(self, x):
        x = self.temporal_conv(x) 
        x = self.bn1(x)
        x_temp = x.clone()
        batch_size, feature_dim, seq_length, channels = x.shape  
        channel_outputs = []
        for i in range(channels):
            channel_input = x[:, :, :, i]
            channel_output = self.conv[i](channel_input)
            channel_outputs.append(channel_output)
        fused_output = torch.mean(torch.stack(channel_outputs), dim=0)
        x = fused_output
        x = x ** 2
        x = self.mean_pool(x) 
        x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x_final = self.fc(x) 


        x_temp = self.mean_pool(x_temp)  
        x_temp = x_temp.reshape(x_temp.size(0), -1)
        x_temp = self.deep_supervision(x_temp)
        return x_final, x_temp




class mulconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(mulconv, self).__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(19,1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        
        self.attention1 = CBAM(in_channels=40)
        self.conv1 = nn.Conv2d(in_channels=40, out_channels=16, kernel_size=(40, 3, 3), sride= (1, 2,1), padding=(1, 1, 1))
        self.attention2 = CBAM(in_channels=16)
        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1, 1), padding=(1, 1, 1))
        self.attention3 = CBAM(in_channels=32)
        self.mp2 = nn.MaxPool2d(kernel_size=(1, 1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1, 1), padding=(1, 1, 1))
        self.attention = CBAM(in_channels=64)
        self.mp3 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.fc = nn.Linear(64 * 75 * 9, 256)
        self.fc_out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.attention1(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.attention2(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.attention3(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.attention(x)
        x = self.mp3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.fc_out(x)
        return x










class Shallow_cwt_with_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 25), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(80)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        


        self.fc = nn.Linear(24000, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(25080, 256)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention1 = CBAM(in_channels=40)
        self.attention2 = CBAM(in_channels=80)
        self.attention3 = CBAM(in_channels=128)
        self.bi_lstm = nn.LSTM(input_size=1, 
                               hidden_size=40, 
                               num_layers=1, 
                               dropout=0.5, 
                               bidirectional=True, 
                               batch_first=True)

       # self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        # x = torch.fft.fft(x, dim=-1)
        # x = torch.abs(x)
        # x= torch.stack((m, p), dim=-1)

        # coeffs = pywt.wavedec2(x.detach().cpu().numpy(), wavelet='haar', level=1)
   
        # cA, (cH, cV, cD) = coeffs


         
        # cA = torch.tensor(cA, device=x.device)
        # cH = torch.tensor(cH, device=x.device)
        # cV = torch.tensor(cV, device=x.device)
        # cD = torch.tensor(cD, device=x.device)
        # x = torch.cat((cA, cH, cV, cD), dim=1)
        # x = torch.tensor(x_wavelet, device=x.device)
        # target_size = cA.shape[2:] 
        # cH = F.interpolate(cH, size=target_size, mode='bilinear', align_corners=False)

        # x = self.temporal_conv(x)
        # x = self.bn1(x)
        # #residual = self.residual_conv(x) 
        # x_temp = x.clone() 
        num = x.size(2)
        
        for i in range(num):
            x_temp = x[:, :, i, :]
            x_temp = x_temp.permute(0, 2, 1)
            x_temp,_ = self.bi_lstm(x_temp)
            x_temp = x_temp.permute(0, 2, 1)
            x_temp = x_temp.unsqueeze(2)
            # print(x_temp.size())
            if i==0:
                x1 = x_temp
            else:
                x1 = torch.cat((x1, x_temp), dim=2)

        # print(x1.size())
        x = self.spatial_conv(x1)

        x = self.attention1(x)
        x = self.pool1(x)
        x = self.bn2(x)
  
        x = self.conv2(x)
        x = self.attention2(x)
        x = self.pool2(x)
        x = self.bn3(x)

        # x = self.conv3(x)
        # x = self.attention3(x)
        # x = self.pool3(x)
        # x = self.bn4(x)



        # x1 = self.temporal_conv(cH)
        # x1 = self.bn1(x1)
        # #residual = self.residual_conv(x) 
        # x_temp2 = x1.clone() 
        # x1 = self.spatial_conv(x1)
        # x1= self.bn2(x1)
        # x1 = self.attention1(x1)
        # x1 = self.pool1(x1)
        # x1 = self.conv2(x1)
        # x1 = self.bn3(x1)
        # x1 = self.attention2(x1)
        # x1 = self.pool2(x1)

        # x = torch.cat((x, x1), dim=1)
        x = x ** 2

        # x = self.attention(x)

        # x = self.mean_pool(x)
        # x = x.reshape(x.size(0), -1)
        x = torch.log(x + 1e-6)
        x = x.view(x.size(0), -1) 


        # x, _ = self.lstm(x)  


        # x_final = x[:, -1, :] 
        x = self.dropout(x)
        x_final = self.fc(x)


        # if self.training:
        #     # x_temp = torch.cat((x_temp1, x_temp2), dim=1)
        #     x_temp = self.mean_pool2(x_temp)
        #     x_temp = x_temp.reshape(x_temp.size(0), -1)  
        #     x_temp = self.deep_supervision(x_temp)
        #     return (x_final+x_temp)/2
        # else:
        return x_final



class Shallow_cwt_with_attentionc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convp = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 1), stride=(1, 1))
        self.temporal_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 25), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(19, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))


        self.fc = nn.Linear(18432, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(25080, 256)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention1 = CBAM(in_channels=40)
        self.attention2 = CBAM(in_channels=64)
        self.attention3 = CBAM(in_channels=128)

       # self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        x = self.convp(x)
        coeffs = pywt.wavedec2(x.squeeze().detach().cpu().numpy(), wavelet='haar', level=1)
        # cA,cH = torch.fft.fft(x, dim=1)
        cA, (cH, cV, cD) = coeffs
        x = self.temporal_conv(cA)
        x = self.bn1(x)
        #residual = self.residual_conv(x) 
        # x_temp = x.clone() 
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.attention1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.attention2(x)
        x = self.pool2(x)


        x = self.temporal_conv(cH)
        x1 = self.bn1(x)
        #residual = self.residual_conv(x) 
        x_temp = x.clone() 
        x1 = self.spatial_conv(x)
        x1= self.bn2(x1)
        x1 = self.attention1(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn3(x1)
        x1 = self.attention2(x1)
        x1 = self.pool2(x1)

        x = torch.cat((x, x1), dim=1)
        x = torch.tensor(x, device=x.device)
        x = x ** 2

        # x = self.attention(x)

        # x = self.mean_pool(x)
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



class Shallow_wt_with_attentionc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(19, 3, 3), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.spatial_conv = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        # self.bn4 = nn.BatchNorm2d(128)
        # self.pool3 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))


        self.fc = nn.Linear(32256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.deep_supervision = nn.Linear(15840, 256)
        self.mean_pool2 = nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 17))
        self.attention1 = CBAM3(in_channels=32)
        self.attention2 = CBAM3(in_channels=64)
        self.attention3 = CBAM3(in_channels=128)
        # self.lstm = nn.LSTM(input_size=252, hidden_size=256, num_layers=2, batch_first=True)

        self.bi_lstm = nn.LSTM(input_size=156, 
                               hidden_size=256, 
                               num_layers=2, 
                               dropout=0.5, 
                               bidirectional=True, 
                               batch_first=True)
       # self.residual_conv = nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))
        self.fb = nn.Linear(512, 256)

    def forward(self, x):
        # print(x.shape)
        # batch_size = x.size(0)
    
        # cwt_output = []

        # for i in range(batch_size):
        #     sample_cwt = []
        #     for j in range(19):
        #         channel_data = input_tensor[i, 0, j, :].numpy()
        

        #         cwt_magnitude = compute_cwt(channel_data)
        
        #         sample_cwt.append(cwt_magnitude)
    

        #         sample_cwt = np.stack(sample_cwt, axis=0) 
        #         cwt_output.append(sample_cwt)

        # x = torch.tensor(np.stack(cwt_output, axis=0)).unsqueeze(1) 

        x = compute_stft(x)

        x = self.temporal_conv(x)
        
        x = self.attention1(x)

        x = self.pool1(x)
        x = self.bn1(x)
        #residual = self.residual_conv(x) 
        x_temp = x.clone() 

        x = self.spatial_conv(x)


        x = self.attention2(x)
        x = self.pool2(x)
        x = self.bn2(x)
   
        x = self.conv2(x)
        
        x = self.attention3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        # x = torch.tensor(x, device=x.device)
        # x = x ** 2

        # x = self.attention(x)

        # x = self.mean_pool(x)
        # x = x.reshape(x.size(0), -1)
        # x = torch.log(x + 1e-6)
        x = x.view(x.size(0), x.size(1), -1) 

        x, _ = self.bi_lstm(x)  


        x_final = x[:, -1, :] 
        # x_final = self.dropout(x_final)
        x_final = self.fb(x_final)
        # x_final = self.fc(x)


        # if self.training:
        #     # x_temp = self.mean_pool2(x_temp)
        #     x_temp = x_temp.reshape(x_temp.size(0), -1)  
        #     x_temp = self.deep_supervision(x_temp)
        #     return (x_final+x_temp)/2
        # else:
        return x_final






def compute_cwt(x, wavelet='cmor', fs=100):

    low_freq = 0.5  
    high_freq = 40  
    

    low_scale = pywt.central_frequency(wavelet) / (low_freq * (1 / fs))
    high_scale = pywt.central_frequency(wavelet) / (high_freq * (1 / fs))
    
    scales = np.arange(int(high_scale), int(low_scale) + 1)
    
    cwt_result, _ = pywt.cwt(x, scales, wavelet)
    cwt_magnitude = np.abs(cwt_result)
    return cwt_magnitude





def compute_stft(input_tensor, n_fft=400, hop_length=120, win_length=120):

    batch_size, _, num_channels, time_steps = input_tensor.shape
    stft_results = []


    freq_resolution = 100 / n_fft

    window = torch.hamming_window(win_length, periodic=True, dtype=input_tensor.dtype, device=input_tensor.device)
    min_freq_bin = int(0.5 / freq_resolution)
    max_freq_bin = int(40 / freq_resolution) + 1 
    # print(min_freq_bin, max_freq_bin)
    for i in range(num_channels):
        channel_data = input_tensor[:, 0, i, :]  
        
        stft_channel = torch.stft(channel_data, n_fft=n_fft, hop_length=hop_length, 
                                  win_length=win_length, window=window, return_complex=True)

        stft_magnitude = torch.abs(stft_channel) 
        stft_filtered = stft_magnitude[:, min_freq_bin:max_freq_bin, :]
        stft_results.append(stft_filtered)


    stft_output = torch.stack(stft_results, dim=2).unsqueeze(1) 
    stft_output = stft_output.permute(0, 1, 3, 4, 2)

    return stft_output







class FFT_Linearformer_GNN(nn.Module):
    def __init__(self,  input_dim, hidden_dim, output_dim, seq_length=600,num_channels=19):
        super(FFT_Linearformer_GNN, self).__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        
  
        self.linearformers = nn.ModuleList([
            LinearTransformer(input_dim, hidden_dim) for _ in range(num_channels)
        ])
        

        self.gcn = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x):
  
        batch_size, _, num_channels, time_steps = x.shape
        

        fft_output = torch.fft.rfft(x, dim=-1)   
        fft_magnitude = torch.abs(fft_output)      
        
     
        fft_magnitude = fft_magnitude.squeeze(1)  
        
   
        channel_outputs = []
        for i in range(num_channels):
    
            channel_output = self.linearformers[i](fft_magnitude[:, i, :]) 
            channel_outputs.append(channel_output)
        
 
        x = torch.stack(channel_outputs, dim=1)
        
   
        gnn_input = x.view(-1, x.size(-1))  
     
        adj_matrix = torch.ones((num_channels, num_channels))
        adj_matrix.fill_diagonal_(0) 
        edge_index, _ = dense_to_sparse(adj_matrix)  
        
  
        gnn_output = self.gcn(gnn_input, edge_index)  
        
     
        gnn_output = gnn_output.view(batch_size, num_channels, -1)
        
        return gnn_output








class FFT_Linearformer_Transformer_GNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_channels=19, sample_rate=100, n_fft=400):
        super(FFT_Linearformer_Transformer_GNN, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft

  
        freq_resolution = sample_rate / n_fft


        self.freq_bins = {
            'initial': int(0.5 / freq_resolution),
            'delta': int(4 / freq_resolution),
            'theta': int(8 / freq_resolution),
            'alpha': int(13 / freq_resolution),
            'beta': int(30 / freq_resolution),
            'gamma': int(40 / freq_resolution)
        }

        self.delta_linearformer = Linformer(dim=hidden_dim, seq_len=self.freq_bins['delta'] - self.freq_bins['initial'], depth = 2)
        self.theta_linearformer = Linformer(dim=hidden_dim, seq_len=self.freq_bins['theta'] - self.freq_bins['delta'], depth = 2)
        self.alpha_linearformer = Linformer(dim=hidden_dim, seq_len=self.freq_bins['alpha'] - self.freq_bins['theta'], depth = 2)
        self.beta_linearformer = Linformer(dim=hidden_dim, seq_len=self.freq_bins['beta'] - self.freq_bins['alpha'], depth = 2)
        self.gamma_linearformer = Linformer(dim=hidden_dim, seq_len=self.freq_bins['gamma'] - self.freq_bins['beta'], depth = 2)

 
        self.global_transformer = nn.Transformer(d_model=hidden_dim, nhead=4, num_encoder_layers=2)
        self.gcn1 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size, _, num_channels, time_steps = x.shape

  
        fft_output = torch.fft.rfft(x, dim=-1)
        fft_magnitude = torch.abs(fft_output)


        delta_out = self.delta_linearformer(fft_magnitude[:, :, self.freq_bins['initial']:self.freq_bins['delta']])
        theta_out = self.theta_linearformer(fft_magnitude[:, :, self.freq_bins['delta']:self.freq_bins['theta']])
        alpha_out = self.alpha_linearformer(fft_magnitude[:, :, self.freq_bins['theta']:self.freq_bins['alpha']])
        beta_out = self.beta_linearformer(fft_magnitude[:, :, self.freq_bins['alpha']:self.freq_bins['beta']])
        gamma_out = self.gamma_linearformer(fft_magnitude[:, :, self.freq_bins['beta']:self.freq_bins['gamma']])

        combined = torch.cat([delta_out, theta_out, alpha_out, beta_out, gamma_out], dim=-1)
        combined = combined.permute(1, 0, 2)  
        global_out = self.global_transformer(combined)
        global_out = global_out.permute(1, 0, 2)  

    
        gnn_input = global_out.view(-1, global_out.size(-1))

  
        adj_matrix = torch.ones((num_channels, num_channels))
        adj_matrix.fill_diagonal_(0)
        edge_index, _ = dense_to_sparse(adj_matrix)

  
        gnn_output = self.gcn1(gnn_input, edge_index)
        
   
        gnn_output = gnn_output.view(batch_size, num_channels, -1)
        
        return gnn_output

    def apply_linearformer(fft_magnitude, start, end, linearformer):
        segment = fft_magnitude[:, :, start:end].unsqueeze(-1)  
        segment = self.expand_dim(segment)  
        segment = segment.permute(0, 3, 1, 2)  
 
        segment = segment.reshape(batch_size * self.num_channels, 4, -1)  
        return linearformer(segment).view(batch_size, self.num_channels, -1)




class ChannelAttention3(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention3, self).__init__()
 
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        
 
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention3, self).__init__()

        self.conv = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, 1), padding=(kernel_size // 2, kernel_size // 2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  
        x = self.conv(x)
        return self.sigmoid(x) * x


class CBAM3(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super(CBAM3, self).__init__()
        self.channel_attention = ChannelAttention3(in_planes=in_channels, ratio=reduction)
        self.spatial_attention = SpatialAttention3(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x 
        x = self.spatial_attention(x) * x 
        return x






class FFT_Linearformer_GNNs(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_channels=19, sample_rate=100, n_fft=400):
        super(FFT_Linearformer_GNNs, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft

     
        freq_resolution = sample_rate / n_fft


        self.freq_bins = {
            'initial': int(0.5 / freq_resolution),
            'gamma': int(40 / freq_resolution)
        }

  
        self.linformer = Linformer(dim=hidden_dim, seq_len=self.freq_bins['gamma'] - self.freq_bins['initial'], depth=2, heads=4)
        self.expand_dim = nn.Linear(1, hidden_dim)

        self.gcn1 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(num_channels * output_dim, 128),

    def forward(self, x):
        batch_size, _, num_channels, time_steps = x.shape

        fft_output = torch.fft.rfft(x, dim=-1)
        fft_magnitude = torch.abs(fft_output) 

  
        fft_magnitude = fft_magnitude[:, :, :, self.freq_bins['initial']:self.freq_bins['gamma']]


        linformer_outs = []
        for channel in range(num_channels):
            channel_data = fft_magnitude[:, :, channel, :]  
            channel_data = channel_data.squeeze(1) 
            channel_data = channel_data.unsqueeze(-1)
            channel_data = self.expand_dim(channel_data) 
            channel_data = self.linformer(channel_data) 
            linformer_outs.append(channel_data)

    
        linformer_outs = torch.stack(linformer_outs, dim=1)


        gnn_input = linformer_outs.view(-1, self.hidden_dim)  

        adj_matrix = torch.ones((num_channels, num_channels))
        adj_matrix.fill_diagonal_(0)
        edge_index, _ = dense_to_sparse(adj_matrix)


        gnn_output = self.gcn1(gnn_input, edge_index)


        gnn_output = gnn_output.view(batch_size, num_channels, -1) 
        final = sslf.fc(gnn_output)

        return final




class BiLSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(BiLSTMNetwork, self).__init__()
        
        # 定义 Bi-LSTM 层
        self.bi_lstm = nn.LSTM(input_size=input_dim, 
                               hidden_size=hidden_dim, 
                               num_layers=num_layers, 
                               dropout=dropout, 
                               bidirectional=True, 
                               batch_first=True)
        
  
        self.dropout = nn.Dropout(dropout)
        
  
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  

    def forward(self, x):
   

   
        lstm_out, _ = self.bi_lstm(x)  
        

        lstm_out = self.dropout(lstm_out)
        

        output = self.fc(lstm_out) 

        return output




class FFT_GNN(nn.Module):
    def __init__(self,  input_dim=158,  output_dim=80, num_channels=19, sample_rate=100, n_fft=400):
        super(FFT_GNN, self).__init__()
        self.num_channels = num_channels
        # self.seq_length = seq_length
        
  
        # self.linearformers = nn.ModuleList([
        #     LinearTransformer(input_dim, hidden_dim) for _ in range(num_channels)
        # ])
        

        self.gcn1 = GATConv(158, 64, heads = 5, add_self_loops = False)

        self.gcn2 = GATConv(5*64, 64, heads = 5, add_self_loops = False)

        self.gcn3 = GATConv(5*64, 64, heads = 1, add_self_loops = False)

        freq_resolution = sample_rate / n_fft
        self.relu = nn.ReLU()


        self.freq_bins = {
            'initial': int(0.5 / freq_resolution),
            'gamma': int(40 / freq_resolution)
        }


        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(16, 1))
        with open('./data/adj_mx_3d.pkl', 'rb') as pf:
            adj_mx_data = pickle.load(pf)

        self.adj_mx = adj_mx_data[2] 
    def forward(self, x):
        device = x.device
        x=x.squeeze(1)
        batch_size, num_channels, time_steps = x.shape
        batch = torch.arange(batch_size).repeat_interleave(num_channels).to(device)

        x = torch.fft.fft(x, n=400, dim=-1)
        x = torch.abs(x)   
        x = x[:, :, self.freq_bins['initial']:self.freq_bins['gamma']]
        
     

        
   
        # channel_outputs = []
        # for i in range(num_channels):
    
        #     channel_output = self.linearformers[i](fft_magnitude[:, i, :]) 
        #     channel_outputs.append(channel_output)
        
 
        # x = torch.stack(channel_outputs, dim=1)
        
   
        # gnn_input = x.view(-1, x.size(-1))  
        adj_mx = torch.tensor(self.adj_mx, dtype=torch.float32)
        adjmx = adj_mx.to(device)
        edge_index, edge_weight = dense_to_sparse(adjmx)

        batch_edge_index = []
        batch_edge_attr = []

        for i in range(batch_size):

            shifted_edge_index = edge_index + i * 19
            batch_edge_index.append(shifted_edge_index)
            batch_edge_attr.append(edge_weight)


        edge_index = torch.cat(batch_edge_index, dim=1)
        edge_weight = torch.cat(batch_edge_attr, dim=0)
        x = x.view(-1,158)
        # print(edge_index.size())
        # print(edge_weight.size())
        # exit()

 
        gnn_output = self.gcn1(x, edge_index, edge_weight)  

        gnn_output = self.relu(gnn_output)
        gnn_output = self.gcn2(gnn_output, edge_index, edge_weight)
        
        gnn_output = self.relu(gnn_output)
        gnn_output = self.gcn3(gnn_output, edge_index, edge_weight)
        gnn_output = self.relu(gnn_output)
        # gnn_output = self.relu(gnn_output)
        # gnn_output = self.gcn2(gnn_output, edge_index)

        # x= self.att_pool(x, batch)
        x = gnn_output**2
        x = global_mean_pool(gnn_output, batch)
        x = torch.log(x + 1e-6)
        return x





class FFT_GNN1(nn.Module):
    def __init__(self,  input_dim=158,  output_dim=80, num_channels=19, sample_rate=100, n_fft=400):
        super(FFT_GNN1, self).__init__()
        self.num_channels = num_channels

        self.gcn = GCNConv(158, 16)

        self.gcn2 = GCNConv(16, 16)

        self.gcn3 = GCNConv(16,16)

        self.fc = nn.Linear(760, 256)
        self.dropout = nn.Dropout(p=0.5)

        freq_resolution = sample_rate / n_fft
        self.relu = nn.ReLU()


        self.freq_bins = {
            'initial': int(0.5 / freq_resolution),
            'gamma': int(40 / freq_resolution)
        }


        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(256, 1))
        self.mean_pool = MeanAggregation()

        with open('./data/adj_mx_3d.pkl', 'rb') as pf:
            adj_mx_data = pickle.load(pf)

        self.adj_mx = adj_mx_data[2] 
        self.feature_compressor = nn.Linear(8, 1)
    def forward(self, x):
        device = x.device
        x=x.squeeze(1)
        batch_size, num_channels, time_steps = x.shape
        batch = torch.arange(batch_size).repeat_interleave(num_channels).to(device)

        x = torch.fft.fft(x, n=400, dim=-1)
        x = torch.abs(x)   
        x = x[:, :, self.freq_bins['initial']:self.freq_bins['gamma']]
        

     
        adj_mx = torch.tensor(self.adj_mx, dtype=torch.float32)
        adjmx = adj_mx.to(device)
        
        edge_index, edge_weight = dense_to_sparse(adjmx)
        # print(edge_index)
        # print(edge_weight)
        batch_edge_index = []
        batch_edge_attr = []

        for i in range(batch_size):

            shifted_edge_index = edge_index + i * 19
            batch_edge_index.append(shifted_edge_index)
            batch_edge_attr.append(edge_weight)


        edge_index = torch.cat(batch_edge_index, dim=1)
        edge_weight = torch.cat(batch_edge_attr, dim=0)
        x = x.view(-1,158)
        # print(edge_index.size())
        # print(edge_weight.size())
        # exit()

 
        gnn_output = self.gcn(x, edge_index, edge_weight)  

        gnn_output = self.relu(gnn_output)
        gnn_output = self.gcn2(gnn_output, edge_index, edge_weight)
        
        gnn_output = self.relu(gnn_output)
        gnn_output = self.gcn3(gnn_output, edge_index, edge_weight)
        gnn_output = self.relu(gnn_output)
        # gnn_output = self.relu(gnn_output)
        # gnn_output = self.gcn2(gnn_output, edge_index)

        # x= self.att_pool(x, batch)
        x = gnn_output**2
        x = global_mean_pool(gnn_output, batch)
        x = torch.log(x + 1e-6)
        # x = self.dropout(x)
        return x


class FFT_GNN3(nn.Module):
    def __init__(self,  input_dim=158,  output_dim=80, num_channels=19, sample_rate=100, n_fft=400):
        super(FFT_GNN, self).__init__()
        self.num_channels = num_channels
        # self.seq_length = seq_length
        
  
        # self.linearformers = nn.ModuleList([
        #     LinearTransformer(input_dim, hidden_dim) for _ in range(num_channels)
        # ])
        

        self.gcn = GCNConv(158, 16)
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        self.gcn2 = GCNConv(16, 16)
        self.gcn3 = GCNConv(16, 16)
        self.bn2 = BatchNorm(512)
        self.fc = nn.Linear(760, 256)
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()


    def forward(self, x, edge_index, edge_weight, batch):

        # for i in range(batch_size):
        # gnn_input = x[i]  
        gnn_output = self.gcn(x, edge_index, edge_weight)  
        # gnn_output, _ = self.attn(gnn_output, gnn_output, gnn_output)
        gnn_output = self.relu(gnn_output)
        # print(gnn_output.size())
        # gnn_output = self.bn1(gnn_output)
        gnn_output = self.gcn2(gnn_output, edge_index, edge_weight)
        # gnn_output,_ = self.attn(gnn_output, gnn_output, gnn_output)
        gnn_output = self.relu(gnn_output)
        # gnn_output = self.bn1(gnn_output)
        gnn_output = self.gcn3(gnn_output, edge_index, edge_weight)
        # gnn_output,_ = self.attn(gnn_output, gnn_output, gnn_output)
        gnn_output = self.relu(gnn_output)
        # gnn_output = self.bn2(gnn_output)
        # outputs.append(gnn_output)
        # gnn_output = torch.stack(outputs, dim=0)
        # print(gnn_output.size())
        # gnn_output = self.bn1(gnn_output)
        # x = gnn_output**2
        # x= self.att_pool(x, batch)

        # x = torch.log(x + 1e-6)
        x = global_mean_pool(x, batch)
        # x = self.dropout(x)

        return x


class ContrastiveNetC(nn.Module):
    def __init__(self, emb_net, emb_size):
        super(ContrastiveNet, self).__init__()
        self.emb_net = emb_net
        self.clf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(emb_size, 1)
        )

 

    def forward(self, anchor_x, anchor_edge_index, anchor_arr, anchor_batch,
                pair_x, pair_edge_index, pair_arr, pair_batch):
        z1 = self.emb_net(anchor_x, anchor_edge_index, anchor_batch)  
   
        z2 = self.emb_net(pair_x, pair_edge_index, pair_batch)
        

        return self.clf(torch.abs(z1 - z2)).flatten()





class FFT_GNN4(nn.Module):
    def __init__(self,  input_dim=158,  output_dim=80, num_channels=19, sample_rate=100, n_fft=400):
        super(FFT_GNN, self).__init__()
        self.num_channels = num_channels

        self.gcn = GCNConv(158, 16)
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        self.gcn2 = GCNConv(16, 16)
        self.gcn3 = GCNConv(16, 16)
        self.bn2 = BatchNorm(512)
        self.fc = nn.Linear(760, 256)
        self.dropout = nn.Dropout(p=0.5)

        freq_resolution = sample_rate / n_fft
        self.relu = nn.LeakyReLU()


        self.freq_bins = {
            'initial': int(0.5 / freq_resolution),
            'gamma': int(40 / freq_resolution)
        }


        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(16, 1))
        with open('./data/adj_mx_3d.pkl', 'rb') as pf:
            adj_mx_data = pickle.load(pf)

        self.adj_mx = adj_mx_data[2] 

    def forward(self, x):
        device = x.device
        batch_size, num_nodes, num_features = x.shape


        adj_mx = torch.tensor(self.adj_mx, dtype=torch.float32, device=device)
        edge_index, edge_weight = dense_to_sparse(adj_mx)


        batch_edge_index = []
        batch_edge_weight = []

        for i in range(batch_size):
    
            shifted_edge_index = edge_index + i * num_nodes
            batch_edge_index.append(shifted_edge_index)
            batch_edge_weight.append(edge_weight)


        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        batch_edge_weight = torch.cat(batch_edge_weight, dim=0)


        x = x.view(batch_size * num_nodes, num_features)  
        gnn_output = self.gcn(x, batch_edge_index, batch_edge_weight)
        gnn_output, _ = self.attn(gnn_output, gnn_output, gnn_output)
        gnn_output = self.relu(gnn_output)

        gnn_output = self.gcn2(gnn_output, batch_edge_index, batch_edge_weight)
        gnn_output, _ = self.attn(gnn_output, gnn_output, gnn_output)
        gnn_output = self.relu(gnn_output)

        gnn_output = self.gcn3(gnn_output, batch_edge_index, batch_edge_weight)
        gnn_output, _ = self.attn(gnn_output, gnn_output, gnn_output)
        gnn_output = self.relu(gnn_output)


        batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        x = gnn_output**2
        x = self.att_pool(x, batch)

        x = torch.log(x + 1e-6)
        x = self.dropout(x)
        x = x.reshape(batch_size, -1)
        return x





