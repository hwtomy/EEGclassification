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
    def __init__(self, encoder, num_layers, hidden_size=100, emb_size=100, Np=1, Nb=10):
        super(CPCModel, self).__init__()
        self.encoder = encoder  
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.bilinear_layers = nn.ModuleList([nn.Bilinear(hidden_size, emb_size, 1) for _ in range(Np)])
        
        self.classifier = nn.Linear(hidden_size, 2)  
    
    def forward(self, context_seq, future_seq, negative_seq):
        # Encode the context sequence
        print(f"Context seq shape: {context_seq.shape}")
        batch_size, seq_len, channels, time_steps = context_seq.size()
        context_seq = context_seq.view(batch_size * seq_len, channels, time_steps)
        z_context = self.encoder(context_seq)  
        z_context = z_context.view(batch_size, seq_len, -1)  

        # Pass through GRU
        z_context, _ = self.gru(z_context)

        # For future predictions
        future_seq = future_seq.view(batch_size * Np, channels, time_steps)
        z_future = self.encoder(future_seq)  
        z_future = z_future.view(batch_size, Np, -1)  

        # Encode negative samples
        negative_seq = negative_seq.view(batch_size * Np * Nb, channels, time_steps)
        z_negatives = self.encoder(negative_seq)  
        z_negatives = z_negatives.view(batch_size, Np, Nb, -1)  

      
        output = []
        for k, bilinear_layer in enumerate(self.bilinear_layers):
            positive_logits = bilinear_layer(z_context, z_future[:, k, :]) 
            
   
            negative_logits = torch.matmul(z_context.unsqueeze(1), z_negatives[:, k, :, :].transpose(2, 3)).squeeze(1)
            
            # Concatenate positive and negative logits for contrastive loss
            logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
            output.append(logits)

        contrastive_output = torch.cat(output, dim=1)  # Concatenate the outputs

        # Downstream binary classification task
        classification_output = self.classifier(z_context[:, -1, :]) 
        
        return contrastive_output, classification_output  



