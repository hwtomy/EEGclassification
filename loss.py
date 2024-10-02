
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositioningLossm(torch.nn.Module):
    def __init__(self, emb_size, w0=0.0):
        super(RelativePositioningLossm, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(emb_size)) 
        self.w0 = w0 

    def forward(self, output, labelss):

        labelss = labelss.view_as(output)  
        score = output + self.w0
        loss = torch.log(1 + torch.exp(-labelss * score))
        
        return loss.mean()


class SimpleCPCLoss(nn.Module):
    def __init__(self, N_p, N_b):
        super(SimpleCPCLoss, self).__init__()
        self.N_p = N_p
        self.N_b = N_b

    def forward(self, context_embeddings, future_embeddings, negative_embeddings):

        batch_size = context_embeddings.size(0)
        total_loss = 0
        
        for k in range(self.N_p):
            positive_logits = torch.sum(context_embeddings * future_embeddings[:, k, :], dim=-1)
            
            negative_logits = torch.matmul(context_embeddings.unsqueeze(1), negative_embeddings[:, k, :, :].transpose(2, 3)).squeeze(1)
            
            logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)


            labels = torch.zeros(batch_size, dtype=torch.long).to(logits.device)  
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss

        return total_loss / self.N_p

