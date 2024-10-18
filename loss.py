
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



class RelativePositioningLoss(torch.nn.Module):
    def __init__(self, emb_size, w0=0.0):
        super(RelativePositioningLoss, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(emb_size))  
        self.w0 = w0  

    def forward(self, output, y):
        score = output + self.w0
        loss = torch.log(1 + torch.exp(-y * score))
        return loss.mean()




class RelativePositioningLoss_deep(torch.nn.Module):
    def __init__(self, emb_size, w0=0.0, lambda_deep=0.5):
        super(RelativePositioningLoss_deep, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(emb_size))  
        self.w_deep = torch.nn.Parameter(torch.randn(emb_size))  
        self.w0 = w0  
        self.lambda_deep = lambda_deep  

    def forward(self, x1, x2, y):
        h_x1_main, h_x1_deep = model(x1)
        h_x2_main, h_x2_deep = model(x2)

        g_RP_main = torch.abs(h_x1_main - h_x2_main)
        score_main = torch.dot(self.w, g_RP_main.T) + self.w0


        g_RP_deep = torch.abs(h_x1_deep - h_x2_deep)
        score_deep = torch.dot(self.w_deep, g_RP_deep.T) + self.w0
        loss_main = torch.log(1 + torch.exp(-y * score_main))

        loss_deep = torch.log(1 + torch.exp(-y * score_deep))

        total_loss = loss_main.mean() + self.lambda_deep * loss_deep.mean()

        return total_loss

