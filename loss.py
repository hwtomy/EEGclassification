import torch

class RelativePositioningLoss(torch.nn.Module):
    def __init__(self, emb_size, w0=0.0):
        super(RelativePositioningLoss, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(emb_size)) 
        self.w0 = w0 

    def forward(self, output, y):

        y = y.view_as(output)  
        score = output + self.w0
        loss = torch.log(1 + torch.exp(-y * score))
        
        return loss.mean()
