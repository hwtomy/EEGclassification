import torch

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
