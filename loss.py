import torch
import torch.nn.functional as F

class RelativePositioningLoss(torch.nn.Module):
    def __init__(self, emb_size, w0=0.0):
        super(RelativePositioningLoss, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(emb_size))  # 初始化 w 参数
        self.w0 = w0  # 偏置项

    def forward(self, x1, x2, y):
    
        h_x1 = model(x1)  
        h_x2 = model(x2)  

        g_RP = torch.abs(h_x1 - h_x2)

        score = torch.dot(self.w, g_RP.T) + self.w0
        loss = torch.log(1 + torch.exp(-y * score))

        return loss.mean()