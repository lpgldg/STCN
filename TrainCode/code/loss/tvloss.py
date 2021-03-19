import torch
import torch.nn as nn
from torch.autograd import Variable

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x,y):
        
        x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
        y = (y-torch.min(y))/(torch.max(y)-torch.min(y))


        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        h_y = y.size()[2]
        w_y = y.size()[3]

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        count_h_y = self._tensor_size(y[:,:,1:,:])
        count_w_y = self._tensor_size(y[:,:,:,1:])

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        h_tv_y = torch.pow((y[:,:,1:,:]-y[:,:,:h_y-1,:]),2).sum()
        w_tv_y = torch.pow((y[:,:,:,1:]-y[:,:,:,:w_y-1]),2).sum()


        return self.TVLoss_weight*2*torch.abs(((h_tv/count_h+w_tv/count_w)/batch_size-(h_tv_y/count_h_y+w_tv_y/count_w_y)/batch_size))

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
