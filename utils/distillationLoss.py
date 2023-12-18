import torch
import torch.nn.functional as F
import torch.nn as nn
from .ditill_utils import MMD


class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, pred, true):
        return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))




class DistillationLoss(nn.Module):
    def __init__(self,args):
        super(DistillationLoss,self).__init__()
        self.args = args
        self.loss_weight={'distill':1.}


        if self.args.loss_func == 'mse':
            self.time_mse_loss = nn.MSELoss()
        elif self.args.loss_func == 'smape':
            self.time_mse_loss = SMAPE()


    def forward(self, outputs, batch_y):
        outputs_text, outputs_time,intermidiate_feat_time, intermidiate_feat_text \
            = outputs['outputs_text'], outputs['outputs_time'], outputs['intermidiate_time'],outputs['intermidiate_text']
        loss1 = F.l1_loss(outputs_time, outputs_text)

        outputs_time = outputs_time[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(loss1.device)
        time_mse_loss = self.time_mse_loss(outputs_time, batch_y)
        ditill_loss = F.l1_loss(outputs_time,outputs_text)

        total_loss = time_mse_loss + self.loss_weight['distill'] * ditill_loss
        return total_loss











