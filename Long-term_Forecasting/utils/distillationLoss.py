import torch
import torch.nn.functional as F
import torch.nn as nn
from .ditill_utils import *
from copy import deepcopy

class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, pred, true):
        return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))




class DistillationLoss(nn.Module):
    def __init__(self, args):
        super(DistillationLoss,self).__init__()
        self.args = args
        self.loss_weight={'logits_w': 1,'feature_w':0.01}
        self.feature_loss = nn.SmoothL1Loss() if args.smooth else nn.L1Loss()
        self.logits_loss = nn.SmoothL1Loss() if args.smooth else nn.L1Loss()
        self.time_mse_loss = nn.SmoothL1Loss() if args.smooth else nn.L1Loss()
        # if self.args.loss_func == 'mse':
        #     self.time_mse_loss = nn.MSELoss()
        # elif self.args.loss_func == 'smape':
        #     self.time_mse_loss = SMAPE()


    def forward(self, outputs, batch_y):
        """
        outputs_time: 隐藏层特征经过残差连接+任务head之后的结果
        intermidiate_feat_time: 大小为num_blk+1, 包含最初的输入特征，最后一个元素是没有经过残差和head的特征。
        """
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text \
            = outputs['outputs_text'], outputs['outputs_time'], outputs['intermidiate_time'],outputs['intermidiate_text']
        #1-----------------中间特征损失
        feature_loss = sum([(0.8**idx)*self.feature_loss(feat_time, feat_text)
                            for idx, (feat_time, feat_text)
                            in enumerate(zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1]))])

        # 2----------------输出层的教师-学生损失
        logits_loss = self.logits_loss(outputs_time, outputs_text)
        # 3----------------任务特定的标签损失
        outputs_time = outputs_time[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(logits_loss.device)
        time_mse_loss = self.time_mse_loss(outputs_time, batch_y)

        total_loss = time_mse_loss + logits_loss + 0.01 * feature_loss
        return total_loss











