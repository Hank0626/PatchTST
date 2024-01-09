import torch
import torch.nn.functional as F
import torch.nn as nn
from .ditill_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class DistillationLoss(nn.Module):
    def __init__(self, distill_loss, logits_loss, task_loss, task_name, feature_w=0.01):
        super(DistillationLoss, self).__init__()
        self.feature_w = feature_w

        self.feature_loss = loss_dict[distill_loss]
        self.logits_loss = loss_dict[logits_loss]
        self.task_loss = loss_dict[task_loss]
        
        self.task_name = task_name

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        """
        outputs_time: 隐藏层特征经过残差连接+任务head之后的结果
        intermidiate_feat_time: 大小为num_blk+1, 包含最初的输入特征，最后一个元素是没有经过残差和head的特征。
        """
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text = (
            outputs["outputs_text"],
            outputs["outputs_time"],
            outputs["intermidiate_time"],
            outputs["intermidiate_text"],
        )
        
        # 1-----------------中间特征损失
        feature_loss = sum(
            [
                (0.8**idx) * self.feature_loss(feat_time, feat_text)
                for idx, (feat_time, feat_text) in enumerate(
                    zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
                )
            ]
        )

        # 2----------------输出层的教师-学生损失
        if self.task_name == "long_term_forecast":
            logits_loss = self.logits_loss(outputs_time, outputs_text)
        elif self.task_name == "short_term_forecast":
            logits_loss = self.logits_loss(in_sample, freq_map, outputs_time, outputs_text, batch_y_mark)

        # 3----------------任务特定的标签损失
        outputs_time = outputs_time
        batch_y = batch_y.to(logits_loss.device)
        
        if self.task_name == "long_term_forecast":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "short_term_forecast":
            task_loss = self.task_loss(in_sample, freq_map, outputs_time, batch_y, batch_y_mark)

        total_loss = task_loss + logits_loss + self.feature_w * feature_loss
        return total_loss
