import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def spatial_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
    s = norm_fm.transpose(1, 2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s


def channel_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 1e-8)
    s = norm_fm.bmm(norm_fm.transpose(1, 2))
    s = s.unsqueeze(1)
    return s


def batch_similarity(fm):
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0, 1))
    normalized_Q = Q / torch.norm(Q, 2, dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q


def FSP(fm1, fm2):
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)
    fsp = torch.bmm(fm1, fm2) / fm1.size(2)

    return fsp


def AT(fm):
    eps = 1e-6
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    return am


def pooled_spatial_similarity(fm, k, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    s = spatial_similarity(fm)
    return s


def gaussian_rbf(fm, k, P, gamma, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    fm = fm.view(fm.size(0), fm.size(1), -1)
    feat = F.normalize(fm, p=2, dim=1)
    sim_mat = torch.bmm(feat.transpose(1, 2), feat)

    corr_mat = torch.zeros_like(sim_mat)
    one = torch.ones_like(sim_mat)
    corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** 0 / \
                math.factorial(0) * one
    for p in range(1, P + 1):
        corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / \
                    math.factorial(p) * torch.pow(sim_mat, p)
    return corr_mat


def MMD(fm, k, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    fm = fm.view(fm.size(0), fm.size(1), -1)
    mean_fm = torch.mean(fm, dim=1)
    num = mean_fm.shape[1]
    a = mean_fm.unsqueeze(-1).repeat(1, 1, num)
    b = mean_fm.unsqueeze(1).repeat(1, num, 1)
    mmd = torch.abs(a - b)
    return mmd



def CORAL(source, target):
    if source.ndim == 4:
        # TODO 也可能会使用spatial-pool
        source = source.view(source.size(0),-1)
        target = target.view(target.size(0),-1)

    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)

    return loss




class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if source.ndim == 4:
            source = source.view(source.size(0), -1)
            target = target.view(target.size(0), -1)
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss