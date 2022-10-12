# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/19 21:33
@Auth ： yusuen
@File ：StyleNormal.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn

class StyleNormal(nn.Module):
    def __init__(self, channel, eps=1e-5, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(32, channel, 1, 1)) # batchsize or batchsize/num of GPU if multi GPU
        self.register_buffer('running_var', torch.ones(32, channel, 1, 1))

    def forward(self, input_):

        if self.training:

            size = input_.size()
            assert (len(size) == 4)
            N, C, H, W = size

            # instance
            var = input_.view(N, C, -1).var(dim=2)
            std = (var + self.eps).view(N, C, 1, 1).sqrt()
            mean = input_.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
            normal = (input_ - mean.expand(size)) / std.expand(size)

            # self.running_mean = (
            #         (1 - self.momentum) * self.running_mean
            #         + self.momentum * mean.detach()
            # )
            #
            # unbias_var = (var * (H * W) / (H * W - 1)).view(N, C, 1, 1)
            # self.running_var = (
            #         (1 - self.momentum) * self.running_var
            #         + self.momentum * unbias_var.detach()
            # )

            self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean.detach()
            )

            unbias_var = (var * (H * W) / (H * W - 1)).view(N, C, 1, 1)
            self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * unbias_var.detach()
            )

            # print(torch.mean(self.running_var), torch.mean(self.running_mean))

            std_b = torch.mean(std, dim=0, keepdim=True)
            mean_b = torch.mean(mean, dim=0, keepdim=True)

            output = normal * std_b + mean_b

            return output

        else:
            size = input_.size()
            assert (len(size) == 4)
            N, C, H, W = size
            var_ = input_.view(N, C, -1).var(dim=2)
            std_ = (var_ + self.eps).view(N, C, 1, 1).sqrt()
            mean_ = input_.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
            normal = (input_ - mean_.expand(size)) / std_.expand(size)
            std_b = torch.mean((self.running_var + self.eps).sqrt(), dim=0, keepdim=True)
            mean_b = torch.mean(self.running_mean, dim=0, keepdim=True)
            output = normal * std_b + mean_b

            return output
