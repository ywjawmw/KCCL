# --*-- conding:utf-8 --*--
# @Time : 2024/6/11 14:42
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : AutoLoss.py
# @Software : PyCharm
# @Description :
import torch.nn as nn
import torch
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
        Params：
        num: int，the number of loss
        x: multi-task loss
       Examples：
           loss1=1
           loss2=2
           awl = AutomaticWeightedLoss(2)
           loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.weight = nn.Parameter(torch.ones(num, requires_grad=True))
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    loss1 = 100000
    loss2 = 7
    awl = AutomaticWeightedLoss(2)
    loss_sum = awl(loss1, loss2)