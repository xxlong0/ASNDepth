"""
add task variables into layers, in order to train task-specific layers
"""
import torch
import torch.nn as nn


class TaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', task='shared'):
        super(TaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias, padding_mode)
        self.task = task


class TaskBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, task='shared'):
        super(TaskBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                                              track_running_stats=track_running_stats)
        self.task = task


def task_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, task='shared'):
    """3x3 convolution with padding"""
    return TaskConv2d(in_planes, out_planes, kernel_size=3, task=task, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation)


def task_conv1x1(in_planes, out_planes, stride=1, task='shared'):
    """1x1 convolution"""
    return TaskConv2d(in_planes, out_planes, kernel_size=1, task=task, stride=stride, bias=False)


class TaskBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, task='shared'):
        super(TaskBasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = task_conv3x3(inplanes, planes, task=task, stride=stride)
        self.bn1 = TaskBatchNorm2d(planes, task=task)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = task_conv3x3(planes, planes, task=task)
        self.bn2 = TaskBatchNorm2d(planes, task=task)
        self.downsample = downsample
        self.stride = stride
        self.task = task

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TaskSABlock(nn.Module):
    """ Spatial self-attention block """

    def __init__(self, in_channels, out_channels, task='shared'):
        super(TaskSABlock, self).__init__()
        self.task = task
        self.attention = nn.Sequential(TaskConv2d(in_channels, out_channels, 3, padding=1, bias=False, task=task),
                                       nn.Sigmoid())
        self.conv = TaskConv2d(in_channels, out_channels, 3, padding=1, bias=False, task=task)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class TaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, task='shared'):
        super(TaskLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.task = task
