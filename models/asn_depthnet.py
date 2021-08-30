import torch
import torch.nn as nn
import torch.nn.functional as F
from models.task_specific_layers import TaskBasicBlock, TaskConv2d, TaskBatchNorm2d

from pac.pac import packernel2d

from termcolor import colored

from depth_normal_conversion.adaptive_depth2normal import AdaptiveDepth2normal


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class ScalePredictionModule(nn.Module):
    """ Module to make the inital task predictions """

    def __init__(self, input_channels, task_channels, task):
        super(ScalePredictionModule, self).__init__()

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.Sequential(TaskBasicBlock(channels, channels, task=task),
                                            TaskBasicBlock(channels, channels, task=task))

        else:

            downsample = nn.Sequential(
                # add feature and task estimation
                TaskConv2d(input_channels, task_channels, 1, bias=False,
                           task=task),
                TaskBatchNorm2d(task_channels, task=task))
            self.refinement = nn.Sequential(
                TaskBasicBlock(input_channels, task_channels,
                               downsample=downsample, task=task),
                TaskBasicBlock(task_channels, task_channels, task=task))

    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None:  # Concat features that were propagated from previous scale
            x = torch.cat(
                (features_curr_scale,
                 F.interpolate(features_prev_scale, scale_factor=2, mode='bilinear'),  # features
                 ), 1)

        else:
            x = features_curr_scale

        # Refinement
        out = self.refinement(x)

        return out


class DepthLayer(nn.Module):
    def __init__(self, input_channels):
        super(DepthLayer, self).__init__()

        self.pred = TaskConv2d(input_channels, 1, 1, task="depth")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pred(x)
        x = self.sigmoid(x)

        return x


class DepthNormalConversion(nn.Module):
    def __init__(self, k_size, dilation, sample_num=40):
        super(DepthNormalConversion, self).__init__()
        self.k_size = k_size
        self.dilation = dilation

        self.depth2norm = AdaptiveDepth2normal(k_size=k_size, dilation=dilation, sample_num=sample_num)

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return compute_kernel(input_for_kernel, input_mask,
                              kernel_size=self.k_size,
                              dilation=self.dilation,
                              padding=self.dilation * (self.k_size - 1) // 2)[0]

    def forward(self, init_depth, intrinsic, guidance=None, if_area=True, if_pa=True):

        if guidance is not None:
            guide_weight = self.compute_kernel(guidance)  # [B, 1, K, K, H, W]
            B, C, K1, K2, H, W = guide_weight.shape

            # smooth the kernel; otherwise, the distribution is too sharp
            ones_constant = torch.ones_like(guide_weight).type_as(guide_weight).to(guide_weight.device) / (K1 * K2)
            guide_weight = guide_weight + ones_constant
            norm = guide_weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
            guide_weight = guide_weight / norm * (K1 * K2)  # scale to larger values

            guide_weight = guide_weight.squeeze(1).permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, K, K]
            B, H, W, K, K = guide_weight.shape
            guide_weight = guide_weight.view(B, H, W, K * K)
        else:
            guide_weight = None

        estimate_normal, _ = self.depth2norm(init_depth, intrinsic, guide_weight, if_area=if_area, if_pa=if_pa)

        return estimate_normal


def task_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, task='shared'):
    """3x3 convolution with padding"""
    return TaskConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation, task=task)


def task_conv1x1(in_planes, out_planes, stride=1, task='shared'):
    """1x1 convolution"""
    return TaskConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, task=task)


def compute_kernel(input_for_kernel, input_mask=None, kernel_size=3, stride=1, padding=1, dilation=1,
                   kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=True):
    return packernel2d(input_for_kernel, input_mask,
                       kernel_size=kernel_size, stride=stride, padding=padding,
                       dilation=dilation, kernel_type=kernel_type,
                       smooth_kernel_type=smooth_kernel_type,
                       smooth_kernel=None,
                       inv_alpha=None,
                       inv_lambda=None,
                       channel_wise=False,
                       normalize_kernel=normalize_kernel,
                       transposed=False,
                       native_impl=None)


class AsnDepthNet(nn.Module):
    """
    Depth estimation Network with pixel adaptive surface normal constraint
    output 1/2 size estimated depth compared with input image
    """

    def __init__(self, p, backbone, backbone_channels, max_depth=10.):
        super(AsnDepthNet, self).__init__()
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.num_scales = len(backbone_channels)

        self.task_channels = backbone_channels

        print("task_channels: ", self.task_channels)
        self.channels = backbone_channels
        self.max_depth = p['max_depth'] if 'max_depth' in p.keys() else max_depth

        self.use_guidance = p['use_guidance'] if 'use_guidance' in p.keys() else False
        self.guidance_reduce = p['guidance_reduce'] if 'guidance_reduce' in p.keys() else False

        self.normal_loss = p['normal_loss'] if 'normal_loss' in p.keys() else False

        self.pasn_if_area = p['pasn_if_area'] if 'pasn_if_area' in p.keys() else True
        self.pasn_if_pa = p['pasn_if_pa'] if 'pasn_if_pa' in p.keys() else True

        # Backbone
        self.backbone = backbone

        # Initial task predictions at multiple scales
        ################################# Depth branch #############################################
        self.scale_0_fea_depth = ScalePredictionModule(self.channels[0] + self.task_channels[1] + 1 * 1,
                                                       self.task_channels[0], task='depth')
        self.scale_0_depth = DepthLayer(self.task_channels[0])
        self.scale_1_fea_depth = ScalePredictionModule(self.channels[1] + self.task_channels[2] + 1 * 1,
                                                       self.task_channels[1], task='depth')
        self.scale_1_depth = DepthLayer(self.task_channels[1])
        self.scale_2_fea_depth = ScalePredictionModule(self.channels[2] + self.task_channels[3] + 1 * 1,
                                                       self.task_channels[2], task='depth')
        self.scale_2_depth = DepthLayer(self.task_channels[2])
        self.scale_3_fea_depth = ScalePredictionModule(self.channels[3],
                                                       self.task_channels[3], task='depth')
        self.scale_3_depth = DepthLayer(self.task_channels[3])

        ################################# Guidance branch #############################################
        if self.use_guidance:
            self.scale_0_guidance = ScalePredictionModule(self.channels[0] + self.task_channels[1],
                                                          self.task_channels[0], task='guidance')

            if self.guidance_reduce:
                self.scale_0_guidance_reduce_dims = nn.Sequential(TaskConv2d(self.task_channels[0], 3, 1, bias=True,
                                                                             task='guidance'),
                                                                  torch.nn.Sigmoid())  # easy to visualize
            self.scale_1_guidance = ScalePredictionModule(self.channels[1] + self.task_channels[2],
                                                          self.task_channels[1], task='guidance')
            self.scale_2_guidance = ScalePredictionModule(self.channels[2] + self.task_channels[3],
                                                          self.task_channels[2], task='guidance')
            self.scale_3_guidance = ScalePredictionModule(self.channels[3],
                                                          self.task_channels[3], task='guidance')

        # Depth Normal conversion modules
        k_size = p['k_size'] if 'k_size' in p.keys() else 5
        sample_num = p['sample_num'] if 'sample_num' in p.keys() else 40

        print(colored("************************************", 'red'))
        print("pasn_if_area", self.pasn_if_area)
        print("pasn_if_pa", self.pasn_if_pa)
        print("k_size", k_size)
        print("sample_num", sample_num)
        print(colored("************************************", 'red'))
        self.scale_0_conversion = DepthNormalConversion(k_size=k_size, dilation=1,
                                                        sample_num=sample_num)  # Depth2NormalLight

    def scale_intrinsic(self, intrinsic, scale_x, scale_y):
        intrinsic = intrinsic.clone()
        intrinsic[:, 0, :] = intrinsic[:, 0, :] * scale_x
        intrinsic[:, 1, :] = intrinsic[:, 1, :] * scale_y

        # print(scale_x, scale_y)
        # print(intrinsic)
        return intrinsic

    def forward(self, x, intrinsic):
        img_size = x.size()[-2:]
        img_H, img_W = img_size[0], img_size[1]

        # upscale 2x for calculate normal
        scale_factor = 2

        out = {}

        # Backbone
        x = self.backbone(x)

        # Predictions at multiple scales
        # Scale 3
        x_3_fea_depth = self.scale_3_fea_depth(x[3])
        x_3_depth = self.scale_3_depth(x_3_fea_depth) * self.max_depth

        x_2_fea_depth = self.scale_2_fea_depth(x[2], torch.cat([x_3_fea_depth, x_3_depth], dim=1))
        x_2_depth = self.scale_2_depth(x_2_fea_depth) * self.max_depth

        x_1_fea_depth = self.scale_1_fea_depth(x[1], torch.cat([x_2_fea_depth, x_2_depth], dim=1))
        x_1_depth = self.scale_1_depth(x_1_fea_depth) * self.max_depth

        x_0_fea_depth = self.scale_0_fea_depth(
            F.interpolate(x[0], scale_factor=scale_factor, mode='bilinear'),
            F.interpolate(torch.cat([x_1_fea_depth, x_1_depth], dim=1), scale_factor=scale_factor, mode='bilinear'))
        x_0_depth = self.scale_0_depth(x_0_fea_depth) * self.max_depth

        if self.use_guidance:
            x_3_guidance = self.scale_3_guidance(x[3])
            x_2_guidance = self.scale_2_guidance(x[2], x_3_guidance)
            x_1_guidance = self.scale_1_guidance(x[1], x_2_guidance)
            x_0_guidance = self.scale_0_guidance(
                F.interpolate(x[0], scale_factor=scale_factor, mode='bilinear'),
                F.interpolate(x_1_guidance, scale_factor=scale_factor, mode='bilinear')
            )
            x_0_guidance = self.scale_0_guidance_reduce_dims(x_0_guidance)

        else:
            x_0_guidance = None


        # scale intrinsic
        _, _, scale_0_H, scale_0_W = x_0_depth.shape
        scale_0_intrinsic = self.scale_intrinsic(intrinsic, scale_x=scale_0_W / img_W, scale_y=scale_0_H / img_H)

        if self.normal_loss:
            x_0_converted_normals = self.scale_0_conversion(
                    x_0_depth, scale_0_intrinsic, x_0_guidance)

            x_0_converted_normals = F.interpolate(x_0_converted_normals, img_size, mode='bilinear')
        else:
            x_0_converted_normals = None

        out['guidance_feature'] = F.interpolate(x_0_guidance, img_size,
                                                mode='bilinear') if x_0_guidance is not None else None

        out['deep_supervision'] = {'scale_0': {'depth': x_0_depth},
                                   'scale_1': {'depth': x_1_depth},
                                   'scale_2': {'depth': x_2_depth},
                                   'scale_3': {'depth': x_3_depth}}

        out['converted_normals'] = x_0_converted_normals

        if x_0_converted_normals is not None:
            out['normals'] = x_0_converted_normals

        out['depth'] = F.interpolate(x_0_depth, img_size, mode='bilinear')

        return out
