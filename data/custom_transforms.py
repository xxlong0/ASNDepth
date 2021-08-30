# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import numpy.random as random
import numpy as np
import torch
import cv2
import math
import torchvision
from torchvision import transforms
from copy import deepcopy


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem or 'intrinsic' in elem:
                    continue
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

                if elem == 'normals':
                    sample[elem][:, :, 0] *= -1

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class NormalizeImage(object):
    """
    Return the given elements between 0 and 1
    """

    def __init__(self, norm_elem='image', clip=False):
        self.norm_elem = norm_elem
        self.clip = clip

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                if np.max(sample[elem]) > 1:
                    sample[elem] /= 255.0
        else:
            if self.clip:
                sample[self.norm_elem] = np.clip(sample[self.norm_elem], 0, 255)
            if np.max(sample[self.norm_elem]) > 1:
                sample[self.norm_elem] /= 255.0
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """

    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class AddIgnoreRegions(object):
    """Add Ignore Regions"""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem]

            if elem == 'normals':
                # Check areas with norm 0
                Nn = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2 + tmp[:, :, 2] ** 2)

                tmp[Nn == 0, :] = 255.
                sample[elem] = tmp

            elif elem == 'depth':
                tmp[tmp == 0] = 255.
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'AddIgnoreRegions'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif elem == 'intrinsic':
                sample[elem] = torch.from_numpy(sample[elem].astype(np.float32))
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            if 'image' in elem:
                # here self.to_tensor() convert image range (0,255) to (0,1)
                sample[elem] = self.to_tensor(
                    tmp.astype(np.uint8))  # Between 0 .. 255 so cast as uint8 to ensure compatible w/ imagenet weight

            else:
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.from_numpy(tmp.astype(np.float32))

        return sample

    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = torchvision.transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        
#         if 'original_image' in sample.keys():
#             sample['original_image'] = self.normalize(sample['original_image'])
        return sample

    def __str__(self):
        return 'Normalize([%.3f,%.3f,%.3f],[%.3f,%.3f,%.3f])' % (
            self.mean[0], self.mean[1], self.mean[2], self.std[0], self.std[1], self.std[2])
