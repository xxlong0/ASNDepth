#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import copy
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

"""
    Model getters 
"""


def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'hrnet_w18':
        from models.seg_hrnet import hrnet_w18
        print(p['backbone_kwargs']['pretrained'])
        backbone = hrnet_w18(p['backbone_kwargs']['pretrained'])
        backbone_channels = [18, 36, 72, 144]

    elif p['backbone'] == 'hrnet_w48':
        from models.seg_hrnet import hrnet_w48
        backbone = hrnet_w48(p['backbone_kwargs']['pretrained'])
        backbone_channels = [48, 96, 192, 384]
    elif p['backbone'] == 'hrnet_w64':
        from models.seg_hrnet import hrnet_w64
        backbone = hrnet_w64(p['backbone_kwargs']['pretrained'])
        backbone_channels = [64, 128, 256, 512]
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)

    from models.asn_depthnet import AsnDepthNet
    model = AsnDepthNet(p, backbone, backbone_channels)

    return model


"""
    Transformations, datasets and dataloaders
"""



def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms as tr

    db_name = p['train_db_name']

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    # Training transformations

    # Horizontal flips with probability of 0.5
    transforms_tr = [tr.RandomHorizontalFlip()]

    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
                                         flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])


    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor()])

    transforms_tr.extend([tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transforms_tr = transforms.Compose(transforms_tr)

    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_train_dataset(p, transforms):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))

    from data.nyud_geonet import NYUD_GeoNet
    database = NYUD_GeoNet(transform=transforms, split='train')

    return database


def get_train_dataloader(p, dataset):
    """ Return the train dataloader """
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], shuffle=True, drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate_mil)
    return trainloader


def get_val_dataset(p, transforms):
    """ Return the validation dataset """

    db_name = p['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    from data.nyud_geonet import NYUD_GeoNet
    database = NYUD_GeoNet(transform=transforms, split='val')

    return database


def get_val_dataloader(p, dataset):
    """ Return the validation dataloader """
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'])
    return testloader


""" 
    Loss functions 
"""


def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'normals':
        from losses.loss_functions import NormalsLoss
        criterion = NormalsLoss(normalize=True, size_average=True, norm=p['normloss'])

    elif task == 'depth':
        from losses.loss_functions import DepthLoss
        criterion = DepthLoss(p['depthloss'])

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'depth, or normals')

    return criterion


def get_criterion(p):
    """ Return training criterion for a given setup """

    from losses.loss_schemes import GeoDepthNetLoss

    loss_weights = p['loss_kwargs']['loss_weights']
    converted_weight = p['loss_kwargs']['converted_weight']
    scale_weightbase = p['loss_kwargs']['scale_weightbase']
    return GeoDepthNetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, None, loss_weights, converted_weight,
                           scale_weightbase)


"""
    Optimizers and schedulers
"""


def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    if p['model'] == 'cross_stitch':  # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert (p['optimizer'] == 'sgd')  # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100 * p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                    momentum=p['optimizer_kwargs']['momentum'],
                                    nesterov=p['optimizer_kwargs']['nesterov'],
                                    weight_decay=p['optimizer_kwargs']['weight_decay'])


    elif p['model'] == 'nddr_cnn':  # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert (p['optimizer'] == 'sgd')  # Adam seems to fail for nddr-cnns
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100 * p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                    momentum=p['optimizer_kwargs']['momentum'],
                                    nesterov=p['optimizer_kwargs']['nesterov'],
                                    weight_decay=p['optimizer_kwargs']['weight_decay'])


    else:  # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()

        if p['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

        elif p['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

        else:
            raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
