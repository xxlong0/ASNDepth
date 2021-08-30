# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import cv2
import yaml
from easydict import EasyDict as edict
from util import mkdir_if_missing
import pdb


def parse_task_dictionary(task_dictionary):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}
    task_cfg.FLAGVALS = {'image': cv2.INTER_LINEAR}  # cv2.INTER_CUBIC}
    task_cfg.INFER_FLAGVALS = {}

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'

        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3
        task_cfg.FLAGVALS[tmp] = cv2.INTER_CUBIC
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        # Depth
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_LINEAR
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['depthloss'] = 'l1'

    return task_cfg, other_args


def create_config(exp_file):
    # Read the files

    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    # Parse the task dictionary separately
    cfg.TASKS, extra_args = parse_task_dictionary(cfg['task_dictionary'])

    for k, v in extra_args.items():
        cfg[k] = v

    cfg.ALL_TASKS = edict()  # All tasks = Main tasks
    cfg.ALL_TASKS.NAMES = []
    cfg.ALL_TASKS.NUM_OUTPUT = {}
    cfg.ALL_TASKS.FLAGVALS = {'image': cv2.INTER_LINEAR}  # cv2.INTER_CUBIC} if image use cubic will cause artifacts
    cfg.ALL_TASKS.INFER_FLAGVALS = {}

    for k in cfg.TASKS.NAMES:
        cfg.ALL_TASKS.NAMES.append(k)
        cfg.ALL_TASKS.NUM_OUTPUT[k] = cfg.TASKS.NUM_OUTPUT[k]
        cfg.ALL_TASKS.FLAGVALS[k] = cfg.TASKS.FLAGVALS[k]
        cfg.ALL_TASKS.INFER_FLAGVALS[k] = cfg.TASKS.INFER_FLAGVALS[k]

    # Parse auxiliary dictionary separately
    if 'auxilary_task_dictionary' in cfg.keys():
        cfg.AUXILARY_TASKS, extra_args = parse_task_dictionary(cfg['auxilary_task_dictionary'])
        for k, v in extra_args.items():
            cfg[k] = v

        for k in cfg.AUXILARY_TASKS.NAMES:  # Add auxilary tasks to all tasks
            if not k in cfg.ALL_TASKS.NAMES:
                cfg.ALL_TASKS.NAMES.append(k)
                cfg.ALL_TASKS.NUM_OUTPUT[k] = cfg.AUXILARY_TASKS.NUM_OUTPUT[k]
                cfg.ALL_TASKS.FLAGVALS[k] = cfg.AUXILARY_TASKS.FLAGVALS[k]
                cfg.ALL_TASKS.INFER_FLAGVALS[k] = cfg.AUXILARY_TASKS.INFER_FLAGVALS[k]

    return cfg
