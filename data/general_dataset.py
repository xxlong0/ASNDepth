import os
import sys
import tarfile
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from util import mkdir_if_missing
from natsort import natsorted


class GeneralDataset(data.Dataset):
    """

    Data can also be found at:
    % Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02;
    fy_d = 5.8269103270988637e+02;
    cx_d = 3.1304475870804731e+02;
    cy_d = 2.3844389626620386e+02;

    """

    def __init__(self,
                 root,
                 transform=None,
                 refname=True
                 ):

        self.root = root  # store color images

        print(root)

        self.transform = transform
        self.retname = refname

        # the camera intrinsic is for NYUD dataset, modify if you use other dataset
        self.cam_intrinsic = np.float32(np.array([[582.64, 0, 313.04],
                                                  [0, 582.69, 238.44],
                                                  [0, 0, 1]]))

        # Original samples: contain image, depth, normal, valid_masks
        self.sample_ids = []

        self.sample_ids = natsorted(os.listdir(self.root))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.sample_ids)))

    def load_color_image(self, filepath):
        if filepath.endswith("npy"):
            image = np.load(filepath)
        elif filepath.endswith("png"):
            image = cv2.imread(filepath, -1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr2rgb
        else:
            raise Exception("not supported data format, file: " + filepath)

        return np.uint8(image)

    def get_one_sample(self, index):
        sample = {}
        _img = self.load_color_image(os.path.join(self.root, self.sample_ids[index]))
        sample['image'] = _img
        sample['original_image'] = _img

        sample['intrinsic'] = self.cam_intrinsic

        if self.retname:
            sample['meta'] = {'image': str(self.sample_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, index):
        sample = self.get_one_sample(index)
        return sample

    def __len__(self):
        return len(self.sample_ids)
