#!/usr/bin/env python
# coding: utf-8
#
# Author: David Eike Tschirschwitz
# URL:    https://github.com/Madave94
# Date:   26 April 2020

from __future__ import absolute_import, print_function

import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class S2DS(_BaseDataset):
    """
    S2DS dataset
    """

    def __init__(self, part=2, **kwargs):
        self.part = part
        super(S2DS, self).__init__(**kwargs)

    def _set_files(self):
        # after this root includes the part number
        self.root = osp.join(self.root, "part{}".format(self.part))
        self.image_dir = osp.join(self.root, "images")
        self.label_dir = osp.join(self.root, "labels")

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = osp.join(
                self.root, "Segmentation", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        #print(image_id)
        image_path = osp.join(self.image_dir, image_id + ".jpg")
        #print(image_path)
        label_path = osp.join(self.label_dir, image_id + ".png")
        #print(label_path)
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_id, image, label