#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt


class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(
        self,
        root,
        split,
        ignore_label,
        mean_bgr,
        augment=True,
        base_size=None,
        crop_size=321,
        scales=(1.0),
        flip=True,
    ):
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.mean_bgr = np.array(mean_bgr)
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self._set_files()

        cv2.setNumThreads(0)

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def _augmentation(self, image, label):
        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        if self.augment:
            image, label = self._augmentation(image, label)
        # Mean subtraction
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
    
    
    def show_pair(self, image_id, image, label, batch_nr = 0, batch=True):
        """
            Added function to plot image, label and overlay of these two.
            Needs also the image_id to show in caption+
            batch_nr default, takes the first image in the batch
        """
        if batch:
            image = image[0].permute(1,2,0)
            label = label[0]
            image_id = image_id[0]
        fig, ax =  plt.subplots(1,3)
        #fig.suptitle(image_id)
        ax[0].imshow(image/255)
        ax[0].axis('off')
        ax[0].set_title("image " +image_id)
        ax[1].imshow(label)
        ax[1].axis('off')
        ax[1].set_title("label " +image_id)
        ax[2].imshow(image/255, interpolation='none')
        ax[2].imshow(label, alpha = 0.4, interpolation='none')
        ax[2].axis('off')
        ax[2].set_title("overlay " +image_id)
