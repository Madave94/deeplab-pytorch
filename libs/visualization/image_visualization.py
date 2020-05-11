#!/usr/bin/env python
# coding: utf-8
#
# Author:   David Tschirschwitz
# URL:      http://kazuto1011.github.io
# Created:  2020-05-11

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image

def show_pair(image_id, image, label, mean_bgr, batch_nr = 0, batch=True):
    """
        Function to plot image, label and overlay of these two.
        Needs also the image_id to show in caption
        
        Args:
            image_id (str): name of the image
            image (numpy array): RGB or BGR image
            label (numpy array): grayscale label
            mean_bgr (Blue, Green, Red)
            batch_nr (int): if image is in batch, this image of the batch will be used
            batch (bool): true since dataloader usually returns images in batches, meaning an extra dimension is added
          
        Returns:
            image triplet with image, label and then overlay of these two
            
        --- Jupyter-Notebook visualization note ---
               Add the following lines before plotting to cover the entire notebook size:
                import matplotlib.pyplot as plt
                %matplotlib inline
                %config InlineBackend.figure_format = 'retina'
                plt.rcParams["figure.figsize"] = (20,10)       
    """
    if batch:
        image = image[batch_nr].numpy().transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) + (mean_bgr[2], mean_bgr[1], mean_bgr[0])
        label = label[batch_nr].numpy()
        print(type(image))
        print(type(label))
        image_id = image_id[batch_nr]
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