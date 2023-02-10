from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import numpy as np

import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np


def plot_gradcam_images(model, target_layers, inp_images, targets, cifar10_labels_dict):

    targets = [ClassifierOutputTarget(i) for i in targets]
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform= T.Compose([
    # T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    T.Normalize(
        mean = (-1 * np.array(mean) / np.array(std)).tolist(),
        std = (1 / np.array(std)).tolist()
    ),
    T.Lambda(lambda x: x.permute(0, 2, 3, 1))])



    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    mod_inp_images = torch.stack([i[0] for i in inp_images])
    grayscale_cams = cam(input_tensor = mod_inp_images, targets = targets, aug_smooth = True)
    rgb_imgs = inv_transform(mod_inp_images).cpu().squeeze().detach().numpy()
    fig, axs = plt.subplots(5, 2, figsize=(10, 10))

    # Increase vertical space between subplots
    fig.subplots_adjust(hspace=0.5)
    # fig.tight_layout()
    for i in range(5):
        for j in range(2):
            visualization = show_cam_on_image(rgb_imgs[2*i+j], grayscale_cams[2*i+j, :], use_rgb=True)
            axs[i, j].imshow(visualization)
            axs[i, j].set_title("Incorrect label: " + cifar10_labels_dict[inp_images[2*i+j][1].item()] + " Correct label: " + cifar10_labels_dict[inp_images[2*i+j][2].item()])



