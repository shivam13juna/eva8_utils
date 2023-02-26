import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


## Train and Teset Phase transformations
def albumentation_augmentation(mean, std, config):
    
    train_transforms = A.Compose([A.HorizontalFlip(p = config['horizontalFlipProb']),
                                A.ShiftScaleRotate(shift_limit = config['shiftLimit'], scale_limit = config['scaleLimit'],
                                                   rotate_limit = config['rotateLimit'], p = config['shiftScaleRotateProb']),
                                A.CoarseDropout(max_holes = config['maxHoles'], min_holes = config['minHoles'], max_height = config['maxHeight'],
                                                max_width = config['maxWidth'], p = config['coarseDropoutProb'], 
                                                fill_value = tuple([x * 255.0 for x in mean]),
                                                min_height = config['minHeight'], min_width = config['minWidth']),
                                A.ToGray(p = config['grayscaleProb']),
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()
                              ])

    test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()])
    
    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]


# Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
def s8_albumentation_augmentation(mean, std, config):

	# pad by 4 albumentations transform
	train_transform = A.Compose([
              A.PadIfNeeded(min_height = 40, min_width = 40, always_apply=True), 
								A.RandomCrop(height = 32, width = 32, p = 1),
            
								A.HorizontalFlip(p = 0.2),
								A.Cutout(num_holes=1, max_h_size=8, max_w_size=8,  fill_value=tuple([x * 255.0 for x in mean])),
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()
								])
								
	test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
								ToTensorV2()])


	return lambda img:train_transform(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

											

# standard_lr : 0.01
# momentum_val : 0.9
# oneCycle_pct_start : 0.2
# L2_penalty : 0
# horizontalFlipProb : 0.2
# shiftScaleRotateProb : 0.25
# maxHoles : 1 
# minHoles : 1
# maxHeight : 8 
# maxWidth : 8
# minHeight : 8
# minWidth : 8
# coarseDropoutProb : 0.5
# padHeightWidth : 40
# randomCropSize : 32
# randomCropProb : 1	