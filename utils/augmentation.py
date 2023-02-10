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