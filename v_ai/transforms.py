# v_ai/transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_3dcnn_transform(image_size=112):
    return A.Compose([
        A.Resize(image_size + 16, image_size + 16),  # e.g., 128 for image_size=112
        A.CenterCrop(image_size, image_size),        # e.g., 112x112
        A.Normalize(
            mean=[0.43216, 0.394666, 0.37645],      # Kinetics mean
            std=[0.22803, 0.22145, 0.216989]        # Kinetics std
        ),
        ToTensorV2()
    ])
    

def resize_only(image_size=640):
    return A.Compose([
        A.Resize(image_size, image_size),
        # A.Normalize(
        #     mean=[0.43216, 0.394666, 0.37645],      # Kinetics mean
        #     std=[0.22803, 0.22145, 0.216989]        # Kinetics std
        # ),
        ToTensorV2()
    ])