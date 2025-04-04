# v_ai/transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import VideoMAEImageProcessor


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


# Renamed from get_3dcnn_transform
def get_3dcnn_val_transforms(image_size=112):
    """Validation/Test transforms for 3D CNN."""
    return A.Compose([
        A.Resize(image_size, image_size), # Keep resize consistent for validation
        # A.Resize(image_size + 16, image_size + 16), # Original resize before crop
        # A.CenterCrop(image_size, image_size),        # Original crop
        A.Normalize(
            mean=[0.43216, 0.394666, 0.37645],      # Kinetics mean
            std=[0.22803, 0.22145, 0.216989]        # Kinetics std
        ),
        ToTensorV2()
    ])

# Updated training transforms for 3D CNN
def get_3dcnn_train_transforms(image_size=112):
    """Training transforms with augmentation for 3D CNN."""
    # Resize to ~115% of target size before random crop
    resize_intermediate = int(image_size * 1.15)
    return A.Compose([
        A.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), p=0.2),
        # --- Optional Augmentations (uncomment to enable) ---
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=10, p=0.2), # Use rotation with caution
        # ----------------------------------------------------
        A.Normalize(
            mean=[0.43216, 0.394666, 0.37645],      # Kinetics mean
            std=[0.22803, 0.22145, 0.216989]        # Kinetics std
        ),
        ToTensorV2()
    ])


def resize_only(image_size=640):
    # This might be used elsewhere, keep it for now
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.43216, 0.394666, 0.37645],      # Kinetics mean
            std=[0.22803, 0.22145, 0.216989]        # Kinetics std
        ),
        ToTensorV2()
    ])


class VideoMAETransform:
    def __init__(self, model_name="MCG-NJU/videomae-base", image_size=224):
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name, size = {'shortest_edge':image_size}, crop_size = {'height': image_size, 'width': image_size})
        self.image_size = image_size

    def __call__(self, frames):
        """
        Args:
            frames (list): A list of frames (as numpy arrays or PIL Images).

        Returns:
            torch.FloatTensor: Processed pixel values with shape [num_frames, num_channels, height, width].
        """
        # The processor handles resizing, center cropping, and normalization.
        inputs = self.processor(frames, return_tensors="pt")
        # Squeeze the batch dimension (since inputs.pixel_values has shape [1, num_frames, C, H, W])
        return inputs.pixel_values.squeeze(0)

def get_videomae_transforms(model_name="MCG-NJU/videomae-base", image_size=224):
    """
    Returns an instance of VideoMAETransform that can be used as a transformation function
    in your DataLoader.
    """
    return VideoMAETransform(model_name, image_size)

# def get_videomae_transforms(model_name, image_size=224):
#     processor = VideoMAEImageProcessor.from_pretrained(model_name)
#     # Use the processor's normalization parameters if available; otherwise, use common defaults.
#     # mean = processor.image_mean if hasattr(processor, "image_mean") else [0.5, 0.5, 0.5]
#     # std = processor.image_std if hasattr(processor, "image_std") else [0.5, 0.5, 0.5]
#     # return A.Compose([
#     #     A.Resize(image_size, image_size),
#     #     A.Normalize(mean=mean, std=std),
#     #     ToTensorV2()
#     # ])