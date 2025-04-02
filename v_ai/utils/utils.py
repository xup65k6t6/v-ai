import os
import numpy as np
import torch
from PIL import Image


def custom_collate(batch):
    """
    Custom collate function to handle variable number of players and person crops.
    """
    # Stack frames and group labels as tensors
    frames = torch.stack([sample["frames"] for sample in batch])  # [B, T, C, H, W]
    group_labels = torch.stack([sample["group_label"] for sample in batch])  # [B]

    # Handle player_annots (optional, can remove if not needed)
    max_players = max(len(sample["player_annots"]) for sample in batch)
    padded_player_annots = []
    for sample in batch:
        player_annots = sample["player_annots"]
        num_players = len(player_annots)
        padded_annots = player_annots + [{"action": 0, "bbox": (0,0,0,0)}] * (max_players - num_players)
        padded_player_annots.append(padded_annots)

    # Handle person_crops: Keep as a list of lists of tensors
    person_crops = [sample["person_crops"] for sample in batch]  # List of length B, each a list of [T, C, 224, 224]

    return {
        "frames": frames,
        "person_crops": person_crops,  # List of lists, no stacking due to variable num_players
        "player_annots": padded_player_annots,  # Optional
        "group_label": group_labels
    }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_checkpoint_dir():
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def display_temporal_tensor_frames(tensor_frames):
    """
    Converts each frame in a tensor to an image and displays it.

    Parameters:
    tensor_frames (torch.Tensor): A tensor of shape [T, C, H, W]
    """
    for i in range(tensor_frames.shape[0]):
        single_frame = tensor_frames[i].numpy()
        single_frame = np.transpose(single_frame, (1, 2, 0))
        image = Image.fromarray((single_frame * 255).astype(np.uint8))
        image.show()

def display_single_frame(frame, unnormalize=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Converts a single frame tensor to an image and displays it.
    
    Parameters:
    frame (torch.Tensor): A tensor of shape [C, H, W]
    unnormalize (bool): Whether to unnormalize the image (default: False)
    mean (tuple): Mean values used for normalization (default: (0.485, 0.456, 0.406))
    std (tuple): Standard deviation values used for normalization (default: (0.229, 0.224, 0.225))
    """
    # Convert tensor to numpy array
    img_array = frame.permute(1, 2, 0).numpy()

    if unnormalize:
        img_array = img_array * np.array(std) + np.array(mean)  # Unnormalize
        img_array = np.clip(img_array, 0, 1)  # Clip to valid range (0,1)

    # Convert to uint8
    img_array = (img_array * 255).astype('uint8')

    # Convert to image
    Image.fromarray(img_array).show()
    return img_array
