import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F


def custom_collate(batch):
    collated = {}
    # Stack frames and group_label as tensors
    collated["frames"] = torch.stack([b["frames"] for b in batch])
    collated["group_label"] = torch.stack([b["group_label"] for b in batch])
    # Leave player_annots as a list of (un-collated) values.
    collated["player_annots"] = [b["player_annots"] for b in batch]
    return collated

# <TODO> consider increase the batch size. For now, batch size > 1 is problematic
# <TODO> consider using the following custom_collate function to pad frames to the same size
# def custom_collate(batch):
#     # Compute the maximum height and width among all samples' frames.
#     max_H = max(sample["frames"].shape[2] for sample in batch)
#     max_W = max(sample["frames"].shape[3] for sample in batch)
#     padded_frames = []
#     for sample in batch:
#         frames = sample["frames"]  # shape: [T, C, H, W]
#         T, C, H, W = frames.shape
#         pad_H = max_H - H
#         pad_W = max_W - W
#         # Pad on the right and bottom only.
#         # Padding format: (pad_left, pad_right, pad_top, pad_bottom) 
#         padded = F.pad(frames, (0, pad_W, 0, pad_H), mode="constant", value=0) #preserving the original coordinate system for your annotations
#         padded_frames.append(padded)
#     collated = {}
#     collated["frames"] = torch.stack(padded_frames)  # shape: [B, T, C, max_H, max_W]
#     collated["group_label"] = torch.stack([sample["group_label"] for sample in batch])
#     # Leave player_annots as list of lists.
#     collated["player_annots"] = [sample["player_annots"] for sample in batch]
#     return collated

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
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