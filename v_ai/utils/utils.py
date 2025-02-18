import torch


def custom_collate(batch):
    collated = {}
    # Stack frames and group_label as tensors
    collated["frames"] = torch.stack([b["frames"] for b in batch])
    collated["group_label"] = torch.stack([b["group_label"] for b in batch])
    # Leave player_annots as a list of (un-collated) values.
    collated["player_annots"] = [b["player_annots"] for b in batch]
    return collated

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")