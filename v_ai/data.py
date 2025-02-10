# v_ai/data.py

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, data_list, transform=None, sequence_length=16, input_type='images'):
        """
        Args:
            data_list (list): List of samples.
                For input_type='images', each sample is a dict with keys:
                    - 'frames': list of image file paths.
                    - 'label': integer label.
                For input_type='video', each sample is a dict with keys:
                    - 'video_path': path to the video file.
                    - 'label': integer label.
            transform: Albumentations transform (or similar) to apply to each image.
            sequence_length (int): Number of frames expected per sample.
            input_type (str): Either 'images' or 'video'.
        """
        self.data_list = data_list
        self.transform = transform
        self.sequence_length = sequence_length
        assert input_type in ['images', 'video'], "input_type must be either 'images' or 'video'."
        self.input_type = input_type

    def _load_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")

        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video file {video_path} has no frames.")

        # Compute evenly spaced indices
        indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)

        frames = []
        current_frame = 0
        idx_set = set(indices.tolist())
        while cap.isOpened() and len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            current_frame += 1

        cap.release()

        # If we did not get enough frames, pad by repeating the last frame
        if len(frames) < self.sequence_length:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1].copy())
        return frames

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        frames = []

        if self.input_type == 'images':
            # Each sample should have a list of image file paths under 'frames'
            frame_paths = sample['frames']
            for path in frame_paths[:self.sequence_length]:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"Failed to load image: {path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(image)
        elif self.input_type == 'video':
            video_path = sample['video_path']
            frames = self._load_frames_from_video(video_path)

        processed_frames = []
        for frame in frames:
            if self.transform:
                augmented = self.transform(image=frame)
                image = augmented["image"]
            else:
                # If no transform is provided, convert to tensor and rearrange dimensions
                image = torch.from_numpy(frame).permute(2, 0, 1).float()
            processed_frames.append(image)

        # Stack frames into a tensor of shape [T, C, H, W]
        frames_tensor = torch.stack(processed_frames)
        label = torch.tensor(sample['label']).long()
        return frames_tensor, label

    def __len__(self):
        return len(self.data_list)
