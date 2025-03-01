# v_ai/dataset_videomae.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from v_ai.data import GROUP_ACTIVITY_MAPPING

class VideoMAEDataset(Dataset):
    def __init__(self, samples_base, video_ids, window_before=5, window_after=4, transform=None, num_frames=8):
        """
        For training/validation, samples are stored in folders.
        Each video folder (samples_base/video_id) contains an annotations.txt file.
        Each line in annotations.txt is expected to have:
          {FrameID.jpg} {GroupActivity} ...
        We use the frame_id (without extension) to build the sample folder path.
        
        For testing, if the video_path is not a directory, it is treated as a video file.
        
        Args:
            samples_base (str): Base directory containing video folders.
            video_ids (list): List of video IDs (as in your config).
            window_before/window_after: Kept for compatibility (not used here).
            transform: Transform to apply to each frame.
            num_frames (int): Number of frames to sample.
        """
        self.samples_base = samples_base
        self.video_ids = [str(v) for v in video_ids]
        self.transform = transform
        self.num_frames = num_frames
        
        self.samples = []
        for vid in self.video_ids:
            video_dir = os.path.join(samples_base, vid)
            ann_path = os.path.join(video_dir, "annotations.txt")
            if not os.path.exists(ann_path):
                print(f"Warning: annotations.txt not found in {video_dir}")
                continue
            with open(ann_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                frame_id = os.path.splitext(tokens[0])[0]
                group_str = tokens[1].replace("-", "_").lower()
                group_label = GROUP_ACTIVITY_MAPPING.get(group_str, 0)
                # For training/validation, sample folder is: samples_base/vid/frame_id
                sample_path = os.path.join(video_dir, frame_id)
                self.samples.append({
                    "video_path": sample_path,
                    "label": group_label
                })

    def _load_frames_from_dir(self, dir_path):
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png'))]
        if not files:
            raise ValueError(f"No image files found in directory: {dir_path}")
        files.sort()
        total = len(files)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            path = os.path.join(dir_path, files[idx])
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    def _load_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video file {video_path} has no frames.")
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        current = 0
        indices_set = set(indices.tolist())
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current in indices_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            current += 1
        cap.release()
        if len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1].copy())
        return frames

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]
        # If the sample path is a directory, load from images; otherwise, treat it as a video file.
        if os.path.isdir(video_path):
            frames = self._load_frames_from_dir(video_path)
        else:
            frames = self._load_frames_from_video(video_path)
        if self.transform:
            proc_frames = []
            for frame in frames:
                augmented = self.transform(image=frame)
                proc_frames.append(augmented["image"])
            frames = proc_frames
        else:
            frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
        # Stack frames: [T, C, H, W] then permute to [C, T, H, W]
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        return {"frames": frames_tensor, "group_label": torch.tensor(label).long()}
