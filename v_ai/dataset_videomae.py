# v_ai/dataset_videomae.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import VideoMAEImageProcessor


GROUP_ACTIVITY_MAPPING = {
    "r_set": 0, "r_spike": 1, "r_pass": 2, "r_winpoint": 3,
    "l_winpoint": 4, "l_pass": 5, "l_spike": 6, "l_set": 7
}

ACTION_MAPPING = {
    "waiting": 0, "setting": 1, "digging": 2, "falling": 3,
    "spiking": 4, "blocking": 5, "jumping": 6, "moving": 7, "standing": 8
}

class VideoMAE_V2_Dataset(Dataset):
    def __init__(self, samples_base, video_ids, window_before=20, window_after=20, 
                 transform=None, num_frames=8, model_name = "OpenGVLab/VideoMAEv2-Base"):
        """
        Args:
            samples_base (str): Base directory containing video folders.
            video_ids (list): List of video IDs (folder names as strings or ints).
            window_before (int): Number of frames before the target frame.
            window_after (int): Number of frames after the target frame.
            transform: Transform to apply on each loaded full–frame image.
            num_frames (int): Desired number of frames to output (e.g. 8).
        """
        self.samples_base = samples_base
        self.video_ids = [str(v) for v in video_ids]
        self.window_before = window_before
        self.window_after = window_after
        # Total frames in window from annotation (usually 20+1+20 = 41)
        self.T = window_before + 1 + window_after
        self.transform = transform
        self.num_frames = num_frames  # desired output number (e.g. 8)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
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
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                # Remove extension (if present) from frame id to match sample folder name.
                frame_id = os.path.splitext(tokens[0])[0]
                group_str = tokens[1].replace("-", "_").lower()
                group_label = GROUP_ACTIVITY_MAPPING.get(group_str, 0)
                sample = {
                    "video_id": vid,
                    "frame_id": frame_id,
                    "group_label": group_label,
                }
                self.samples.append(sample)

    def _load_frame_window(self, sample_dir, target_filename):
        """
        Reads all image files in sample_dir, finds the target frame using target_filename,
        extracts a window of self.T frames starting from (target_idx - window_before),
        and then uniformly samples self.num_frames frames from that window.
        """
        all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))]
        if not all_files:
            raise ValueError(f"No image files found in {sample_dir}")
        all_files.sort()

        try:
            target_idx = all_files.index(target_filename)
        except ValueError:
            target_idx = len(all_files) // 2

        start_idx = max(0, target_idx - self.window_before)
        end_idx = min(len(all_files), start_idx + self.T)
        selected_files = all_files[start_idx:end_idx]

        frames = []
        for fname in selected_files:
            fpath = os.path.join(sample_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                raise ValueError(f"Failed to load image: {fpath}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(image=img)["image"].float()  # [C, H, W]
                img = self.transform(img).squeeze(0).float()  # [C, H, W]
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)
        #padding
        if len(frames) < self.T:
            last_frame = frames[-1]
            padding = [last_frame.clone() for _ in range(self.T - len(frames))]
            frames.extend(padding)
        frames = torch.stack(frames)  # [T, C, H, W]

        # Uniformly sample self.num_frames frames if needed.
        if frames.shape[0] > self.num_frames:
            indices = torch.linspace(0, frames.shape[0] - 1, steps=self.num_frames).long()
            frames = frames[indices]
        return frames


    def _load_frames_from_video(self, video_path):
        """
        Loads self.num_frames frames uniformly from a video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video file {video_path} has no frames.")
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        current_frame = 0
        indices_set = set(indices.tolist())
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in indices_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(image=frame)["image"].float()
                else:
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
            current_frame += 1
        cap.release()
        if len(frames) < self.num_frames:
            # Pad by repeating the last frame.
            while len(frames) < self.num_frames:
                frames.append(frames[-1].clone())
        frames = torch.stack(frames)  # [num_frames, C, H, W]
        return frames

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        frame_id = sample["frame_id"]
        possible_names = [frame_id + ext for ext in [".jpg", ".png"]]
        # First assume the sample is stored as a directory: samples_base/video_id/frame_id
        sample_dir = os.path.join(self.samples_base, video_id, frame_id)
        if os.path.isdir(sample_dir):
            all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))]
            all_files.sort()
            target_filename = next((name for name in possible_names if name in all_files), all_files[len(all_files)//2])
            frames = self._load_frame_window(sample_dir, target_filename)  # [num_frames, C, H, W]
        else:
            # If the expected folder doesn't exist, try using the video file.
            video_path = os.path.join(self.samples_base, video_id)
            if os.path.isfile(video_path):
                frames = self._load_frames_from_video(video_path)
            else:
                raise ValueError(f"Cannot find directory or video file for video {video_id} with frame {frame_id}")
        return {
            "frames": frames,  # shape: [num_frames, C, H, W]
            "group_label": torch.tensor(sample["group_label"]).long()
        }
