# v_ai/data.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

# Define mappings for group activity and individual action classes.
GROUP_ACTIVITY_MAPPING = {
    "r_set": 0, "r_spike": 1, "r_pass": 2, "r_winpoint": 3,
    "l_winpoint": 4, "l_pass": 5, "l_spike": 6, "l_set": 7
}

ACTION_MAPPING = {
    "waiting": 0, "setting": 1, "digging": 2, "falling": 3,
    "spiking": 4, "blocking": 5, "jumping": 6, "moving": 7, "standing": 8
}


class GroupActivityDataset(Dataset):
    def __init__(self, samples_base, video_ids, window_before=5, window_after=4, transform=None, img_size = 1280):
        """
        Args:
            samples_base (str): Path to the "data/samples" directory.
            video_ids (list): List of video IDs (folder names as strings or ints) to include.
            window_before (int): Number of frames to take before the target frame.
            window_after (int): Number of frames to take after the target frame.
                             The total temporal window will be (window_before + 1 + window_after).
            transform: Optional transform to apply to each loaded full–frame image.
                       (Note: cropping for person–regions is done later in the model.)
        """
        self.samples_base = samples_base
        self.video_ids = [str(v) for v in video_ids]  # ensure strings
        self.window_before = window_before
        self.window_after = window_after
        self.T = window_before + 1 + window_after  # total frames in window
        self.transform = transform
        self.img_size = img_size

        # Build a list of sample dictionaries.
        # For each video directory, open its annotations.txt file and parse each line.
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
                # Expected format: {FrameID.jpg} {GroupActivity} {PlayerAnn1} {PlayerAnn2} ...
                # Each player annotation consists of 5 tokens: {ActionClass} X Y W H
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                # Remove extension (if present) from frame id to match sample folder name.
                frame_id = os.path.splitext(tokens[0])[0]
                group_str = tokens[1].replace("-", "_").lower()
                group_label = GROUP_ACTIVITY_MAPPING.get(group_str, 0)  # default to 0 if not found
                # Parse player annotations.
                player_annots = []
                # Remaining tokens in groups of 5.
                num_players = (len(tokens) - 2) // 5
                for i in range(num_players):
                    base_idx = 2 + i * 5  # Corrected index.
                    try:
                        x = float(tokens[base_idx])
                        y = float(tokens[base_idx + 1])
                        w = float(tokens[base_idx + 2])
                        h = float(tokens[base_idx + 3])
                    except ValueError:
                        continue
                    act_str = tokens[base_idx + 4].lower().replace("-", "_")
                    action_label = ACTION_MAPPING.get(act_str, 0)
                    player_annots.append({"action": action_label, "bbox": (x, y, w, h)})
                sample = {
                    "video_id": vid,
                    "frame_id": frame_id,
                    "group_label": group_label,
                    "player_annots": player_annots
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def _load_frame_window(self, sample_dir, target_filename):
        """
        Loads a temporal window from the sample directory.
        Args:
            sample_dir (str): Directory path for this annotated sample.
            target_filename (str): The target frame image filename (with extension) as given in annotation.
        Returns:
            frames: A list of T images (as numpy arrays). Each image is read with cv2.imread and converted to RGB.
        """
        # List all image files in the sample directory.
        all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))]
        if not all_files:
            raise ValueError(f"No image files found in {sample_dir}")
        all_files.sort()  # sort by filename (assumed sequential)

        # Find the index of the target frame.
        try:
            target_idx = all_files.index(target_filename)
        except ValueError:
            # If not found, default to middle of list.
            target_idx = len(all_files) // 2

        # Determine start and end indices for the temporal window.
        start_idx = max(0, target_idx - self.window_before)
        end_idx = start_idx + self.T
        # If end exceeds, adjust start.
        if end_idx > len(all_files):
            end_idx = len(all_files)
            start_idx = max(0, end_idx - self.T)
        selected_files = all_files[start_idx:end_idx]

        frames = []
        fixed_size = (self.img_size, self.img_size)
        for fname in selected_files:
            fpath = os.path.join(sample_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                raise ValueError(f"Failed to load image: {fpath}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented["image"]
            else:
                # Convert to tensor [C, H, W] and scale to float.
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                # Resize to fixed_size using bilinear interpolation.
                # img = F.interpolate(img.unsqueeze(0), size=fixed_size, mode='bilinear', align_corners=False).squeeze(0)
            frames.append(img)
        # Stack into a tensor of shape [T, C, H, W]
        frames = torch.stack(frames)
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        frame_id = sample["frame_id"]
        # In the annotation, frame_id may not include extension – try common ones.
        possible_names = [frame_id + ext for ext in [".jpg", ".png"]]
        # The sample directory is at: samples/<video_id>/<frame_id>
        sample_dir = os.path.join(self.samples_base, video_id, frame_id)
        # For safety, if the directory does not exist, try using frame_id with extension removed.
        if not os.path.isdir(sample_dir):
            # Try removing extension if accidentally included.
            sample_dir = os.path.join(self.samples_base, video_id, os.path.splitext(frame_id)[0])
            if not os.path.isdir(sample_dir):
                raise ValueError(f"Sample directory not found for video {video_id} frame {frame_id}")
        # Now, load the temporal window. Find the target frame filename from the files in sample_dir.
        all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))]
        all_files.sort()
        target_filename = None
        for name in possible_names:
            if name in all_files:
                target_filename = name
                break
        if target_filename is None:
            # If not found, take the middle image.
            target_filename = all_files[len(all_files)//2]
        frames = self._load_frame_window(sample_dir, target_filename)
        # Return a dictionary with full–frame window, player annotations, and group label.
        return {
            "frames": frames,  # [T, C, H, W]
            "player_annots": sample["player_annots"],  # list of dicts, each with "action" and "bbox"
            "group_label": torch.tensor(sample["group_label"]).long()
        }



class VideoAnnotationDataset(Dataset):
    def __init__(self, annotation, split='train', base_path='', transform=None,
                 sequence_length=16, video_list=None):
        """
        Args:
            annotation (dict): Dictionary loaded from the annotation pickle file.
            split (str): 'train' or 'test'.
            base_path (str): Base directory where video files are stored (e.g. "data").
            transform: Albumentations transform (or similar) to apply to each image.
            sequence_length (int): Number of frames to sample from each video.
            video_list (list): Optionally, a custom list of video relative paths. If None,
                               the list is taken from annotation['train_videos'][0] or
                               annotation['test_videos'][0] based on the split.
        """
        self.annotation = annotation
        self.split = split
        self.base_path = base_path
        self.transform = transform
        self.sequence_length = sequence_length

        # Use provided video_list or load from annotation based on the split.
        if video_list is None:
            if split == 'train':
                self.video_list = annotation['train_videos'][0]
            elif split == 'test':
                self.video_list = annotation['test_videos'][0]
            else:
                raise ValueError("Invalid split: choose 'train' or 'test'")
        else:
            self.video_list = video_list

        # Build the list of samples.
        self.samples = []
        for vid in self.video_list:
            # Determine the label from the gttubes field.
            if vid in annotation['gttubes']:
                tube_dict = annotation['gttubes'][vid]
                if len(tube_dict) > 0:
                    # The keys in the tube dictionary are integers.
                    sorted_keys = sorted(tube_dict.keys())
                    # Choose the smallest key (or modify this logic if needed)
                    label_idx = sorted_keys[0]
                else:
                    label_idx = 0  # Default if no tubes.
            else:
                label_idx = 0  # Default label if missing in gttubes.
            sample = {'video_path': vid, 'label': label_idx}
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def _load_frames_from_video(self, video_path):
        # Construct full path.
        full_path = os.path.join(self.base_path, video_path)
        # If the file does not exist and extension is missing, try appending .mp4
        if not os.path.exists(full_path):
            full_path += '.mp4'
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {full_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video file {full_path} has no frames.")

        # Compute evenly spaced frame indices (0-indexed)
        indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        frames = []
        current_frame = 0
        idx_set = set(indices.tolist())
        while cap.isOpened() and len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in idx_set:
                # Convert from BGR to RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            current_frame += 1

        cap.release()

        # If fewer frames were collected, pad by repeating the last frame.
        if len(frames) < self.sequence_length:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1].copy())
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        frames = self._load_frames_from_video(video_path)

        processed_frames = []
        for frame in frames:
            if self.transform:
                augmented = self.transform(image=frame)
                image = augmented["image"]
            else:
                # Convert to tensor and change shape to [C, H, W].
                image = torch.from_numpy(frame).permute(2, 0, 1).float()
            processed_frames.append(image)

        # Stack frames into a tensor of shape [T, C, H, W].
        frames_tensor = torch.stack(processed_frames)
        label = torch.tensor(sample['label']).long()
        return frames_tensor, label


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
