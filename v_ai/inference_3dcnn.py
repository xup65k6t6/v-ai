# v_ai/inference_3dcnn.py

import argparse
import os
import cv2
import torch
import yaml

from collections import deque
from tqdm import tqdm # For progress bars

from v_ai.data import GROUP_ACTIVITY_MAPPING
from v_ai.models.model import Video3DClassificationModel
from v_ai.transforms import resize_only # Use the same transform as training
from v_ai.utils.utils import get_device

# Reverse mapping for easy lookup of class names
INDEX_TO_GROUP_ACTIVITY = {v: k for k, v in GROUP_ACTIVITY_MAPPING.items()}

def load_model_checkpoint(model, checkpoint_path, device):
    """Loads the model checkpoint, handling potential 'module.' prefix."""
    try:
        # Load onto CPU first to avoid GPU memory issues if checkpoint was saved on different GPU setup
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check if the checkpoint is a dictionary with 'model_state_dict'
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Remove 'module.' prefix if present (from DDP training)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # Load the state dict
        model.load_state_dict(new_state_dict)
        # Move model to the target device *after* loading state_dict
        model.to(device)
        print(f"Successfully loaded model checkpoint from: {checkpoint_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Attempt loading without stripping 'module.' as a fallback
        try:
            model.load_state_dict(state_dict)
            model.to(device)
            print(f"Successfully loaded model checkpoint (without stripping 'module.') from: {checkpoint_path}")
            return model
        except Exception as fallback_e:
            print(f"Fallback loading also failed: {fallback_e}")
            exit(1)

def predict_clips(video_path, model, transform, config, device, stride, batch_size=8):
    """
    Pass 1: Processes a video file, predicts group activity for each clip,
            and returns a list of predictions with frame ranges.

    Args:
        video_path (str): Path to the input video file.
        model (torch.nn.Module): The loaded 3D CNN model.
        transform (callable): The transformation function for frames.
        config (dict): The configuration dictionary.
        device (torch.device): The device to run inference on.
        stride (int): Number of frames to step forward between clips.

    Returns:
        list: A list of tuples: [(start_frame_idx, end_frame_idx, predicted_label), ...]
              Frame indices are 0-based.
        int: Total number of frames in the video.
        float: Frames per second (FPS) of the video.
        int: Video width.
        int: Video height.
    """
    window_before = config['window_before']
    window_after = config['window_after']
    sequence_length = window_before + 1 + window_after  # T

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, 0, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Pass 1: Predicting clips for {video_path} ({frame_count} frames, {fps:.2f} FPS)")
    print(f"Using clip length (T): {sequence_length}, stride: {stride}")

    frames_buffer = deque(maxlen=sequence_length)
    frame_idx = -1 # Start at -1, will become 0 on first read
    clip_predictions = [] # Store results here

    model.eval() # Set model to evaluation mode

    # Batch buffers
    batch_clip_tensors = []
    batch_clip_info = []

    # Use tqdm for progress bar
    pbar = tqdm(total=frame_count, desc="Predicting Clips")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            frame_idx += 1
            pbar.update(1)

            # Convert BGR (OpenCV default) to RGB for transform
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transformations (expects numpy array [H, W, C])
            augmented = transform(image=frame_rgb)
            # Transform outputs tensor [C, H, W]
            transformed_frame = augmented["image"]
            frames_buffer.append(transformed_frame)

            # Check if the buffer is full enough to form a clip
            if len(frames_buffer) == sequence_length:
                clip_frames = list(frames_buffer)
                clip_tensor = torch.stack(clip_frames)             # [T, C, H, W]
                clip_tensor = clip_tensor.permute(1, 0, 2, 3)      # [C, T, H, W]
                clip_tensor = clip_tensor.unsqueeze(0)             # [1, C, T, H, W]

                batch_clip_tensors.append(clip_tensor.squeeze(0))  # [C, T, H, W]
                batch_clip_info.append((frame_idx - sequence_length + 1, frame_idx))

                if len(batch_clip_tensors) == batch_size:
                    batch_tensor = torch.stack(batch_clip_tensors).to(device, non_blocking=True)  # [B, C, T, H, W]
                    logits = model(batch_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_indices = torch.argmax(probabilities, dim=1)

                    for i in range(batch_size):
                        pred_idx = predicted_indices[i].item()
                        pred_label = INDEX_TO_GROUP_ACTIVITY.get(pred_idx, "Unknown")
                        confidence = probabilities[i, pred_idx].item()
                        start, end = batch_clip_info[i]
                        clip_predictions.append((start, end, pred_label, confidence))

                    batch_clip_tensors.clear()
                    batch_clip_info.clear()

                # Slide the window
                for _ in range(stride):
                    if frames_buffer:
                        frames_buffer.popleft()

    # Final flush
    if batch_clip_tensors:
        batch_tensor = torch.stack(batch_clip_tensors).to(device, non_blocking=True)
        logits = model(batch_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)

        for i in range(len(batch_clip_tensors)):
            pred_idx = predicted_indices[i].item()
            pred_label = INDEX_TO_GROUP_ACTIVITY.get(pred_idx, "Unknown")
            confidence = probabilities[i, pred_idx].item()
            start, end = batch_clip_info[i]
            clip_predictions.append((start, end, pred_label, confidence))

    pbar.close()
    cap.release()
    print(f"Pass 1 Finished. Generated {len(clip_predictions)} predictions.")
    return clip_predictions, frame_count, fps, width, height

def annotate_video(video_path, output_path, clip_predictions, total_frames, fps, width, height, confidence_threshold=0.0):
    """
    Pass 2: Reads the video again, draws the predicted labels onto frames,
            and writes the annotated video to the output file.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output annotated video.
        clip_predictions (list): List of tuples from predict_clips.
        total_frames (int): Total number of frames in the video.
        fps (float): Frames per second.
        width (int): Video width.
        height (int): Video height.
    """
    print(f"\nPass 2: Annotating video and writing to {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not reopen video file for annotation: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        return

    current_prediction_idx = 0
    active_label = None

    pbar = tqdm(total=total_frames, desc="Annotating Frames")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} during annotation pass.")
            break
        pbar.update(1)

        # Update active_label based on prediction and confidence
        while (current_prediction_idx < len(clip_predictions) and
               clip_predictions[current_prediction_idx][0] <= frame_idx):
            start, end, label, conf = clip_predictions[current_prediction_idx]
            if end >= frame_idx:
                active_label = f"{label} ({conf:.2f})" if conf >= confidence_threshold else None
            current_prediction_idx += 1

        # Backup check in case of gap
        if current_prediction_idx > 0:
            prev = clip_predictions[current_prediction_idx - 1]
            if prev[0] <= frame_idx <= prev[1]:
                active_label = f"{prev[2]} ({prev[3]:.2f})" if prev[3] >= confidence_threshold else None

        annotated_frame = frame.copy()

        if active_label:
            font_scale = 1.2
            thickness = 2
            margin_top = 30

            (text_width, text_height), baseline = cv2.getTextSize(active_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = margin_top + text_height

            cv2.rectangle(annotated_frame,
                          (text_x - 10, margin_top - 5),
                          (text_x + text_width + 10, margin_top + text_height + baseline + 5),
                          (0, 0, 0),
                          cv2.FILLED)

            cv2.putText(annotated_frame,
                        active_label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                        cv2.LINE_AA)

        out.write(annotated_frame)

    pbar.close()
    cap.release()
    out.release()
    print(f"Pass 2 Finished. Output video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch inference for 3D CNN Volleyball Activity Recognition")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save output annotated videos.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the training YAML config file.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', 'mps'). Auto-detects if None.")
    parser.add_argument("--stride", type=int, default=None, help="Frame stride between consecutive clips. Defaults to sequence_length // 2.")

    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit(1)

    # --- Parameters ---
    image_size = config.get('image_size', 640) 
    window_before = config.get('window_before', 3)
    window_after = config.get('window_after', 4)
    sequence_length = window_before + 1 + window_after
    num_classes = len(GROUP_ACTIVITY_MAPPING)
    checkpoint_dir = args.checkpoint_path if args.checkpoint_path else config.get('checkpoint_dir', 'checkpoints')

    # --- Device ---
    if args.device:
        # Basic validation
        if args.device not in ['cuda', 'cpu', 'mps']:
             print(f"Warning: Invalid device '{args.device}' specified. Using auto-detection.")
             device = get_device()
        elif args.device == 'cuda' and not torch.cuda.is_available():
             print("Warning: Device 'cuda' specified but not available. Using auto-detection.")
             device = get_device()
        elif args.device == 'mps' and not torch.backends.mps.is_available():
             print("Warning: Device 'mps' specified but not available. Using auto-detection.")
             device = get_device()
        else:
             device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")


    # --- Checkpoint Path ---
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Default name matches the one saved in train_3dcnn.py
        checkpoint_path = os.path.join(checkpoint_dir, "best_3dcnn.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        print(f"Please specify --checkpoint_path or ensure the file exists (expected name 'best_3dcnn.pt' in '{checkpoint_dir}').")
        exit(1)

    # --- Stride ---
    stride = args.stride if args.stride is not None else sequence_length // 2
    if stride <= 0:
        print(f"Warning: Stride ({stride}) must be positive. Setting stride to 1.")
        stride = 1
    # No warning needed if stride > sequence_length, just means non-overlapping clips


    # --- Load Model ---
    # Note: pretrained=False here, as we are loading our fine-tuned weights.
    model = Video3DClassificationModel(num_classes=num_classes, pretrained=False)
    model = load_model_checkpoint(model, checkpoint_path, device) # Function now handles moving model to device

    # --- Load Transform ---
    # Use the *exact same* transform configuration as during training/validation
    transform = resize_only(image_size=image_size)

    # --- Process Each Video ---
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        exit(0)

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    for video_name in video_files:
        input_path = os.path.join(args.input_dir, video_name)
        output_path = os.path.join(args.output_dir, video_name)

        print(f"\n--- Processing {video_name} ---")

        # --- Pass 1: Predict Clips ---
        clip_predictions, total_frames, fps, width, height = predict_clips(
            input_path, model, transform, config, device, stride
        )

        if clip_predictions is None or not clip_predictions:
            print(f"Skipping {video_name} due to insufficient predictions.")
            continue

        annotate_video(input_path, output_path, clip_predictions, total_frames, fps, width, height, confidence_threshold=0.9)



if __name__ == "__main__":
    main()