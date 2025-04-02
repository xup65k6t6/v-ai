import os
import cv2
from tqdm import tqdm

def frames_to_video(frame_dir, output_path, fps=30):
    images = sorted(
        [img for img in os.listdir(frame_dir) if img.endswith((".jpg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not images:
        print(f"No images found in {frame_dir}")
        return

    first_frame_path = os.path.join(frame_dir, images[0])
    frame = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read the first image: {first_frame_path}")
    height, width, _ = frame.shape

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video: {output_path}")
    for img_name in tqdm(images, desc=f"Processing {os.path.basename(frame_dir)}"):
        img_path = os.path.join(frame_dir, img_name)
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: Skipped unreadable image {img_path}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()

def process_all(base_dir="data/videos", output_base="data/rendered_videos", fps=30):
    for main_folder in sorted(os.listdir(base_dir)):
        main_path = os.path.join(base_dir, main_folder)
        if not os.path.isdir(main_path):
            continue

        for sub_folder in sorted(os.listdir(main_path)):
            frame_dir = os.path.join(main_path, sub_folder)
            if not os.path.isdir(frame_dir):
                continue

            # Create matching output subfolder and file
            output_dir = os.path.join(output_base, main_folder)
            output_path = os.path.join(output_dir, f"{sub_folder}.mp4")
            frames_to_video(frame_dir, output_path, fps)

if __name__ == "__main__":
    # === Test one folder ===
    # test_folder = "data/videos/0/13286"
    # test_output = "data/rendered_videos/0/13286.mp4"
    # frames_to_video(test_folder, test_output)

    # === Process all ===
    process_all()
