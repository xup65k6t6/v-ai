#!/usr/bin/env python3
"""
V-AI Demo Script
Simple wrapper for volleyball activity recognition inference using the 3D CNN model.

Usage:
    python demo.py --video path/to/your/video.mp4
    python demo.py --video path/to/your/video.mp4 --output path/to/output.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v_ai.inference_3dcnn import main as inference_main


def create_demo_args(video_path, output_path=None, checkpoint_path=None):
    """Create arguments for the inference script."""
    # Default paths
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"demo_output_{video_name}.mp4"
    
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/best_3dcnn.pt"
    
    # Create temporary directories for single video processing
    temp_input_dir = Path("temp_demo_input")
    temp_output_dir = Path("temp_demo_output")
    temp_input_dir.mkdir(exist_ok=True)
    temp_output_dir.mkdir(exist_ok=True)
    
    # Copy video to temp input directory
    import shutil
    temp_video_path = temp_input_dir / Path(video_path).name
    shutil.copy2(video_path, temp_video_path)
    
    # Create argument list for inference script
    args = [
        "--input_dir", str(temp_input_dir),
        "--output_dir", str(temp_output_dir),
        "--config", "config/config_inference.yaml",
        "--checkpoint_path", checkpoint_path,
    ]
    
    return args, temp_output_dir / Path(video_path).name, output_path


def main():
    parser = argparse.ArgumentParser(
        description="V-AI Demo: Volleyball Activity Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single video with default settings
  python demo.py --video path/to/video.mp4
  
  # Specify output location
  python demo.py --video path/to/video.mp4 --output results/annotated.mp4
  
  # Use a custom checkpoint
  python demo.py --video path/to/video.mp4 --checkpoint checkpoints/my_model.pt
        """
    )
    
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path for output video (default: demo_output_<video_name>.mp4)"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_3dcnn.pt)"
    )
    
    args = parser.parse_args()
    
    # Validate input video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Check if checkpoint exists (use default if not specified)
    checkpoint = args.checkpoint or "checkpoints/best_3dcnn.pt"
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        print("Please ensure you have trained a model or downloaded pretrained weights.")
        print("Expected location: checkpoints/best_3dcnn.pt")
        sys.exit(1)
    
    # Create output directory if needed
    if args.output:
        os.makedirs(Path(args.output).parent, exist_ok=True)
    
    print("🏐 V-AI Volleyball Activity Recognition Demo")
    print(f"📹 Processing video: {args.video}")
    print(f"🧠 Using checkpoint: {checkpoint}")
    print(f"💾 Output will be saved to: {args.output or f'demo_output_{Path(args.video).stem}.mp4'}")
    print("-" * 50)
    
    # Prepare arguments for inference script
    inference_args, temp_output_path, final_output_path = create_demo_args(args.video, args.output, checkpoint)
    
    # Temporarily modify sys.argv for the inference script
    original_argv = sys.argv
    sys.argv = ["inference_3dcnn.py"] + inference_args
    
    try:
        # Run inference
        inference_main()
        
        # Move the output file to the desired location
        import shutil
        if temp_output_path.exists():
            shutil.move(str(temp_output_path), str(final_output_path))
        
        # Clean up temporary directories
        shutil.rmtree("temp_demo_input", ignore_errors=True)
        shutil.rmtree("temp_demo_output", ignore_errors=True)
        
        print("\n✅ Demo completed successfully!")
        print(f"📹 Output video saved to: {final_output_path}")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        # Clean up temporary directories even on error
        import shutil
        shutil.rmtree("temp_demo_input", ignore_errors=True)
        shutil.rmtree("temp_demo_output", ignore_errors=True)
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
