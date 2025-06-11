# 🚀 V-AI Quick Start Guide

Get up and running with volleyball activity recognition in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/v-ai.git
cd v-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Download Pre-trained Model

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download pre-trained model (example command - replace with actual URL)
# wget https://example.com/best_3dcnn.pt -O checkpoints/best_3dcnn.pt
```

## Run Demo

```bash
# Process a single video
python demo.py --video path/to/your/volleyball_video.mp4

# The annotated video will be saved as demo_output_<video_name>.mp4
```

## Understanding the Output

The model will annotate your video with the detected volleyball activities:

- **Waiting**: Players in ready position
- **Setting**: Ball being set for attack
- **Digging**: Defensive save
- **Spiking**: Attacking hit
- **Blocking**: Defensive block at net
- **Moving**: Players in motion
- **Jumping**: Vertical movement
- **Falling**: Player going to ground

Each prediction includes a confidence score (0.00-1.00).

## Training Your Own Model

### 1. Prepare Your Dataset

Structure your data as follows:
```
data/
└── videos/
    ├── video_001/
    │   ├── frames/
    │   └── annotations.json
    ├── video_002/
    │   ├── frames/
    │   └── annotations.json
    └── ...
```

### 2. Configure Training

```bash
# Copy and modify the example config
cp config/config_example_3dcnn.yaml config/my_training.yaml

# Edit the config file
# - Set your video IDs for train/val/test splits
# - Adjust hyperparameters as needed
```

### 3. Start Training

```bash
python v_ai/train_3dcnn.py --config config/my_training.yaml
```

### 4. Monitor Progress

Training metrics are logged to Weights & Biases. You'll see:
- Real-time loss curves
- Accuracy metrics
- Model checkpoints saved automatically

## Batch Processing

Process multiple videos at once:

```bash
python v_ai/inference_3dcnn.py \
    --input_dir path/to/videos/ \
    --output_dir path/to/results/ \
    --config config/config_inference.yaml
```

## Common Issues

### Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size` (default: 320)
- Use CPU instead of GPU

### Slow Inference
- Reduce `image_size` in config
- Increase `stride` parameter
- Use GPU if available

### Poor Predictions
- Ensure video quality is good
- Check if the sport is volleyball
- Consider fine-tuning on your data

## Next Steps

1. **Fine-tune on your data**: Collect volleyball videos and annotations
2. **Optimize for speed**: Experiment with model quantization
3. **Deploy**: Create a web service or mobile app
4. **Contribute**: Submit improvements back to the project!

## Need Help?

- Check the [full documentation](README.md)
- Open an [issue](https://github.com/yourusername/v-ai/issues)
- Read the [contribution guide](CONTRIBUTING.md)

Happy volleying! 🏐
