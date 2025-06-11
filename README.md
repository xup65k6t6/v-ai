# 🏐 V-AI: Volleyball Activity Recognition

An AI system for automatic volleyball activity recognition and scoring event detection from video footage. This project uses deep learning models to analyze volleyball game videos and identify key game activities in real-time.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 🎯 Features

- **Automatic Activity Detection**: Identifies volleyball game activities including spikes, sets, passes, and scoring events
- **3D CNN Architecture**: Uses ResNet3D-18 for spatiotemporal feature extraction
- **Real-time Processing**: Efficient inference pipeline for video analysis
- **Visual Annotations**: Outputs videos with activity labels overlaid
- **Configurable Pipeline**: Easy-to-customize training and inference settings

## 📊 Supported Activities

The model recognizes 8 different volleyball activities:
- Waiting
- Setting
- Digging
- Falling
- Spiking
- Blocking
- Jumping
- Moving

## 🚀 Quick Start

See our [Quick Start Guide](QUICK_START.md) for detailed setup instructions.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/v-ai.git
cd v-ai

# Install dependencies
pip install -r requirements.txt
```

### Demo

Run inference on your volleyball video:

```bash
# Simple demo with default settings
python demo.py --video path/to/your/volleyball_video.mp4

# Specify output location
python demo.py --video input.mp4 --output results/annotated.mp4
```

## 📂 Project Structure

```
v-ai/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── demo.py                      # Simple demo script
├── config/
│   ├── config_example_3dcnn.yaml      # Example training config
│   ├── config_example_videomae.yaml   # VideoMAE config (experimental)
│   └── config_inference.yaml          # Inference settings
├── data/
│   └── videos/                  # Place your training videos here
├── checkpoints/                 # Model checkpoints
├── examples/                    # Example outputs
└── v_ai/
    ├── train_3dcnn.py          # 3D CNN training script
    ├── train_videomae_v2.py    # VideoMAE training (experimental)
    ├── inference_3dcnn.py      # Inference pipeline
    ├── models/                 # Model architectures
    ├── utils/                  # Utility functions
    └── ...
```

## 🎓 Training

### Prepare Your Dataset

1. Organize your volleyball videos in the `data/videos/` directory
2. Each video should be labeled with frame-level activity annotations
3. Update the video IDs in your config file for train/val/test splits

### Train 3D CNN Model

```bash
# Copy and modify the example config
cp config/config_example_3dcnn.yaml config/my_config.yaml

# Start training
python v_ai/train_3dcnn.py --config config/my_config.yaml

# Resume from checkpoint
python v_ai/train_3dcnn.py --config config/my_config.yaml --resume
```

### Monitor Training

Training progress is logged to Weights & Biases. View metrics in real-time:
- Loss curves
- Accuracy, Precision, Recall, F1 scores
- Learning rate scheduling

## 🔧 Configuration

Key parameters in the config file:

```yaml
# Video processing
window_before: 3      # Frames before key frame
window_after: 4       # Frames after key frame
image_size: 320       # Input image size

# Training
batch_size: 8         # Adjust based on GPU memory
learning_rate: 0.0001 # Initial learning rate
num_epochs: 50        # Total training epochs
```

## 📈 Model Performance

The 3D CNN model achieves strong performance on volleyball activity recognition:
- **Architecture**: ResNet3D-18 with temporal convolutions
- **Input**: 8-frame sequences (configurable)
- **Training time**: ~2-4 hours on a single GPU (dataset dependent)

## 🔍 Advanced Usage

### Batch Inference

Process multiple videos:

```bash
python v_ai/inference_3dcnn.py \
    --input_dir path/to/videos/ \
    --output_dir path/to/results/ \
    --config config/config_inference.yaml \
    --checkpoint_path checkpoints/best_3dcnn.pt
```

### Custom Training

For advanced training options:
- Distributed training supported via PyTorch DDP
- Custom data augmentations in `transforms.py`
- Early stopping and learning rate scheduling
- Checkpoint saving and resuming

## 🧪 Experimental: VideoMAE V2

We also include an experimental VideoMAE V2 implementation, though the 3D CNN currently performs better:

```bash
python v_ai/train_videomae_v2.py \
    --config config/config_example_videomae.yaml \
    --output_dir ./checkpoints/videomae_run1
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{vai2024,
  title = {V-AI: Volleyball Activity Recognition},
  author = {Christopher Lin},
  year = {2024},
  url = {https://github.com/yourusername/v-ai}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet3D architecture from torchvision
- VideoMAE implementation from HuggingFace Transformers
- Volleyball dataset annotations and preprocessing tools

---

<p align="center">
  Made with ❤️ for the volleyball community
</p>
