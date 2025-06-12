# Contributing to V-AI

Thank you for your interest in contributing to V-AI! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/xup65k6t6/v-ai.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## 📋 Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## 🧪 Testing

Before submitting a pull request, ensure:

1. Your code follows the existing style
2. All tests pass (when available)
3. You've added tests for new functionality
4. Documentation is updated

```bash
# Run tests
pytest

# Check code style
black v_ai/
flake8 v_ai/
isort v_ai/
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Comment complex logic

Example docstring format:
```python
def process_video(video_path: str, model: nn.Module) -> List[Tuple[int, str]]:
    """
    Process a video and return activity predictions.
    
    Args:
        video_path: Path to the input video file
        model: Trained model for inference
        
    Returns:
        List of tuples containing (frame_index, activity_label)
    """
```

## 🌟 Areas for Contribution

### High Priority
- Improve model accuracy
- Add support for more video formats
- Optimize inference speed
- Add unit tests

### Medium Priority
- Support for real-time video streams
- Web interface for demo
- Better visualization of results
- Support for additional sports

### Low Priority
- Multi-language documentation
- Docker containerization
- Cloud deployment examples

## 🔄 Pull Request Process

1. Update the README.md with details of changes if needed
2. Increase version numbers in setup.py and pyproject.toml
3. Ensure your PR description clearly describes the problem and solution
4. Link any relevant issues

### PR Title Format
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks

Example: `feat: Add support for real-time video processing`

## 🐛 Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

Feature requests are welcome! Please:
- Check existing issues first
- Clearly describe the feature
- Explain why it would be useful
- Provide examples if possible

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing opinions

## 📬 Contact

For questions not covered here, please open an issue or contact the maintainers.

Thank you for contributing to V-AI! 🏐
