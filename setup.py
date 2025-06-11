"""Setup script for V-AI: Volleyball Activity Recognition"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='v-ai',
    version='0.1.0',
    author='Christopher Lin',
    author_email='example@example.com',
    description='AI system for automatic volleyball activity recognition and scoring event detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/v-ai',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=22.0',
            'flake8>=4.0',
            'isort>=5.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'v-ai-demo=demo:main',
            'v-ai-train-3dcnn=v_ai.train_3dcnn:main',
            'v-ai-inference=v_ai.inference_3dcnn:main',
        ],
    },
    include_package_data=True,
    package_data={
        'v_ai': ['config/*.yaml'],
    },
)
