# AnimeGAN2 - PyTorch Lighining Implementation

PyTorch Lightning implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)

with improved project structure and modern practices.
*This code is forked from [bryandlee's animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)*

## Project Structure

```
animegan2-pytorch_lightning/    # Project root directory
│
├── animegan2/                  # Main package
│   ├── data/                   # Custom dataset class
│   ├── inference/              # Inference logic
│   ├── models/                 # Model definitions
│   ├── scripts/                # Scripts for training, inference, etc.
│   ├── serving/                # Serving Logic
│   └── training/               # Lightning module for training
│
├── deployment/                 # Deployment configurations
│   ├── docker/                 # Dockerfiles and Docker Compose files
│   ├── kubernetes/             # Kubernetes deployment files
│   └── triton/                 # Triton Inference Server configurations
│
├── docs/                       # Papers for this project
├── examples/                   # Example images and results
├── tests/                      # Test files
└── weights/                    # Pre-trained model weights
```

## Data
### FFHQ Dataset
https://github.com/NVlabs/ffhq-dataset
### Anime Dataset
https://github.com/kangnam7654/dfhq

## Installation

```bash
# Clone the repository
git clone https://github.com/kangnam7654/animegan2-pytorch_lightning.git
cd animegan2
```

## Usage
### Training
#### Step 1: Prepare edge smoothing data
```bash
uv run animegan2/utils/edge_smooth.py --root_dir {ANIME_IMAGE_DIRECTORY} --out_dir {EDGE_SMOOTH_SAVE_DIRECTORY} --image_size {EDGE_SMOOTH_IMAGE_SIZE}
```

#### Step 2: Train the model
```bash
uv run animegan2/scripts/train.py --photo_dir {PHOTO_IMAGE_DIRECTORY} --anime_dir {ANIME_IMAGE_DIRECTORY} --smooth_dir {SMOOTH_IMAGE_DIRECTORY}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original AnimeGAN2 implementation: [TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- PyTorch implementation: [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)