# AnimeGAN2 - PyTorch Lighining Implementation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


PyTorch Lightning implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)

*This code is forked from [bryandlee's animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)*


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

## Acknowledgments

- Original AnimeGAN2 implementation: [TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- PyTorch implementation: [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)

## License

MIT License - see [LICENSE](LICENSE) file for details.

# Contact
Mail: kangnam7653@gmail.com  
LinkedIn: [kangnam7654](https://www.linkedin.com/in/kangnam7654)  
Resume: [Link](https://kangnam7654.github.io)

