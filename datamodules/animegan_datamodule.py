from os import PathLike
from typing import Union
from pathlib import Path

from torch.utils.data import Dataset
import torch
from PIL import Image


class AnimeDataSet(Dataset):
    def __init__(
        self,
        photo_dir: Union[str, PathLike],
        anime_dir: Union[str, PathLike],
        smooth_dir: Union[str, PathLike],
        transform=None,
    ):
        self.photo_dir = Path(photo_dir)
        self.anime_dir = Path(anime_dir)
        self.smooth_dir = Path(smooth_dir)
        self.photo_files = sorted(self.photo_dir.rglob("*.png"))
        self.anime_files = sorted(self.anime_dir.rglob("*.jpg"))
        self.smooth_files = sorted(self.smooth_dir.rglob("*.jpg"))
        self._transform = transform

        self.len_photo = len(self.photo_files)
        self.len_anime = len(self.anime_files)

    def __len__(self):
        return max(len(self.photo_files), len(self.anime_files))

    def __getitem__(
        self, index: int
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        photo_idx = index
        anime_idx = index

        if anime_idx > self.len_anime - 1:
            anime_idx -= self.len_anime * (index // self.len_anime)

        if photo_idx > self.len_photo - 1:
            photo_idx -= self.len_photo * (index // self.len_photo)

        photo = self.load_photo(photo_idx)
        anime, anime_gray = self.load_anime(anime_idx)
        smooth_gray = self.load_anime_smooth(anime_idx)

        return photo, anime, anime_gray, smooth_gray

    def load_photo(self, index: int) -> torch.Tensor:
        fpath = self.photo_files[index]
        image = Image.open(fpath).convert("RGB")
        image = self._transform(image)
        return image

    def load_anime(self, index: int) -> torch.Tensor:
        fpath = self.anime_files[index]
        image = Image.open(fpath).convert("RGB")
        image = self._transform(image)

        image_gray = Image.open(fpath).convert("L")  # Gray Scale
        image_gray = self._transform(image_gray)

        return image, image_gray

    def load_anime_smooth(self, index: int) -> torch.Tensor:
        fpath = self.smooth_files[index]

        image = Image.open(fpath).convert("L")  # Gray Scale
        image = self._transform(image)
        return image
