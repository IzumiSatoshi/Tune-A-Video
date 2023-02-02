import os
from PIL import Image
import imageio
import numpy as np

import torch
import torchvision

from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=3):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# TODO: This code may degrade image quality
def read_gif(path) -> list[Image.Image]:
    gif_image = Image.open(path)
    images = []
    for index in range(gif_image.n_frames):
        gif_image.seek(index)
        images.append(gif_image.copy().convert("RGB"))

    return images


# TODO: This code may degrade image quality
def save_as_gif(images: list[Image.Image], path, duration=500, loop=0):
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration, loop=loop
    )
