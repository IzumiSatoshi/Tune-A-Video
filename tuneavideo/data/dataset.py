import decord

decord.bridge.set_bridge("torch")
import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange

from transformers import CLIPTokenizer


class TuneAVideoDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": self.prompt_ids}

        return example


class TuneAVideoMoreShotDataset(Dataset):
    """
    Adjust TuneAVideoDataset to more general Text2Video training.
    """

    def __init__(
        self,
        annotations_file: str,
        video_dir: str,
        tokenizer: CLIPTokenizer,
        size: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.df = None

        if annotations_file.endswith(".jsonl"):
            self.df = pd.read_json(annotations_file, lines=True)
        elif annotations_file.endswith(".csv"):
            self.df = pd.read_csv(annotations_file)

        self.video_dir = video_dir
        self.tokenizer = tokenizer  # is this right place?

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # process video
        video_path = os.path.join(self.video_dir, self.df.loc[idx, "video_file"])
        vr = decord.VideoReader(video_path)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        video = video / 255.0  # 0~255 -> 0~1
        video = (video * 2.0) - 1.0  # 0~1 -> -1~1
        video = self.transform(video)

        # process prompt
        prompt = self.df.loc[idx, "prompt"]
        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        example = {"pixel_values": video, "prompt_ids": prompt_ids, "prompt": prompt}

        return example

class TuneAVideoControlNetDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        prompt: str,
        controlnet_hint_path: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None
        self.controlnet_hint_path = controlnet_hint_path

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        x_vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        x_sample_index = list(
            range(self.sample_start_idx, len(x_vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        x_video = x_vr.get_batch(x_sample_index)
        x_video = rearrange(x_video, "f h w c -> f c h w")

        hint_vr = decord.VideoReader(self.controlnet_hint_path, width=self.width, height=self.height)
        hint_sample_index = list(
            range(self.sample_start_idx, len(hint_vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        hint_video = hint_vr.get_batch(hint_sample_index)
        hint_video = rearrange(hint_video, "f h w c -> f c h w")

        example = {
            "pixel_values": (x_video / 127.5 - 1.0), 
            "pixel_values_hint": (hint_video / 127.5 - 1.0), 
            "prompt_ids": self.prompt_ids, 
        }
        return example
