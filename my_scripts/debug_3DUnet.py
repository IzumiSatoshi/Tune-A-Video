import sys
sys.path.append(".")

from tuneavideo.models.unet import UNet3DConditionModel
import torch

path = "./checkpoints/tav_yor_dedede/unet"
unet = UNet3DConditionModel.from_pretrained(path, torch_dtype=torch.float)
sample = torch.randn(1, 4, 1, 64, 64)
timestep = 1
encoder_hidden_states = torch.randn(1, 320, 768)
res = unet(sample, timestep, encoder_hidden_states)