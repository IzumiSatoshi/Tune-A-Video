import sys
sys.path.append("./Tune-A-Video")

from tuneavideo.models.unet import UNet3DConditionModel
import torch

unet = UNet3DConditionModel(sample_size=64, cross_attention_dim=768)
sample = torch.randn(1, 4, 3, 64, 64)
timestep = 1
encoder_hidden_states = torch.randn(1, 320, 768)
hint = torch.randn(1, 3, 3, 512, 512)
res = unet(sample, timestep, encoder_hidden_states, disable_sc_attn=True)