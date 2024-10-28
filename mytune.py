import torch
import os
import glob
import numpy as np

from regen_pipe import ReSD3Pipeline, ReSDPipeline
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline

from utils import eval_psnr_ssim_msssim, bytearray_to_bits
from watermarker import InvisibleWatermarker
from wmattacker import DiffWMAttacker, VAEWMAttacker, JPEGAttacker, Diff3WMAttacker
from diffusers.schedulers import PNDMScheduler  # TODO: 更换别的schedular
from modules import add_watermark, attack_wm


device = 'cuda:7'
data_path = './workshop'

# 生成水印
wm_text = 'test'
wm_name = ['dwtDct', 'dwtDctSvd', 'rivaGan']

dwtDct_wm = InvisibleWatermarker(wm_text, wm_name[0])
dwtDctSvd_wm = InvisibleWatermarker(wm_text, wm_name[1])
rivaGan_wm = InvisibleWatermarker(wm_text, wm_name[2])

# 加载pipeline
model_path = "/work/jingyiwang/models/stable-diffusion-3-medium-diffusers"
pipe = ReSD3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")
# pipe.scheduler = PNDMScheduler()

# model_path = "/work/jingyiwang/models/stable-diffusion-2-1"
# pipe = ReSDPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

pipe.set_progress_bar_config(disable=True)
pipe.to(device)

diff_attaacker = Diff3WMAttacker(pipe, batch_size=64, noise_step=27, num_inference_steps=28, captions={}) # 

# add_watermark(wm_name[0], dwtDct_wm, data_path)
# add_watermark(wm_name[1], dwtDctSvd_wm, data_path)
# add_watermark(wm_name[2], rivaGan_wm, data_path)

att_name = 'diff-5000'
# attack_wm(wm_name[0], att_name, diff_attaacker, data_path)
# attack_wm(wm_name[1], att_name, diff_attaacker, data_path)
# attack_wm(wm_name[2], att_name, diff_attaacker, data_path)
attack_wm(att_name, diff_attaacker, data_path)


pass
