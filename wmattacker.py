from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from bm3d import bm3d_rgb
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError


class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, quality=1, metric='mse', device='cpu'):
        if model_name == 'bmshj2018-factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'bmshj2018-hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018-mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'cheng2020-anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
        else:
            raise ValueError('model name not supported')
        self.device = device

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512))
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            out = self.model(img)
            out['x_hat'].clamp_(0, 1)
            rec = transforms.ToPILImage()(out['x_hat'].squeeze().cpu())
            rec.save(out_path)


class GaussianBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = cv2.imread(img_path)
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            cv2.imwrite(out_path, img)


class GaussianNoiseAttacker(WMAttacker):
    def __init__(self, std):
        self.std = std

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            image = cv2.imread(img_path)
            image = image / 255.0
            # Add Gaussian noise to the image
            noise_sigma = self.std  # Vary this to change the amount of noise
            noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
            # Clip the values to [0, 1] range after adding the noise
            noisy_image = np.clip(noisy_image, 0, 1)
            noisy_image = np.array(255 * noisy_image, dtype='uint8')
            cv2.imwrite(out_path, noisy_image)


class BM3DAttacker(WMAttacker):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path).convert('RGB')
            y_est = bm3d_rgb(np.array(img) / 255, 0.1)  # use standard deviation as 0.1, 0.05 also works
            plt.imsave(out_path, np.clip(y_est, 0, 1), cmap='gray', vmin=0, vmax=1)


class JPEGAttacker(WMAttacker):
    def __init__(self, quality=80):
        self.quality = quality

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            img.save(out_path, "JPEG", quality=self.quality)


class BrightnessAttacker(WMAttacker):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
            img.save(out_path)


class ContrastAttacker(WMAttacker):
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
            img.save(out_path)


class RotateAttacker(WMAttacker):
    def __init__(self, degree=30):
        self.degree = degree

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            img = img.rotate(self.degree)
            img.save(out_path)


class ScaleAttacker(WMAttacker):
    def __init__(self, scale=0.5):
        self.scale = scale

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            w, h = img.size
            img = img.resize((int(w * self.scale), int(h * self.scale)))
            img.save(out_path)


class CropAttacker(WMAttacker):
    def __init__(self, crop_size=0.5):
        self.crop_size = crop_size

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            w, h = img.size
            img = img.crop((int(w * self.crop_size), int(h * self.crop_size), w, h))
            img.save(out_path)


def add_noise(sigmas, latents, t, noise):
    sigmas = sigmas.to(device=latents.device, dtype=latents.dtype)
    sigma = sigmas[t].flatten()

    while len(sigma.shape) < len(latents.shape):
        sigma = sigma.unsqueeze(-1)

    latents = sigma * noise + (1.0 - sigma) * latents
    return latents


class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe # 预训练的diffusion模型
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device) # 攻击噪声加入的denoise步数
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                # 图像重建：prompts_buf + latens_buf -> images
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents, # 加入攻击噪声后的latents
                                   head_start_step=50 - max(self.noise_step // 20, 1), # 默认num_inference_steps=50，减去加入攻击噪声已用的denoise步数，从此步继续denoise
                                   guidance_scale=7.5, # > 1.0，使用classifier_free_guidance
                                   generator=generator, 
                                   num_inference_steps = self.num_inference_steps,
                                   )
                images = images[0]
                # 重建的图像保存到outs_buf
                for img, out in zip(images, outs_buf):
                    img.save(out)

            # 如果有caption，则使用图像文件名对应的caption作为prompt
            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                
                # 通过VAE获取latents
                latents = self.pipe.vae.encode(img).latent_dist # latens分布
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor # 从分布中采样，并scale
                
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device) # 对应vae下采样8倍latens生成攻击噪声
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half) # 攻击噪声经timestep步denoise加入latents，生成新的latents
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                
                # batch个图像都处理为latents，并添加噪声后，进行重建
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                    
                if return_latents:
                    ret_latents.append(latents.cpu())

            # 处理剩余的图像
            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            
            if return_latents:
                return ret_latents
            

class Diff3WMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=None, num_inference_steps=None, captions={}):
        self.pipe = pipe # 预训练的diffusion模型
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        self.num_inference_steps = num_inference_steps
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []

            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                # 图像重建：prompts_buf + latens_buf -> images
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents, # 加入攻击噪声后的latents
                                   head_start_step=28 - max(self.noise_step // 20, 1), # 默认num_inference_steps=50，减去加入攻击噪声已用的denoise步数，从此步继续denoise
                                #    head_start_step=28,
                                   guidance_scale=7.0, # > 1.0，使用classifier_free_guidance
                                   generator=generator, )
                images = images[0]
                # 重建的图像保存到outs_buf
                for img, out in zip(images, outs_buf):
                    img.save(out)

            # 如果有caption，则使用图像文件名对应的caption作为prompt
            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            timesteps = np.linspace(1, self.num_inference_steps, self.num_inference_steps, dtype=np.float32)[::-1].copy()
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
            sigmas = timesteps / self.num_inference_steps


            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                
                # 通过VAE获取latents
                latents = self.pipe.vae.encode(img).latent_dist # 此处stable-diffusion-3与stable-diffusion-2用法相同
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor # 从分布中采样，并scale
                
                chs = latents.shape[1]
                noise = torch.randn([1, chs, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device, dtype=latents.dtype) # 对应vae下采样8倍latens生成攻击噪声
                latents = add_noise(sigmas, latents, self.noise_step, noise)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                
                # batch个图像都处理为latents，并添加噪声后，进行重建
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                    
                if return_latents:
                    ret_latents.append(latents.cpu())

            # 处理剩余的图像
            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            
            if return_latents:
                return ret_latents
