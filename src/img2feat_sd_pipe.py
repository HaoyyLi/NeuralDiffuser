import os
import PIL
import torch
import numpy as np

from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from PIL import Image

import json
import h5py
import pandas as pd

from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel


def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():


    seed_everything(42)

    # imgidx = opt.imgidx
    gpu = 0
    resolution = 512        # cvpr--320  minddiffusion--512
    batch_size = 1
    max_length = 77
    torch_dtype = torch.float32
    model_dtype = 'fp32'
    # ddim_steps = 50
    # ddim_eta = 0.0
    # strength = 0.8
    # scale = 5.0
    nsd_path = ''
    output_dir_z = f'MindEyeV2/mindeyev2/'
    output_dir_c = f'MindEyeV2/mindeyev2/'
    
    torch.cuda.set_device(gpu)
    os.makedirs(output_dir_z, exist_ok=True)
    os.makedirs(output_dir_c, exist_ok=True)

    # Load moodels
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    model_path_diffusion = "/opt/data/private/huggingface/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", revision=model_dtype, torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_path_diffusion, subfolder="tokenizer", revision=model_dtype, torch_dtype=torch_dtype)
    text_encoder = CLIPTextModel.from_pretrained(model_path_diffusion, subfolder="text_encoder", revision=model_dtype, torch_dtype=torch_dtype)

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    vae.to(device)
    text_encoder.to(device)
    

    # info
    stim_info = pd.read_csv(nsd_path+'nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    cocoId = stim_info['cocoId']

    # stim - imgs
    h5 = h5py.File(nsd_path+'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    imgs = h5['imgBrick']
    # stim - caps
    f = open(nsd_path+f'nsddata_stimuli/stimuli/nsd/annotations/nsd_captions.json', 'r')
    cap = json.load(f)
    f.close()

    cs = []
    zs = []
    # Sample
    for s in tqdm(range(73000)):
        print(f"Now processing image {s:06}")
        prompt = cap[str(cocoId[s])]
        img = imgs[s]

        init_image = load_img_from_arr(img,resolution).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size).to(torch_dtype)

        init_latent = vae.encode(init_image).latent_dist.sample() * vae.config.scaling_factor

        with torch.no_grad():
            with precision_scope("cuda"):
                # with model.ema_scope():
                cond_input = tokenizer(prompt, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True)
                c = text_encoder(cond_input.input_ids.to('cuda'))[0].mean(axis=0).unsqueeze(0)


        init_latent = init_latent.cpu().detach().numpy().flatten()
        c = c.cpu().detach().numpy().flatten()

        cs.append(c)
        zs.append(init_latent)
    
    cs=np.stack(cs)
    f = h5py.File(f'{output_dir_z}c_all.hdf5', 'w')
    f.create_dataset('data', data=cs)
    f.close()
    del cs
    zs=np.stack(zs)
    f = h5py.File(f'{output_dir_z}z_all.hdf5', 'w')
    f.create_dataset('data', data=zs)
    f.close()

if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
