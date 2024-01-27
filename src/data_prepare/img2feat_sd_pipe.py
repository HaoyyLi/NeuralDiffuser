import argparse, os
import sys
root = '/home/src/NeuralDiffuser'
sys.path.append(root)
sys.path.append(root+'/src')
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from modules.nsd_access import NSDAccess
from PIL import Image
from modules.nsd_access import NSDAccess
import json
import h5py
import pandas as pd
import scipy

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel


def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    # Set Parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    subject = opt.subject
    # imgidx = opt.imgidx
    gpu = opt.gpu
    resolution = 512        # cvpr--320  minddiffusion--512
    batch_size = 1
    max_length = 77
    torch_dtype = torch.float32
    model_dtype = 'fp32'
    # ddim_steps = 50
    # ddim_eta = 0.0
    # strength = 0.8
    # scale = 5.0
    nsd_path = '/home/dataset/nsd/'
    output_dir_z = f'{root}/data/feature/{subject}/z/'
    output_dir_c = f'{root}/data/feature/{subject}/c/'
    
    nsda = NSDAccess(nsd_path)
    torch.cuda.set_device(gpu)
    os.makedirs(output_dir_z, exist_ok=True)
    os.makedirs(output_dir_c, exist_ok=True)

    # Load moodels
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    model_path_diffusion = "/home/huggingface/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", revision=model_dtype, torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_path_diffusion, subfolder="tokenizer", revision=model_dtype, torch_dtype=torch_dtype)
    text_encoder = CLIPTextModel.from_pretrained(model_path_diffusion, subfolder="text_encoder", revision=model_dtype, torch_dtype=torch_dtype)

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    vae.to(device)
    text_encoder.to(device)
    

    # info
    stim_info = pd.read_csv(nsd_path+'nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    nsd_expdesign = scipy.io.loadmat(nsd_path+'nsddata/experiments/nsd/nsd_expdesign.mat')
    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix'] -1 
    behs = pd.DataFrame()
    for i in tqdm(range(1,41)):
        beh = nsda.read_behavior(subject=subject,
                                session_index=i)
        behs = pd.concat((behs,beh))
    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1
    cocoId = stim_info['cocoId']

    # stim - imgs
    h5 = h5py.File(nsd_path+'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    imgs = h5['imgBrick']
    # stim - caps
    f = open(nsd_path+f'nsddata_stimuli/stimuli/nsd/annotations/{subject}_nsd_captions.json', 'r')
    cap = json.load(f)
    f.close()

    # Sample
    for s in tqdm(stims_unique):
        # print(f"Now processing image {s:06}")
        prompt = cap[str(cocoId[s])]
        img = imgs[s]

        init_image = load_img_from_arr(img,resolution).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size).to(torch_dtype)

        init_latent = vae.encode(init_image).latent_dist.sample()# * vae.config.scaling_factor
        # init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        with torch.no_grad():
            with precision_scope("cuda"):
                # with model.ema_scope():
                cond_input = tokenizer(prompt, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True)
                c = text_encoder(cond_input.input_ids.to('cuda'))[0].mean(axis=0).unsqueeze(0)


        init_latent = init_latent.cpu().detach().numpy().flatten()
        c = c.cpu().detach().numpy().flatten()
        np.save(f'{output_dir_z}{s:06}.npy',init_latent)
        np.save(f'{output_dir_c}{s:06}.npy',c)


if __name__ == "__main__":
    main()

'''
python img2feat_sd_pipe.py  --subject subj01 --gpu 0
'''

