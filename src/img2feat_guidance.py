# %%
import argparse, os
import sys
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from PIL import Image

import json
import h5py
import pandas as pd
import scipy

from modules.utils.guidance_function import get_model, get_feature_maps, transform_for_CLIP

def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


# %%
def main():

    seed_everything(42)

    # imgidx = opt.imgidx
    gpu = 0
    target_layers = ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12']

    nsd_path = ''
    output_dir = f'MindEyeV2/mindeyev2/'

    torch.cuda.set_device(gpu)
    os.makedirs(output_dir, exist_ok=True)

    # Load moodels
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    model = get_model()
    transformer = transform_for_CLIP

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # stim - imgs
    h5 = h5py.File(nsd_path+'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    imgs = h5['imgBrick']
    # %%
    feature_maps = {layer:[] for layer in target_layers}
    # Sample
    for s in tqdm(range(73000)):
        # print(f"Now processing image {s:06}")
        img = imgs[s]

        init_image = transformer(img).unsqueeze(0).to(device)
        # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        
        with torch.no_grad():
            with precision_scope("cuda"):
                batch_feature_maps = get_feature_maps(model, init_image, target_layers )

        
        for k in batch_feature_maps.keys():
            temp = batch_feature_maps[k].cpu().detach().numpy()
            if k != 'VisionTransformer-1':
                temp = temp.transpose((1, 0, 2)).reshape(1,-1)
            feature_maps[k].append(temp)
            # print(feature_maps[k].shape)
    
    for k in feature_maps:
        y=np.stack(feature_maps[k])
        f = h5py.File(f"{output_dir}g_{k.split('-')[1]}_all.hdf5", 'w')
        f.create_dataset('data', data=y)
        f.close()
        del y

if __name__ == "__main__":
    torch.set_num_threads(2)
    main()

'''
python img2feat_guidance.py --subject subj01 --gpu 0

'''