# %%
import argparse, os
import sys
root = '/opt/data/private/src/fMRI/Decoding/NeuralDiffuser'
sys.path.append(root)
sys.path.append(root+'/src_icml24')
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
    parser.add_argument(
        '--target_layers', 
        help='CLIP_layers', 
        default=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12','VisionTransformer-1'], 
        type=list
    )

    # Set Parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    subject = opt.subject
    # imgidx = opt.imgidx
    gpu = opt.gpu
    target_layers = opt.target_layers

    batch_size = 1
    # ddim_steps = 50
    # ddim_eta = 0.0
    # strength = 0.8
    # scale = 5.0
    nsd_path = '/opt/data/private/dataset/nsd/'
    output_dir = f'{root}/data/feature/{subject}/g/'

    nsda = NSDAccess(nsd_path)
    torch.cuda.set_device(gpu)
    os.makedirs(output_dir, exist_ok=True)

    # Load moodels
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    model = get_model()
    transformer = transform_for_CLIP

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

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

    # stim - imgs
    h5 = h5py.File(nsd_path+'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    imgs = h5['imgBrick']
    # %%
    img = imgs[0]
    init_image = transformer(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with precision_scope("cuda"):
            batch_feature_maps = get_feature_maps(model, init_image, target_layers )

    feature_maps = {layer:[] for layer in target_layers}
    for k in batch_feature_maps.keys():
        temp = batch_feature_maps[k].cpu().detach().numpy()
        if k != 'VisionTransformer-1':
            temp = temp.transpose((1, 0, 2)).reshape(1,-1)
        feature_maps[k].append(temp)
    for k in feature_maps.keys():
        feature_maps[k] = np.concatenate(feature_maps[k])
        # print(feature_maps[k].shape)

    # Sample
    for s in tqdm(stims_unique):
        # print(f"Now processing image {s:06}")
        img = imgs[s]

        init_image = transformer(img).unsqueeze(0).to(device)
        # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        
        with torch.no_grad():
            with precision_scope("cuda"):
                batch_feature_maps = get_feature_maps(model, init_image, target_layers )

        feature_maps = {layer:[] for layer in target_layers}
        for k in batch_feature_maps.keys():
            temp = batch_feature_maps[k].cpu().detach().numpy()
            if k != 'VisionTransformer-1':
                temp = temp.transpose((1, 0, 2)).reshape(1,-1)
            feature_maps[k].append(temp)
        for k in feature_maps.keys():
            feature_maps[k] = np.concatenate(feature_maps[k])
            # print(feature_maps[k].shape)
        y=np.concatenate([feature_maps[feature_map] for feature_map in feature_maps ],axis=1)
        np.save(f'{output_dir}{s:06}.npy',y)

if __name__ == "__main__":
    main()

'''
python img2feat_guidance.py --subject subj01 --gpu 0

'''