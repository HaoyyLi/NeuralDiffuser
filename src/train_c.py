# %%
import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# # SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# sys.path.append('generative_models/')
# import sgm
# from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
torch.set_num_threads(2)
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

# %%
### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1


NUM_GPUS=1  # Set to equal gres=gpu:#!
BATCH_SIZE=42 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
GLOBAL_BATCH_SIZE=BATCH_SIZE * NUM_GPUS

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
if utils.is_interactive(): # set batch size here if using interactive notebook instead of submitting job
    global_batch_size = batch_size = 8
else:
    global_batch_size = GLOBAL_BATCH_SIZE
    batch_size = int(GLOBAL_BATCH_SIZE) // num_devices

# %%
print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "--subj", 
    type=int,
)

parser.add_argument(
    "--multi_subject",action='store_true',
    help="if multi_subject",
)

parser.add_argument(
    "--wandb_log",action='store_true',
    help="if wandb_log",
)


args = parser.parse_args()
subj=args.subj
wandb_log=args.wandb_log
multi_subject = args.multi_subject

data_path='MindEyeV2/mindeyev2'


batch_size= BATCH_SIZE
max_lr=3e-4 
mixup_pct=.33 
num_epochs=150 
use_prior = True
prior_scale=30 
clip_scale=1 
blur_scale=.5 
blurry_recon=False
use_image_aug=False 
n_blocks=4 
hidden_dim=1024 
num_sessions=40 
ckpt_interval=999 
ckpt_saving=True


visualize_prior=False
resume_from_ckpt=False
wandb_project="stability"
new_test=True
seq_past=0
seq_future=0
lr_scheduler_type = 'cycle'
seed = 42

if multi_subject:
    # multisubject pretraining
    model_name = f"multisubject_subj0{subj}_ext1_c_{hidden_dim}_mlp"
    multisubject_ckpt=None
else:
    # singlesubject finetuning
    model_name=f"pretrained_subj0{subj}_40sess_ext1_c_{hidden_dim}_mlp"
    multisubject_ckpt=f'../train_logs/multisubject_subj0{subj}_ext1_c_{hidden_dim}_mlp'
# %%  
# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../train_logs/{model_name}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
    
if use_image_aug or blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )
    
if multi_subject:
    subj_list = np.arange(1,9)
    subj_list = subj_list[subj_list != subj]
else:
    subj_list = [subj]

print("subj_list", subj_list, "num_sessions", num_sessions)

# %% [markdown]
# # Prep data, models, and dataloaders

# %% [markdown]
# ### Creating wds dataloader, preload betas and all 73k possible images

# %%
def my_split_by_node(urls): return urls
num_voxels_list = []

if multi_subject:
    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750*40) // num_devices 
else:
    num_samples_per_epoch = (750*num_sessions) // num_devices 

print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
batch_size = batch_size // len(subj_list)

num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)

# %%
train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
for s in subj_list:
    print(f"Training with {num_sessions} sessions")
    if multi_subject:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
    print(train_url)
    
    train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

# Validate only on one subject
if multi_subject: 
    subj = subj_list[0] # cant validate on the actual held out person so picking first in subj_list
if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
elif new_test: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
print(test_url)
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

seq_len = seq_past + 1 + seq_future
print(f"currently using {seq_len} seq_len (chose {seq_past} past behav and {seq_future} future behav)")

# %%
# Load 73k NSD images
f = h5py.File(f'{data_path}/c_all.hdf5', 'r')
images = f['data'] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)
# images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD text_embeds to cpu!", images.shape)

# %% [markdown]
# ## Load models

clip_seq_dim = 77
clip_emb_dim = 768

class fMRIDecoder_Z(nn.Module):
    def __init__(self):
        super(fMRIDecoder_Z, self).__init__()
    def forward(self, x):
        return x
        
model = fMRIDecoder_Z()

class VoxelEncoder(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, voxel_sizes, shared_embed_size, hidden_size, seq_len): 
        super(VoxelEncoder, self).__init__()
        self.shared_embed_size = shared_embed_size
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(voxel_size, shared_embed_size) for voxel_size in voxel_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out
        
model.voxelenc = VoxelEncoder(voxel_sizes=num_voxels_list, shared_embed_size=hidden_dim, seq_len=seq_len)
utils.count_params(model.voxelenc)
utils.count_params(model)

from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, seq_len=seq_len, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          clip_scale=clip_scale)
utils.count_params(model.backbone)
utils.count_params(model)

if use_prior:
    from mindeye_models import *

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb"
        )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    
    utils.count_params(model.diffusion_prior)
    utils.count_params(model)

# %%
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

opt_grouped_parameters = [
    {'params': [p for n, p in model.voxelenc.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
if use_prior:
    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ])

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible voxelenc layer that will otherwise error
        state_dict.pop('voxelenc.linears.0.0.weight',None)
        state_dict.pop('voxelenc.linears.0.1.weight',None)
        state_dict.pop('voxelenc.linears.0.1.bias',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)

# %% [markdown]
# # Weights and Biases

# %%
if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'mindeye_mlp'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_sessions": num_sessions,
      "num_params": num_params,
      "clip_scale": clip_scale,
      "prior_scale": prior_scale,
      "blur_scale": blur_scale,
      "use_image_aug": use_image_aug,
      "max_lr": max_lr,
      "mixup_pct": mixup_pct,
      "num_samples_per_epoch": num_samples_per_epoch,
      "num_test": num_test,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_url": train_url,
      "test_url": test_url,
    }
    print("wandb_config:\n",wandb_config)
    print("wandb_id:",model_name)
    wandb.init(
        id=model_name,
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False

# %% [markdown]
# # Main

# %%
epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()

# %%
# load multisubject stage1 ckpt if set
if multisubject_ckpt is not None and not resume_from_ckpt:
    load_ckpt("last",outdir=multisubject_ckpt,load_lr=False,load_optimizer=False,load_epoch=False,strict=False,multisubj_loading=True)

# %%
# load saved ckpt model weights into current model
if resume_from_ckpt:
    load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True)
elif wandb_log:
    if wandb.run.resumed:
        load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True)

# %%
train_dls = [train_dl[f'subj0{s}'] for s in subj_list]

model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals

# %%
print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.
    
    recon_cossim = 0.
    test_recon_cossim = 0.
    recon_mse = 0.
    test_recon_mse = 0.

    loss_clip_total = 0.
    test_loss_clip_total = 0.
    
    loss_prior_total = 0.
    test_loss_prior_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

    # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
    voxel_iters = {} # empty dict because diff subjects have differing # of voxels
    # image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), clip_seq_dim, clip_emb_dim).float()
    annot_iters = {}
    perm_iters, betas_iters, select_iters = {}, {}, {}
    for s, train_dl in enumerate(train_dls):
        with torch.cuda.amp.autocast(dtype=data_type):
            loop = tqdm(enumerate(train_dl), total=num_iterations_per_epoch, disable=(local_rank!=0))
            for iter, (behav0, past_behav0, future_behav0, old_behav0) in loop: 
                # OOM edit
                idx = behav0[:,0,0].cpu().long().numpy()
                if len(np.unique(idx)) == len(idx):
                    argidx = idx.argsort()
                    image0 = images[np.sort(idx)]
                    image0[argidx,:] = image0[range(len(argidx)),:]
                else:
                    image0 = np.stack([images[i] for i in idx])
                image0 = torch.from_numpy(image0).float().to("cpu").to(data_type)

                # image0 = images[behav0[:,0,0].cpu().long()].float()

                image0 = image0.view(len(image0),-1, clip_emb_dim)
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0
                
                voxel0 = voxels[f'subj0{subj_list[s]}'][behav0[:,0,5].cpu().long()]
                voxel0 = torch.Tensor(voxel0)
                
                if seq_len==1:
                    voxel0 = voxel0.unsqueeze(1)
                else:
                    if seq_past>0:
                        past_behavior = past_behav0[:,:(seq_past),5].cpu().long()
                        past_voxel0 = voxels[f'subj0{subj_list[s]}'][past_behavior]
                        past_voxel0[past_behavior==-1] = voxel0[torch.where(past_behavior==-1)[0]] # replace invalid past voxels 
                        past_voxel0 = torch.Tensor(past_voxel0)

                        # if shared1000, then you need to mask it out 
                        for p in range(seq_past):
                            mask = (past_behav0[:,p,-1] == 1) # [16,] bool
                            index = torch.nonzero(mask.cpu()).squeeze()
                            past_voxel0[index,p,:] = torch.zeros_like(past_voxel0[index,p,:])

                    if seq_future>0:
                        future_behavior = future_behav0[:,:(seq_future),5].cpu().long()
                        future_voxel0 = voxels[f'subj0{subj_list[s]}'][future_behavior]
                        future_voxel0[future_behavior==-1] = voxel0[torch.where(future_behavior==-1)[0]] # replace invalid past voxels 
                        future_voxel0 = torch.Tensor(future_voxel0)

                        # if shared1000, then you need to mask it out 
                        for p in range(seq_future):
                            mask = (future_behav0[:,p,-1] == 1) # [16,] bool
                            index = torch.nonzero(mask.cpu()).squeeze()
                            future_voxel0[index,p,:] = torch.zeros_like(future_voxel0[index,p,:])

                    # concatenate current timepoint with past/future
                    if seq_past > 0 and seq_future > 0:
                        voxel0 = torch.cat((voxel0.unsqueeze(1), past_voxel0), axis=1)
                        voxel0 = torch.cat((voxel0, future_voxel0), axis=1)
                    elif seq_past > 0:
                        voxel0 = torch.cat((voxel0.unsqueeze(1), past_voxel0), axis=1)
                    else:
                        voxel0 = torch.cat((voxel0.unsqueeze(1), future_voxel0), axis=1)

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                    betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                    select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                if iter >= num_iterations_per_epoch-1:
                    break

    # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
    loop = tqdm(range(num_iterations_per_epoch), total =num_iterations_per_epoch, disable=(local_rank!=0))
    for train_i in loop:
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss=0.

            voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
            image = image_iters[train_i].detach()
            image = image.to(device)

            if use_image_aug: 
                image = img_augment(image)

            clip_target = image
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            voxel_voxelenc_list = [model.voxelenc(voxel_list[si],si) for si,s in enumerate(subj_list)]
            voxel_voxelenc = torch.cat(voxel_voxelenc_list, dim=0)

            backbone, clip_voxels = model.backbone(voxel_voxelenc)

            if clip_scale>0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if use_prior:
                loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                loss_prior_total += loss_prior.item()
                loss_prior *= prior_scale
                loss += loss_prior

                recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                recon_mse += mse(prior_out, clip_target).item()

            if clip_scale>0:
                if epoch < int(mixup_pct * num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip


            if clip_scale>0:
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()
            
            #更新信息
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss_prior=loss_prior.item(), loss_clip=loss_clip.item(), loss=loss.item(), trn_loss=np.mean(losses[-(train_i+1):]), recon_cossim=recon_cossim / (train_i + 1), recon_mse=recon_mse / (train_i + 1), lr=optimizer.param_groups[0]['lr'])


    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            loop = tqdm(enumerate(test_dl), total =1, disable=(local_rank!=0))
            for test_i, (behav, past_behav, future_behav, old_behav) in loop:  
                # all test samples should be loaded per batch such that test_i should never exceed 0
                assert len(behav) == num_test

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
                    
                    if seq_len==1:
                        voxel = voxel.unsqueeze(1)
                    else:
                        if seq_past>0:
                            past_behavior = past_behav[:,:(seq_past),5].cpu().long()
                            past_voxels = voxels[f'subj0{subj}'][past_behavior]
                            if torch.any(past_behavior==-1).item(): # remove invalid voxels (-1 if there is no timepoint available)
                                past_voxels[torch.where(past_behavior==-1)[0]] = 0

                        if seq_future>0:
                            future_behavior = future_behav[:,:(seq_future),5].cpu().long()
                            future_voxels = voxels[f'subj0{subj}'][future_behavior]                    
                            if torch.any(future_behavior==-1).item(): # remove invalid voxels (-1 if there is no timepoint available)
                                future_voxels[torch.where(future_behavior==-1)[0]] = 0
                            
                        if seq_past > 0 and seq_future > 0:
                            voxel = torch.cat((voxel.unsqueeze(1), past_voxels), axis=1)
                            voxel = torch.cat((voxel, future_voxels), axis=1)
                        elif seq_past > 0:
                            voxel = torch.cat((voxel.unsqueeze(1), past_voxels), axis=1)
                        else:
                            voxel = torch.cat((voxel.unsqueeze(1), future_voxels), axis=1)

                    image = behav[:,0,0].cpu().long()

                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if len(locs)==1:
                            locs = locs.repeat(3)
                        elif len(locs)==2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs)==3
                        if test_image is None:
                            test_image = torch.from_numpy(images[im][None]).float().to("cpu").to(data_type).view(1,-1, clip_emb_dim)
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack((test_image, torch.from_numpy(images[im][None]).float().to("cpu").to(data_type).view(1,-1, clip_emb_dim)))
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))
                    # for im in unique_image:
                    #     locs = torch.where(im == image)[0]
                    #     if len(locs)==1:
                    #         locs = locs.repeat(3)
                    #     elif len(locs)==2:
                    #         locs = locs.repeat(2)[:3]
                    #     assert len(locs)==3
                    #     if test_image is None:
                    #         test_image = images[im][None]
                    #         test_voxel = voxel[locs][None]
                    #     else:
                    #         test_image = torch.vstack((test_image, images[im][None]))
                    #         test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                loss=0.
                            
                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(device)
                assert len(image) == 300

                # clip_target = clip_img_embedder(image.float())
                clip_target = image

                for rep in range(3):
                    voxel_voxelenc = model.voxelenc(voxel[:,rep],0) # 0th index of subj_list
                    backbone0, clip_voxels0 = model.backbone(voxel_voxelenc)
                    if rep==0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3

                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                
                if use_prior:
                    loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior
                    
                    if visualize_prior:
                        # now get unCLIP prediction without feeding it the image embed to get uncontaminated reconstruction
                        prior_out = model.diffusion_prior.p_sample_loop(backbone[random_samps].shape, 
                                        text_cond = dict(text_embed = backbone[random_samps]), 
                                        cond_scale = 1., timesteps = 20)

                        test_recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target[random_samps]).mean().item()
                        test_recon_mse += mse(prior_out, clip_target[random_samps]).item()
                        
                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                
                utils.check_loss(loss)                
                test_losses.append(loss.item())
                
                #更新信息
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss_prior=loss_prior.item(), loss_clip=loss_clip.item(), loss=loss.item(), test_loss=np.mean(test_losses[-(train_i+1):]), test_recon_cossim=test_recon_cossim / (test_i + 1), test_recon_mse=test_recon_mse / (test_i + 1))


            # if utils.is_interactive(): clear_output(wait=True)
            print("---")

            assert (test_i+1) == 1
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "test/recon_cossim": test_recon_cossim / (test_i + 1),
                "train/recon_mse": recon_mse / (train_i + 1),
                "test/recon_mse": test_recon_mse / (test_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
                }

            progress_bar.set_postfix(**logs)

            if wandb_log: wandb.log(logs)
            
    # Save model checkpoint and reconstruct
    if (ckpt_saving) and ((epoch+1) % ckpt_interval == 0):
        save_ckpt(f'last')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt(f'last')


# %%
plt.plot(losses)
plt.show()
plt.plot(test_losses)
plt.show()


