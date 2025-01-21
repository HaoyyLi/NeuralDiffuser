import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import h5py
from tqdm import tqdm
import webdataset as wds

from accelerate import Accelerator

torch.set_num_threads(2)
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--subj", 
    type=int,  
)
args = parser.parse_args()

data_path='MindEyeV2/mindeyev2'
subj=args.subj
batch_size= BATCH_SIZE
use_prior = True
clip_scale=1 
blurry_recon=True
n_blocks=4 
hidden_dim=4096 
seq_len=1
new_test=True
seed = 42
use_cont=True

model_name=f"pretrained_subj0{subj}_40sess_ext1_z_{hidden_dim}"
# seed all random functions
utils.seed_everything(seed)

num_sessions = 40
print(f"Training with {num_sessions} sessions")
train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
print(train_url)

def my_split_by_node(urls): return urls
train_data = wds.WebDataset(train_url,resampled=False,nodesplitter=my_split_by_node)\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
train_dl = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=False, drop_last=False, pin_memory=True)
print(f"Loaded train dl for subj{subj}!\n")

voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

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
else: # using larger test set from after full dataset released
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
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")



# Load 73k NSD images
f = h5py.File(f'{data_path}/z_all.hdf5', 'r')
vae_embeds = f['data'] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)

# images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD text_embeds to cpu!", vae_embeds.shape)

# Prep train voxels and indices of test images
train_images_idx = []
train_voxels_idx = []
for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
    train_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    train_voxels_idx = np.append(train_images_idx, behav[:,0,5].cpu().numpy())
    train_images_idx = np.append(train_images_idx, behav[:,0,0].cpu().numpy())
train_images_idx = train_images_idx.astype(int)
train_voxels_idx = train_voxels_idx.astype(int)

print(train_i, len(train_voxels), len(train_images_idx), len(np.unique(train_images_idx)))

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))

class fMRIDecoder_Z(nn.Module):
    def __init__(self):
        super(fMRIDecoder_Z, self).__init__()
    def forward(self, x):
        return x
        
model = fMRIDecoder_Z()

class VoxelEncoder(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, voxel_sizes, shared_embed_size, seq_len): 
        super(VoxelEncoder, self).__init__()
        self.shared_embed_size = shared_embed_size
        # self.linears = torch.nn.ModuleList([
        #         torch.nn.Linear(voxel_size, shared_embed_size) for voxel_size in voxel_sizes
        #     ])
        self.linears = torch.nn.ModuleList([
                torch.nn.Sequential(
                torch.nn.Linear(voxel_size, shared_embed_size, bias=False),
                torch.nn.LayerNorm(shared_embed_size),
                torch.nn.SiLU(inplace=True),
                torch.nn.Dropout(0.5)
            ) for voxel_size in voxel_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out
        
model.voxelenc = VoxelEncoder(voxel_sizes=[num_voxels], shared_embed_size=hidden_dim, seq_len=seq_len)

from models import Voxel2StableDiffusionModel
voxel2sd = Voxel2StableDiffusionModel(h=hidden_dim, n_blocks=n_blocks, use_cont=use_cont)
voxel2sd.to(device)
voxel2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2sd)
utils.count_params(voxel2sd)
model.voxel2sd = voxel2sd

# Load pretrained model ckpt
tag='last'
outdir = os.path.abspath(f'../train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")

# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

utils.seed_everything(seed)
# 不打乱顺序
import pandas as pd
test_unique_idx = pd.DataFrame(test_images_idx)[0].unique()
scores_z = []
scores_c = []
minibatch_size = 100
# with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
    for batch in tqdm(range(0,len(test_unique_idx), minibatch_size)):
        uniq_imgs = test_unique_idx[batch:batch+minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx==uniq_img)[0]
            if len(locs)==1:
                locs = locs.repeat(3)
            elif len(locs)==2:
                locs = locs.repeat(2)[:3]
            assert len(locs)==3
            if voxel is None:
                voxel = test_voxels[None,locs] # 1, num_image_repetitions, num_voxels
            else:
                voxel = torch.vstack((voxel, test_voxels[None,locs]))
        voxel = voxel.to(device)
        for rep in range(3):
            voxel_ridge = model.voxelenc(voxel[:,[rep]],0) # 0th index of subj_list
            if use_cont:
                blurry_image_enc0, transformer_feats = model.voxel2sd(voxel_ridge, return_transformer_feats=True)
            else:
                blurry_image_enc0 = model.voxel2sd(voxel_ridge)
            if rep==0:
                blurry_image_enc = blurry_image_enc0
            else:
                blurry_image_enc += blurry_image_enc0
        blurry_image_enc /= 3

        image_enc_pred = blurry_image_enc

        scores_z.append(image_enc_pred.flatten(1).detach().cpu().numpy())

score_zs=np.concatenate(scores_z)

Y_tr = vae_embeds[np.unique(train_images_idx)]

preprocess_pipeline = StandardScaler(with_mean=True, with_std=True)
preprocess_pipeline.fit(Y_tr)
Y_mean = preprocess_pipeline.mean_
Y_std = preprocess_pipeline.scale_
# 不打乱顺序
import pandas as pd
test_unique_idx = pd.DataFrame(test_images_idx)[0].unique()
scores_z = []
scores_c = []
minibatch_size = 100
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
# with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
    for batch in tqdm(range(0,len(test_unique_idx), minibatch_size)):
        uniq_imgs = test_unique_idx[batch:batch+minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx==uniq_img)[0]
            if len(locs)==1:
                locs = locs.repeat(3)
            elif len(locs)==2:
                locs = locs.repeat(2)[:3]
            assert len(locs)==3
            if voxel is None:
                voxel = test_voxels[None,locs] # 1, num_image_repetitions, num_voxels
            else:
                voxel = torch.vstack((voxel, test_voxels[None,locs]))
        voxel = voxel.to(device)

        voxel = voxel.mean(dim=1).unsqueeze(1)
        voxel_ridge = model.voxelenc(voxel,0) # 0th index of subj_list
        if use_cont:
            blurry_image_enc, transformer_feats = model.voxel2sd(voxel_ridge, return_transformer_feats=True)
        else:
            blurry_image_enc = model.voxel2sd(voxel_ridge)
        image_enc_pred = blurry_image_enc

        scores_z.append(image_enc_pred.flatten(1).detach().cpu().numpy())
score_zs=np.concatenate(scores_z)

score_zs2 = preprocess_pipeline.fit_transform(score_zs)
score_zs2 = score_zs2*Y_std+Y_mean

savedir_scores = f'../scores_all/subj0{subj}/{model_name}/'
os.makedirs(savedir_scores, exist_ok=True)
np.save(f'{savedir_scores}/nsdgeneral.npy',score_zs2)