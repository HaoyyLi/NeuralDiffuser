# %%
import argparse
import os
import sys
from tqdm import tqdm
project_path = '/opt/data/private/src/fMRI/Decoding/NeuralDiffuser'
sys.path.append(project_path)
sys.path.append(project_path+'/src_icml24')
import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import torch
from torch.utils.data import DataLoader
from modules.brain_model.BrainModel import *
from modules.brain_model import utils
from sklearn.metrics import r2_score
from modules.brain_model.models import VersatileDiffusionPriorNetwork, BrainDiffusionPrior

from accelerate import Accelerator
accelerator = Accelerator(split_batches=True,mixed_precision='fp16')  
print("PID of this process =",os.getpid())
print = accelerator.print # only print if local_rank=0
device = accelerator.device
print("device:",device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--method",     
    type=str,
    default='neuraldiffuser',
)

parser.add_argument(
    "--subject",     
    type=str,
    default='subj01',  
    help="",
)
parser.add_argument(
    "--fetch_gt",     
    type=str,
    default='c',    
    help="",
)
parser.add_argument(
    "--fetch_fmri",   
    type=str,
    default='nsdgeneral',     
    help="",
)



opt = parser.parse_args()

fetch_gt=opt.fetch_gt 
fetch_fmri=opt.fetch_fmri
subject=opt.subject
method=opt.method

backend = set_backend("numpy", on_error="warn")

preprocess_pipeline = StandardScaler(with_mean=True, with_std=True)
# =================================================================
mridir = f'{project_path}/data/fmri/{subject}/{fetch_fmri}/'
featdir = f'{project_path}/data/proj_feat/{subject}/{fetch_gt}/'
savedir_scores = f'{project_path}/scores/{subject}/{method}/{fetch_gt}/'
savedir_model = f'{project_path}/models/{subject}/{method}/{fetch_gt}/'
os.makedirs(savedir_scores, exist_ok=True)
os.makedirs(savedir_model, exist_ok=True)


X = np.load(f'{mridir}/betas_tr.npy').astype("float32")
X_te = np.load(f'{mridir}/betas_ave_te.npy').astype("float32")
Y = np.load(f'{featdir}/tr_each.npy').astype("float32").reshape([X.shape[0],-1])
Y_te = np.load(f'{featdir}/te_ave.npy').astype("float32").reshape([X_te.shape[0],-1])

print(f'Now making decoding model for... {subject}:  {fetch_fmri}, {fetch_gt}')
print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')

X = preprocess_pipeline.fit_transform(X)
X_te = preprocess_pipeline.fit_transform(X_te)
preprocess_pipeline.fit(Y)
Y_mean = preprocess_pipeline.mean_
Y_std = preprocess_pipeline.scale_


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X

    def __getitem__(self, idx):
        
        return self.X[idx], self.Y[idx]
    # 返回数据的个数
    def __len__(self):
        return np.shape(self.Y)[0]

max_lr=3e-4
epochs = 100
batch_size = 32
num_devices = 1
num_epochs = 100
lr_scheduler_type = 'cycle'
prior_mult = 0.030
mixup_pct = .33
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

trn_loader = DataLoader(MyDataset(X,Y), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(MyDataset(X_te,Y_te), batch_size=batch_size, shuffle=False)

num_voxels = X.shape[1]
out_dim = Y.shape[1]
# model = RidgeRegression(num_voxels, out_dim, alpha=alpha).cuda()
clip_size = 768
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim,clip_size=clip_size,use_projector=True)
voxel2clip = BrainNetwork(**voxel2clip_kwargs).cuda()

# setup prior network
out_dim = clip_size
depth = 6
dim_head = 64
heads = clip_size//64 # heads * dim_head = 12 * 64 = 768

timesteps = 100
prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = 77,
        learned_query_mode="pos_emb"
    ).cuda()
print("prior_network loaded")

# custom version that can fix seeds
diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
).cuda()

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
# optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr, eps=1e-5)

global_batch_size = batch_size * num_devices
if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(num_epochs*len(trn_loader)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(num_epochs*len(trn_loader))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )

diffusion_prior, optimizer, trn_loader, val_loader, lr_scheduler = accelerator.prepare(
diffusion_prior, optimizer, trn_loader, val_loader, lr_scheduler
)


def loss_fn(prev, target, perm=None, betas=None, select=None):
    loss_mse = torch.nn.functional.mse_loss(prev, target)
    p_norm = nn.functional.normalize(prev, dim=-1)
    y_norm = nn.functional.normalize(target, dim=-1)
    loss_nce = utils.mixco_nce(
                p_norm,
                y_norm,
                temp=.006, 
                perm=perm, betas=betas, select=select)
    return loss_mse, loss_nce

best_loss = 1e6
for epoch in (range(num_epochs)):
    diffusion_prior.train()
    trn_loss = 0
    val_loss = 0
    sims_base = 0
    fwd_percent_correct = 0
    bwd_percent_correct = 0
    v_sims_base = 0
    v_fwd_percent_correct = 0
    v_bwd_percent_correct = 0
    loop = tqdm(enumerate(trn_loader), total =len(trn_loader))
    for i, (voxel, clip_target) in loop:
        with torch.cuda.amp.autocast():
            # torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            voxel = voxel.cuda()
            clip_target = clip_target.cuda()

            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel, s_thresh=0.5)

            clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
            clip_voxels = clip_voxels.view(len(voxel),-1,clip_size)
            clip_target = clip_target.view(len(voxel),-1,clip_size)

            loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
            aligned_clip_voxels /= diffusion_prior.module.image_embed_scale if distributed else diffusion_prior.image_embed_scale

            clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if epoch < int(mixup_pct * num_epochs):
                loss_nce = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006, 
                    perm=perm, betas=betas, select=select)
            else:
                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)
            
            loss = loss_nce + (prior_mult * loss_prior)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            
            # print(f"loss:{loss.item():04}")
            trn_loss += loss.item()

            sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item()
            
            labels = torch.arange(len(clip_target_norm)).cuda()
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm,clip_target_norm), labels, k=1).item()
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
            lr_scheduler.step()
            #更新信息
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss_prior=loss_prior.item(), loss_nce=loss_nce.item(), loss=loss.item(), trn_loss=trn_loss/(i+1), lr=optimizer.param_groups[0]['lr'])
    trn_loss = trn_loss/(i+1)
    sims_base = sims_base/(i+1)
    fwd_percent_correct = fwd_percent_correct/(i+1)
    bwd_percent_correct = bwd_percent_correct/(i+1)

    diffusion_prior.eval()
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    scores = []
    for i, (voxel, clip_target) in loop:
        voxel = voxel.cuda()
        clip_target = clip_target.cuda()

        clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
        clip_voxels = clip_voxels.view(len(voxel),-1,clip_size)
        clip_target = clip_target.view(len(voxel),-1,clip_size)

        v_loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
        aligned_clip_voxels /= diffusion_prior.module.image_embed_scale if distributed else diffusion_prior.image_embed_scale

        clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

        if epoch < int(mixup_pct * num_epochs):
            v_loss_nce = utils.mixco_nce(
                clip_voxels_norm,
                clip_target_norm,
                temp=.006)
        else:
            epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
            v_loss_nce = utils.soft_clip_loss(
                clip_voxels_norm,
                clip_target_norm,
                temp=epoch_temp)
        
        v_loss = v_loss_nce + (prior_mult * v_loss_prior)
        
        scores.append(aligned_clip_voxels.flatten(1).detach().cpu().numpy())

        val_loss += v_loss.item()

        v_sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item()
        
        labels = torch.arange(len(clip_target_norm)).cuda()
        v_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm,clip_target_norm), labels, k=1).item()
        v_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

        #更新信息
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(v_loss_prior=v_loss_prior.item(), v_loss_nce=v_loss_nce.item(), v_loss=v_loss.item(), val_loss=val_loss/(i+1))
    val_loss = val_loss/(i+1)
    v_sims_base = v_sims_base/(i+1)
    v_fwd_percent_correct = v_fwd_percent_correct/(i+1)
    v_bwd_percent_correct = v_bwd_percent_correct/(i+1)
    print(f"trn_loss:{trn_loss:3.4}\tval_loss:{val_loss:3.4}")
    print(f'sims_base:{sims_base:3.4}\tv_sims_base:{v_sims_base:3.4}')
    print(f'fwd_percent_correct:{fwd_percent_correct:3.4}\tv_fwd_percent_correct:{v_fwd_percent_correct:3.4}')
    print(f'bwd_percent_correct:{bwd_percent_correct:3.4}\tv_bwd_percent_correct:{v_bwd_percent_correct:3.4}')
    scores=np.concatenate(scores)
    scores1 = scores*Y_std+Y_mean
    if (epoch+1)%5==0:
        rs = correlation_score(Y_te.T,scores1.T)
        r2 = r2_score(Y_te, scores1)
        print(f'Prediction accuracy is: {np.mean(rs):3.3}\tr2_score is: {r2:3.3}')
    scores2 = preprocess_pipeline.fit_transform(scores)
    scores2 = scores2*Y_std+Y_mean
    if (epoch+1)%5==0:
        rs = correlation_score(Y_te.T,scores2.T)
        r2 = r2_score(Y_te, scores2)
        print(f'Prediction accuracy is: {np.mean(rs):3.3}\tr2_score is: {r2:3.3}')
    np.save(f'{savedir_scores}/{fetch_fmri}_1.npy',scores1)
    np.save(f'{savedir_scores}/{fetch_fmri}_2.npy',scores2)
    torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion_prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trn_loss': trn_loss,
            'val_loss': val_loss,
            }, f'{savedir_model}/{fetch_fmri}.pth')


'''
python train.py --method neuraldiffuser --fetch_gt c --fetch_fmri nsdgeneral --subject subj01
'''
# %%
