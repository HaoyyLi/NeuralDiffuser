# %%
import argparse
import os
import sys
from tqdm import tqdm
project_path = '/home/src/NeuralDiffuser'
sys.path.append(project_path)
sys.path.append(project_path+'/src')
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
from modules.brain_model.models import Voxel2StableDiffusionModel

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
    help="",
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
    default='z',
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
use_stim=opt.use_stim
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
# %%
# X = preprocess_pipeline.fit_transform(X)
# X_mean = preprocess_pipeline.mean_
# X_std = preprocess_pipeline.scale_
# X_te = preprocess_pipeline.fit_transform(X_te)
# X_te_mean = preprocess_pipeline.mean_
# X_te_std = preprocess_pipeline.scale_
preprocess_pipeline.fit(Y)
Y_mean = preprocess_pipeline.mean_
Y_std = preprocess_pipeline.scale_
# Y_te_ = preprocess_pipeline.fit_transform(Y_te)
# Y_te_mean = preprocess_pipeline.mean_
# Y_te_std = preprocess_pipeline.scale_
# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X

    def __getitem__(self, idx):
        
        return self.X[idx], self.Y[idx]
    # 返回数据的个数
    def __len__(self):
        return np.shape(self.Y)[0]
# %%
batch_size = 32
# trn_loader = DataLoader(MyDataset(X,Y), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(MyDataset(X_te,Y_te), batch_size=batch_size, shuffle=False)
# load model
autoencoder_name='autoencoder_subj01_4x_locont_no_reconst'
num_voxels = X.shape[1]
outdir = f'{project_path}/models/{subject}/{method}/autoencoder_{subject}_4x_locont_no_reconst'
ckpt_path = os.path.join(outdir, f'epoch120.pth')

checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']

voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)

voxel2sd.load_state_dict(state_dict,strict=False)
voxel2sd.eval()
voxel2sd.to(device)
print("Loaded low-level model!")


voxel2sd.eval()
loop = tqdm(enumerate(val_loader), total=len(val_loader))
scores = []
for i, (voxel, _) in loop:
    voxel = voxel.cuda()
    with torch.no_grad():
        ae_preds = voxel2sd(voxel.float())  
    scores.append(ae_preds.flatten(1).detach().cpu().numpy())

scores=np.concatenate(scores)
scores1 = scores*Y_std+Y_mean
rs = correlation_score(Y_te.T,scores1.T)
r2 = r2_score(Y_te, scores1)
print(f'Prediction accuracy is: {np.mean(rs):3.3}\tr2_score is: {r2:3.3}')
scores2 = preprocess_pipeline.fit_transform(scores)
scores2 = scores2*Y_std+Y_mean
rs = correlation_score(Y_te.T,scores2.T)
r2 = r2_score(Y_te, scores2)
print(f'Prediction accuracy is: {np.mean(rs):3.3}\tr2_score is: {r2:3.3}')

np.save(f'{savedir_scores}/{fetch_fmri}_2_{method}_1.npy',scores1)
np.save(f'{savedir_scores}/{fetch_fmri}_2_{method}_2.npy',scores2)