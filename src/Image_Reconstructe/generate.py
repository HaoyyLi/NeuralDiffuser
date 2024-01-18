import os, sys
nsd_path = '/opt/data/private/dataset/nsd'
project_path = '/opt/data/private/src/fMRI/Decoding/NeuralDiffuser'
model_path = "/opt/data/private/huggingface/stable-diffusion-v1-4"

sys.path.append(project_path)
sys.path.append(project_path+'/src_icml24')
import h5py
from PIL import Image
import scipy.io
import torch
import numpy as np
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.schedulers import DDIMScheduler
from modules.stable_diffusion_pipe.guided_diffusion import GuidedStableDiffusion
import argparse

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
    "--imgidx",     
    type=int,
    default=0,  
    help="",
)

parser.add_argument(
    "--guidance_scale",     
    type=float,
    default=500000,  
    help="",
)

parser.add_argument(
    "--guidance_strength",     
    type=float,
    default=0.2,  
    help="",
)

parser.add_argument(
    "--niter",     
    type=int,
    default=5,  
    help="",
)

parser.add_argument(
    "--ma",     
    type=bool,
    default=True,  
    help="",
)


opt = parser.parse_args()
subject=opt.subject
method=opt.method
imgidx = opt.imgidx
guidance_scale = opt.guidance_scale
guidance_strength = opt.guidance_strength
niter = opt.niter
ma = opt.ma

if guidance_scale==0:
    guidance_strength=0


gpu=0
seed=42
resolution = 512
model_type = 'fp32'
torch_dtype = torch.float32


outpath = f'{project_path}/outputs/{subject}/{method}/{guidance_scale}/{guidance_strength}'
os.makedirs(outpath, exist_ok=True)

if ma:
    score_zs = np.load(f'{project_path}/scores/{subject}/{method}/z/nsdgeneral_pretrain_mindeye_2.npy')
    score_cs = np.load(f'{project_path}/scores/{subject}/{method}/c/nsdgeneral_2.npy')
else:
    score_zs = np.load(f'{project_path}/scores/{subject}/{method}/z/nsdgeneral_pretrain_mindeye_1.npy')
    score_cs = np.load(f'{project_path}/scores/{subject}/{method}/c/nsdgeneral_1.npy')

# ===================== load GT Imgs =====================

# Load NSD information
nsd_expdesign = scipy.io.loadmat(f'{nsd_path}/nsddata/experiments/nsd/nsd_expdesign.mat')
# Note that mos of them are 1-base index!
# This is why I subtract 1
sharedix = nsd_expdesign['sharedix'] -1 

h5 = h5py.File(f'{nsd_path}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
Images_GT = h5.get('imgBrick')

stims_ave = np.load(f'{project_path}/data/fmri/{subject}/behs_stims_ave.npy')
tr_idx = np.zeros_like(stims_ave)
for idx, s in enumerate(stims_ave):
    if s in sharedix:
        tr_idx[idx] = 0
    else:
        tr_idx[idx] = 1

# ======================= load LDM model ======================
# Load Stable Diffusion Model
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=model_type, torch_dtype=torch_dtype)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=model_type, torch_dtype=torch_dtype)
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", revision=model_type, torch_dtype=torch_dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=model_type, torch_dtype=torch_dtype)
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision=model_type, torch_dtype=torch_dtype)

torch.cuda.set_device(gpu)
device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")

unet.to(device)
vae.to(device)
text_encoder.to(device)

model = GuidedStableDiffusion(
    vae = vae,
    unet = unet,
    scheduler=scheduler
)

# ===================== define DDIM ================
n_samples = 1
ddim_steps = 50
ddim_eta = 0.0
strength = 0.75
scale = 7.5
n_iter = 5
precision = 'autocast'
precision_scope = autocast if precision == "autocast" else nullcontext
batch_size = n_samples

assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
t_enc = int(strength * ddim_steps)
print(f"target t_enc is {t_enc} steps") 

# ================== load Guided model ======================
from modules.utils.guidance_function import get_model, get_feature_maps, clip_transform_reconstruction
from torch import nn
clip_model = get_model()
clip_model = clip_model.to(device)
def loss_fn(t, p):
    loss = nn.MSELoss()(t, p)
    return loss
Layers=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12']

guidance_dic = dict()
for layer in Layers:
    if ma:
        scores_g_path = f'{project_path}/scores/{subject}/{method}/g/{layer}/nsdgeneral_2.npy'
    else:
        scores_g_path = f'{project_path}/scores/{subject}/{method}/g/{layer}/nsdgeneral_1.npy'
    scores_g = np.load(scores_g_path)
    guidance_dic[layer] = scores_g

def cal_loss(image, guided_condition):
    x_samples = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    init_image = clip_transform_reconstruction(x_samples).to(device)
    CLIP_generated = get_feature_maps(clip_model, init_image, Layers )
    CLIP_generated = {k:CLIP_generated[k].permute((1, 0, 2)).reshape(1,-1) for k in CLIP_generated.keys()}
    loss = MSE_CLIP(target=guided_condition, generated=CLIP_generated)
    return -loss

def MSE_CLIP(target, generated):
    MSE = []
    VIT = []
    norms = []
    for layer in target.keys():
        norms.append(np.linalg.norm(target[layer]))

    norms = np.array(norms)
    b = 1./norms
    weights = b/b.sum()

    a1 = np.array([1.5,1.2,1,1,1,1])
    weights = weights*a1

    for idx , feature_map in enumerate(target.keys()):
        t = torch.tensor(target[feature_map]).to(device)
        g = generated[feature_map]
        MSEi = loss_fn(t, g)
        MSE.append(MSEi)
    mse = VIT[0] if len(VIT)!=0 else 0
    for i in range(weights.shape[0]):
        mse = mse+MSE[i]*(torch.tensor(weights[i]).to(device))
    return mse

# %%
precision_scope = autocast if precision == "autocast" else nullcontext
CLIP_target = {k:guidance_dic[k][imgidx:imgidx+1].astype(np.float32) for k in guidance_dic.keys()}
# %%
c = torch.Tensor(score_cs[imgidx,:].reshape(77,-1)).unsqueeze(0).to(device)
z = torch.Tensor(score_zs[imgidx,:].reshape(4,64,64)).unsqueeze(0).to(device)
seed_everything(seed)
with torch.no_grad():
    with precision_scope("cuda"):
                zz = 1/vae.config.scaling_factor * z
                x_samples = vae.decode(zz).sample
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
im = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
im.show()
# %%
# ===================================== step 3 ======================================================


seed_everything(seed)
x_sampless = []
x_mid_outs = []
with torch.no_grad():
    with precision_scope("cuda"):
            for n in range(niter):
                uncond_input = tokenizer([""], padding="max_length", max_length=c.shape[1], return_tensors="pt")
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
                z_enc = scheduler.add_noise(z, torch.randn_like(z), torch.tensor([int(t_enc/ddim_steps*1000)]))
                x_samples, x_mid_out = model(
                            condition=c,
                            latents=z_enc,
                            num_inference_steps=ddim_steps,
                            t_start=t_enc,
                            guidance_scale=scale,
                            eta=0.,
                            uncond_embeddings=uncond_embeddings,
                            num_images_per_prompt = 1,
                            output_type='np',
                            classifier_guidance_scale=guidance_scale,
                            guided_condition=CLIP_target,
                            cal_loss = cal_loss,
                            # sag_scale=0.75,
                            num_cfg_steps=int(t_enc*guidance_strength),
                            return_dict=False
                        )    
                # ===============================
                # 循环迭代seed，为了符合cvpr效果
                for i in range(40):
                    torch.randn_like(z)
                # ===============================
                for i in range(x_samples.shape[0]):
                    x_sample = 255. * x_samples[i]
                    Image.fromarray(x_sample.astype(np.uint8)).save(f'{outpath}/result{imgidx}_{n}.png')