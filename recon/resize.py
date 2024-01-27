# %%
import os
import numpy as np 
from PIL import Image
# %%
dirpath='G:/workware/src/fMRI/Decoding/StructureConstiantMindDiffusion/output/tv01-guided-sag-us/subj01/500000.0/0.2'
savepath = 'E:/GitHub/NeuralDiffuser/recon/500000.0/0.2/'
os.makedirs(savepath, exist_ok=True)
img_path = sorted([os.path.join(dirpath, name) for name in os.listdir(dirpath) if name.endswith('.png')])
# %%
scale = 4
for p in img_path:
    img = Image.open(p)
    img.resize((img.width//scale, img.height//scale)).save(savepath+p[-9:])
