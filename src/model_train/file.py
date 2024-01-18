# %%
import argparse
import os, sys
project_path = '/opt/data/private/src/fMRI/Decoding/NeuralDiffuser'
sys.path.append(project_path)
sys.path.append(project_path+'/src_icml24')
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject", 
    type=str,
    default='subj01',
    help="",
)
parser.add_argument(
    "--file",  
    type=str,
    default='tr_each',
    help="",
)
opt = parser.parse_args()
subject = opt.subject
file = opt.file
# %%
def CLIP_feature_layer(CLIP_feature_all_layer,layer_name):
    if layer_name == 'VisionTransformer-1':
        CLIP_layer = CLIP_feature_all_layer[:,38400*6:]
    elif layer_name[-2] == '-':
        index = int(int(layer_name[-1])/2)
        CLIP_layer = CLIP_feature_all_layer[:, 38400 * (index - 1):38400 * index]
    else:
        index = int(int(layer_name[-2:])/2)
        CLIP_layer = CLIP_feature_all_layer[:, 38400 * (index - 1):38400 * index]
    return CLIP_layer

dirpath = f'{project_path}/data/proj_feat/{subject}/g'

X = np.load(f'{dirpath}/{file}.npy').squeeze()
layers = ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12', 'VisionTransformer-1']
for layer in layers:
    tmp = CLIP_feature_layer(X, layer)
    savepath = f'{dirpath}/{layer}'
    os.makedirs(savepath, exist_ok=True)
    np.save(f'{savepath}/{file}', tmp)