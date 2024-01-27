# %%
import os
import sys
root = '/home/src/NeuralDiffuser'
sys.path.append(root)
sys.path.append(root+'/src')
import numpy as np
from modules.nsd_access import NSDAccess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject",
    required=True,
    type=str,
    help=""
)

opt = parser.parse_args()
# %%
subject = opt.subject
out_name = f'nsdgeneral'
output_dir = f"{root}/data/mask/{subject}/"
os.makedirs(output_dir, exist_ok=True)
nsd_path = '/home/dataset/nsd/'
nsda = NSDAccess(nsd_path)
atlasnames = {
    # "prf-visualrois": ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
    # "Kastner2015": ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'TO1', 'TO2', 'LO2', 'LO1', 'V3A', 'V3B', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5'],
    # "streams": ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']      # ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']
    "nsdgeneral": ['nsdgeneral']
}

masks=False
for k in atlasnames.keys():
    atlas = nsda.read_atlas_results(subject=subject, atlas=k, data_format='func1pt8mm')
    for atlasname in atlasnames[k]:
        masks |= atlas[0]==atlas[1][atlasname]
print(f"{out_name}: {subject}: {masks.sum()}")
np.save(output_dir+f'{out_name}.npy',masks)