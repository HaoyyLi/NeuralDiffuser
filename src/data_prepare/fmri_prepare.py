# %%
import argparse
import os
import sys
root = '/opt/data/private/src/fMRI/Decoding/NeuralDiffuser'
sys.path.append(root)
sys.path.append(root+'/src_icml24')
import numpy as np
import pandas as pd
from modules.nsd_access import NSDAccess
import scipy.io
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# %%

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        help=""
    )
    parser.add_argument(
        "--mask_name",
        required=True,
        type=str,
        help="",
    )

    opt = parser.parse_args()
    subject = opt.subject
    mask_name = opt.mask_name

    nsd_path = '/opt/data/private/dataset/nsd/'
    mask_path = f"{root}/data/mask/{subject}/{mask_name}.npy"

    nsda = NSDAccess(nsd_path)
    nsd_expdesign = scipy.io.loadmat(nsd_path+'nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix'] -1 

    behs = pd.DataFrame()
    for i in tqdm(range(1,38)):
        beh = nsda.read_behavior(subject=subject,
                                session_index=i)
        behs = pd.concat((behs,beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1
    stims_all = behs['73KID'] - 1

    savedir = f'{root}/data/fmri/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(f'{savedir}/behs_stims.npy'):
        np.save(f'{savedir}/behs_stims.npy',stims_all)
        np.save(f'{savedir}/behs_stims_ave.npy',stims_unique)
    # %%
    mask = np.load(mask_path)

    for i in tqdm(range(1,38)):
        print(i)
        beta_trial = nsda.read_betas(subject=subject, 
                                session_index=i, 
                                trial_index=[], # empty list as index means get all for this session
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm').transpose(0,3,2,1)
        beta_trial = beta_trial[:,mask]
        beta_trial = StandardScaler(with_mean=True, with_std=True).fit_transform(beta_trial)
        if i==1:
            betas_all = beta_trial
        else:
            betas_all = np.concatenate((betas_all,beta_trial),0)    

    # %%
    # Averaging for each stimulus
    betas_all_ave = []
    for stim in tqdm(stims_unique):
        stim_mean = np.mean(betas_all[stims_all == stim,:],axis=0)
        betas_all_ave.append(stim_mean)
    betas_all_ave = np.stack(betas_all_ave)
    print(betas_all_ave.shape)
    # %%
    # Train/Test Split
    # ALLDATA
    betas_tr = []
    betas_te = []

    for idx,stim in enumerate(tqdm(stims_all)):
        if stim in sharedix:
            betas_te.append(betas_all[idx,:])
        else:
            betas_tr.append(betas_all[idx,:])

    betas_tr = np.stack(betas_tr)
    betas_te = np.stack(betas_te)
    # %%
    # AVERAGED DATA        
    betas_ave_tr = []
    betas_ave_te = []
    for idx,stim in enumerate(tqdm(stims_unique)):
        if stim in sharedix:
            betas_ave_te.append(betas_all_ave[idx,:])
        else:
            betas_ave_tr.append(betas_all_ave[idx,:])
    betas_ave_tr = np.stack(betas_ave_tr)
    betas_ave_te = np.stack(betas_ave_te)
    # %%
    # Save
    os.makedirs(f'{savedir}/{mask_name}/', exist_ok=True)
    np.save(f'{savedir}/{mask_name}/betas_tr.npy',betas_tr)
    np.save(f'{savedir}/{mask_name}/betas_te.npy',betas_te)
    np.save(f'{savedir}/{mask_name}/betas_ave_tr.npy',betas_ave_tr)
    np.save(f'{savedir}/{mask_name}/betas_ave_te.npy',betas_ave_te)

if __name__=='__main__':
    main()

'''
python fmri_prepare.py --subject subj01 --mask_name nsdgeneral
python fmri_prepare.py --subject subj02 --mask_name nsdgeneral
python fmri_prepare.py --subject subj05 --mask_name nsdgeneral
python fmri_prepare.py --subject subj07 --mask_name nsdgeneral
'''