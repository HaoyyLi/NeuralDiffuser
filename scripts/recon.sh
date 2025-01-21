#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_CUDA_CHECK=1

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

subj=1
guidance_scale=300000
guidance_strength=0.2

run_cmd="python ../src/recon.py --subjuct ${subj} --guidance_scale ${guidance_scale} --guidance_strength ${guidance_strength}"
echo ${run_cmd}
eval ${run_cmd} 
set +x

