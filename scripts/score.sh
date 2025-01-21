#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_CUDA_CHECK=1

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

subj=1

run_cmd="python ../src/score_c.py --subj ${subj} "
echo ${run_cmd}
eval ${run_cmd} 
set +x

feats=('g_2' 'g_4' 'g_6' 'g_8' 'g_10' 'g_12')
for feat in "${feats[@]}"; do
    run_cmd="python ../src/score_g.py --subj ${subj} --feat ${feat} "
    echo ${run_cmd}
    eval ${run_cmd} 
    set +x
done

run_cmd="python ../src/score_z.py --subj ${subj} "
echo ${run_cmd}
eval ${run_cmd} 
set +x
