#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_CUDA_CHECK=1

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

subj=1
multi_subject=true
wandb_log=true

run_cmd="python ../src/train_c.py --subj ${subj} "
if ${multi_subject}; then 
        run_cmd+=" --multi_subject" 
fi
if ${wandb_log}; then 
        run_cmd+=" --wandb_log" 
fi
echo ${run_cmd}
eval ${run_cmd} 
set +x

feats=('g_2' 'g_4' 'g_6' 'g_8' 'g_10' 'g_12')
for feat in "${feats[@]}"; do
    run_cmd="python ../src/train_g.py --subj ${subj} --feat ${feat} "
    if ${multi_subject}; then 
            run_cmd+=" --multi_subject" 
    fi
    if ${wandb_log}; then 
            run_cmd+=" --wandb_log" 
    fi
    echo ${run_cmd}
    eval ${run_cmd} 
    set +x
done

run_cmd="python ../src/train_z.py --subj ${subj} "
if ${multi_subject}; then 
        run_cmd+=" --multi_subject" 
fi
if ${wandb_log}; then 
        run_cmd+=" --wandb_log" 
fi
echo ${run_cmd}
eval ${run_cmd} 
set +x
