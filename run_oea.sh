#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate your_env_name

model_name=('PMF')

data_choice=('OEA_EN_FR_15K_V1' 'OEA_EN_DE_15K_V1' 'OEA_D_W_15K_V1' 'OEA_D_W_15K_V2')
data_split=('norm')
data_rate=(0.3)

use_surface=(0) # whether to use surface information
cross_modal=(1) # whether to apply cross-modal loss
freeze=(1) # whether to apply PMF
il=(1) # whether to apply iterative learning

for cm in "${cross_modal[@]}"
do
for mn in "${model_name[@]}"
do
for dc in "${data_choice[@]}"
do
for ds in "${data_split[@]}"
do
for dr in "${data_rate[@]}"
do
for us in "${use_surface[@]}"
do
for de in "${freeze[@]}"
do

echo "model: " "${mn}"
echo "data_choice: " "${dc}"
echo "data_split: " "${ds}"
echo "data_rate: " "${dr}"
echo "use_surface: " "${us}"
echo "cross_modal: " "${cm}"
echo "freeze" "${de}"


CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
            --eval_epoch    1  \
            --model_name    "${mn}" \
            --data_choice   "${dc}" \
            --data_split    "${ds}" \
            --data_rate     "${dr}"\
            --semi_learn_step 5\
            --il "${il}" \
            --lr            0.001  \
            --save_model    1 \
	        --csls          \
	        --csls_k        3 \
            --exp_name      PMF \
            --exp_id        ${dc}_${ds}_il${il}_sf${us} \
            --use_surface   ${us}    \
            --cross_modal "${cm}" \
            --freeze "${de}" \
            --enable_sota 

sleep 10
done
done
done
done
done
done
done
done
