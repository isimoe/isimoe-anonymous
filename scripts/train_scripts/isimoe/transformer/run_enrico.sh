#!/bin/bash
export device=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# for lr in 0.0001
for lr in 0.0001
do
for temperature_rw in 1
do
for hidden_dim_rw in 128
do
for num_layer_rw in 3
do
for interaction_loss_weight in 0.00
do
for hidden_dim in 128
do
for num_layers_enc in 1
do
for train_epochs in 30
do
for amss_aux_weight in 0.1
do
for amss_orth_weight in 0.3
do
for amss_tau in 1.0
do
for scale_factor in 1.0
do
for ldiv_level in feature
do

    CUDA_VISIBLE_DEVICES=$device $PYTHON_BIN src/isimoe/train_transformer.py \
        --amss_enabled True \
        --use_interaction True \
        --enable_r_path False \
        --data enrico \
        --modality SW \
        --train_epochs $train_epochs \
        --batch_size 8 \
        --lr $lr \
        --hidden_dim $hidden_dim \
        --hidden_dim_rw $hidden_dim_rw \
        --num_layer_rw $num_layer_rw \
        --temperature_rw $temperature_rw \
        --num_layers_fus 1 \
        --num_layers_enc $num_layers_enc \
        --num_layers_pred 1 \
        --num_patches 4 \
        --num_experts 4 \
        --num_routers 1 \
        --top_k 2 \
        --num_heads 2 \
        --dropout 0.5 \
        --gate None \
        --fusion_sparse False \
        --gate_loss_weight 0.01 \
        --interaction_loss_weight $interaction_loss_weight \
        --n_runs 3 \
        --save True \
        --use_common_ids True \
        --amss_aux_weight $amss_aux_weight \
        --amss_orth_weight $amss_orth_weight \
        --amss_tau $amss_tau \
        --scale_factor $scale_factor \
        --amss_momentum 0.1 \
        --preprocessed True \
        --topk_ratio 0.3 \
        --ldiv_level $ldiv_level \
        --track_mir_curve True \
        --mir_log_dir ./outputs/mir_curves

done
done
done
done
done
done
done
done
done
done
done
done
done
