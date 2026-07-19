export device=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# for lr in 0.0001
for lr in 0.0001
do
for temperature_rw in 1.0
do
for hidden_dim_rw in 256
do
for num_layer_rw in 2
do
for interaction_loss_weight in 0.5
do
for hidden_dim in 256
do
for num_layers_enc in 2
do
for train_epochs in 2
do
for amss_aux_weight in 0.1
do
for amss_orth_weight in 0.3
do
for amss_tau in 1.0
do
for scale_factor in 0.001
do

CUDA_VISIBLE_DEVICES=$device python src/isimoe/train_transformer.py \
    --data mmimdb \
    --amss_enabled True \
    --use_interaction False \
    --enable_r_path True \
    --modality LI \
    --train_epochs $train_epochs \
    --batch_size 32 \
    --lr $lr \
    --hidden_dim $hidden_dim \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --temperature_rw $temperature_rw \
    --num_layers_fus 2 \
    --num_layers_enc $num_layers_enc \
    --num_layers_pred 2 \
    --num_patches 8 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 4 \
    --dropout 0.5 \
    --gate None \
    --fusion_sparse False \
    --gate_loss_weight 0.01 \
    --interaction_loss_weight $interaction_loss_weight \
    --n_runs 3 \
    --seed 0 \
    --save True \
    --use_common_ids True \
    --use_fisher_rebalance True \
    --amss_aux_weight $amss_aux_weight \
    --amss_orth_weight $amss_orth_weight \
    --amss_tau $amss_tau \
    --scale_factor $scale_factor \
    --amss_momentum 0.9 \
    --preprocessed True \
    --topk_ratio 0.3

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