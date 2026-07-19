export device=1
# 补全缺失的变量定义，防止传参为空报错
export warm_up_epochs=0

# 所有循环变量已对齐表格I²MoE-MulT MOSI数据集的超参数
for lr in 0.0001
do
for top_k in 1
do
for modality in TVA
do
for gate in SwitchGate
do
for batch_size in 32
do
for hidden_dim in 256
do
for num_patches in 4
do
for num_experts in 4
do
for num_layers_pred in 1
do
for num_layers_fus in 1
do
for num_layers_enc in 1
do
for num_routers in 1
do
for num_heads in 1
do
for gate_loss_weight in 0.01
do
for interaction_loss_weight in 0.005
do
for temperature_rw in 2.0
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for train_epochs in 30
do
# ================= 新增：M2-AMSS 策略参数遍历 =================
for use_interaction in True
do
for amss_enabled in True
do
for topk_ratio in 0.2
do
for amss_aux_weight in 0.3
do
for amss_orth_weight in 0.3
do
# ==============================================================
CUDA_VISIBLE_DEVICES=$device python src/isimoe/train_switchgate.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --data mosi \
    --gate $gate \
    --train_epochs $train_epochs \
    --modality $modality \
    --fusion_sparse True \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --warm_up_epochs $warm_up_epochs \
    --num_layers_enc $num_layers_enc \
    --num_layers_fus $num_layers_fus \
    --num_layers_pred $num_layers_pred \
    --num_patches $num_patches \
    --num_experts $num_experts \
    --num_routers $num_routers \
    --top_k $top_k \
    --num_heads $num_heads \
    --dropout 0.5 \
    --n_runs 1 \
    --gate_loss_weight $gate_loss_weight \
    --interaction_loss_weight $interaction_loss_weight \
    --save False \
    --use_common_ids True \
    # 以下为新增的M2-AMSS策略参数
    --use_interaction $use_interaction \
    --amss_enabled $amss_enabled \
    --topk_ratio $topk_ratio \
    --amss_aux_weight $amss_aux_weight \
    --amss_orth_weight $amss_orth_weight \
    --amss_tau 1.0 \
    --amss_momentum 0.9 \
    --scale_factor 1.0
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
