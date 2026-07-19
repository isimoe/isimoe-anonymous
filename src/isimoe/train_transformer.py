import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import numpy as np
import argparse
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

from src.common.fusion_models.transformer import Transformer
from src.isimoe.isimoe_train import train_and_evaluate_isimoe
from src.common.utils import setup_logger, str2bool


# 解析输入参数
def parse_args():
    parser = argparse.ArgumentParser(description="Interaction-Transformer")
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument(
        "--modality", type=str, default="IGCB"
    )  # ADNI 使用 I/G/C/B，CMU-MOSI 使用 T/V/A
    parser.add_argument('--preprocessed', type=str2bool, default=True)  # 是否使用预处理后的图像模态
    parser.add_argument("--initial_filling", type=str, default="mean")  # 缺失模态的初始填充方式
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=4
    )  # DataLoader 工作进程数
    parser.add_argument(
        "--pin_memory", type=str2bool, default=True
     )  # 是否在 DataLoader 中启用内存锁页
    parser.add_argument(
        "--use_common_ids", type=str2bool, default=True
    )  # 是否在不同模态间使用共同样本 ID
    parser.add_argument(
        "--save", type=str2bool, default=True
    )  # 是否保存模型和结果

    parser.add_argument(
        "--debug", type=str2bool, default=False
    )  # 是否启用调试模式

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--temperature_rw", type=float, default=1
    )  # 重加权模型的温度系数
    parser.add_argument(
        "--hidden_dim_rw", type=int, default=256
    )  # 重加权模型的隐藏维度
    parser.add_argument(
        "--num_layer_rw", type=int, default=1
    )  # 重加权模型的层数
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--num_layers_enc", type=int, default=1
    )  # 编码器的 MLP 层数
    parser.add_argument(
        "--num_layers_fus", type=int, default=1
    )  # 融合模型的 MLP 层数
    parser.add_argument(
        "--num_layers_pred", type=int, default=1
    )  # 预测头的 MLP 层数
    parser.add_argument("--num_heads", type=int, default=4)  # 注意力头数
    parser.add_argument(
        "--patch", type=str2bool, default=True
    )  # 是否对输入进行分块
    parser.add_argument(
        "--num_patches", type=int, default=16
    )  # 输入分块数量

    parser.add_argument(
        "--fusion_sparse", type=str2bool, default=False
    )  # 是否在融合层中使用稀疏 MoE
    parser.add_argument("--gate", type=str, default="None")
    parser.add_argument("--num_experts", type=int, default=16)  # 专家数量
    parser.add_argument("--num_routers", type=int, default=1)  # 路由器数量
    parser.add_argument("--top_k", type=int, default=2)  # 每个路由器选择的专家数量
    parser.add_argument("--dropout", type=float, default=0.5)  # 丢弃率
    parser.add_argument("--gate_loss_weight", type=float, default=1e-2)
    parser.add_argument(
        "--amss_enabled", type=str2bool, default=False, help="启用 M2-AMSS 自适应掩码"
    )
    parser.add_argument(
        "--amss_tau", type=float, default=1.0, help="AMSS softmax 缩放的温度系数"
    )
    parser.add_argument(
        "--amss_momentum", type=float, default=0.9, help="平滑模态重要性估计的动量"
    )
    parser.add_argument(
        "--scale_factor", type=float, default=1.0, help="掩码缩放系数"
    )
    parser.add_argument(
        "--topk_ratio", type=float, default=0.2, help="提取相关部分时使用的 Top-k 比例"
    )
    parser.add_argument(
        "--amss_aux_weight", type=float, default=0.0, help="逐专家辅助监督损失的权重（L_aux）"
    )
    parser.add_argument(
        "--amss_orth_weight", type=float, default=0.0, help="专家间正交正则项的权重（L_orth）"
    )
    parser.add_argument(
        "--ldiv_level",
        type=str,
        default="feature",
        choices=["feature", "prediction"],
        help="L_div/L_orth 使用的表示：中间特征或最终预测",
    )
    parser.add_argument(
        "--use_interaction", type=str2bool, default=True, help="是否使用交叉注意力交互"
    )
    parser.add_argument(
        "--enable_r_path",
        type=str2bool,
        default=False,
        help="是否启用可选的冗余专家路径",
    )
    parser.add_argument(
        "--track_mir_curve",
        type=str2bool,
        default=False,
        help="记录训练期间的批级 MIR，并绘制各模态轨迹",
    )
    parser.add_argument(
        "--mir_log_dir",
        type=str,
        default="./outputs/mir_curves",
        help="MIR 轨迹 CSV 与图像的保存目录",
    )

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    logger = setup_logger(
        f"./logs/isimoe/transformer/{args.data}",
        f"{args.data}",
        f"{args.modality}.txt",
    )
    seeds = np.arange(args.n_runs)  # [0, 1, 2]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    log_summary = "======================================================================================\n"

    model_kwargs = {
        "model": "Interaction-MoE-Transformer",
        "lr": args.lr,
        "temperature_rw": args.temperature_rw,
        "hidden_dim_rw": args.hidden_dim_rw,
        "num_layer_rw": args.num_layer_rw,
        "interaction_loss_weight": args.interaction_loss_weight,
        "modality": args.modality,
        "data": args.data,
        "gate_loss_weight": args.gate_loss_weight,
        "interaction_loss_weight": args.interaction_loss_weight,
        "train_epochs": args.train_epochs,
        "num_experts": args.num_experts,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "amss_enabled": args.amss_enabled,
        "amss_tau": args.amss_tau,
        "scale_factor": args.scale_factor,
        "amss_momentum": args.amss_momentum,
        "amss_aux_weight": args.amss_aux_weight,
        "amss_orth_weight": args.amss_orth_weight,
        "ldiv_level": args.ldiv_level,
        "use_interaction": args.use_interaction,
        "enable_r_path": args.enable_r_path,
        "track_mir_curve": args.track_mir_curve,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print("Modality:", args.modality)
    print(model_kwargs)
    data_to_nlabels = {
        "adni": 3,
        "mmimdb": 23,
        "enrico": 20,
        "mosi": 2,
        "mosi_regression": 1,
    }
    n_labels = data_to_nlabels[args.data]
    num_modalities = num_modality = len(args.modality)

    if args.data == "mosi_regression":
        val_losses = []
        val_accs = []
        test_accs = []
        test_maes = []
    else:
        val_accs = []
        val_f1s = []
        val_f1_micros = []
        val_aucs = []
        test_accs = []
        test_f1s = []
        test_f1_micros = []
        test_aucs = []

    ############ 效率统计
    train_times = []
    infer_times = []
    flops = []
    params = []
    ############ 效率统计

    if len(seeds) == 1:
        fusion_model = Transformer(
            num_modalities,
            args.num_patches,
            args.hidden_dim,
            n_labels,
            args.num_layers_fus,
            args.num_layers_pred,
            args.num_experts,
            args.num_routers,
            args.top_k,
            args.num_heads,
            args.dropout,
            args.fusion_sparse,
            args.gate,
        ).to(device)

        if args.data == "mosi_regression":
            (
                val_loss,
                val_acc,
                test_acc,
                test_mae,
                train_time,
                infer_time,
                flop,
                param,
            ) = train_and_evaluate_isimoe(args, args.seed, fusion_model, "transformer")
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_maes.append(test_mae)

        else:

            (
                val_acc,
                val_f1,
                val_f1_micro,
                val_auc,
                test_acc,
                test_f1,
                test_f1_micro,
                test_auc,
                train_time,
                infer_time,
                flop,
                param,
            ) = train_and_evaluate_isimoe(args, args.seed, fusion_model, "transformer")

            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_f1_micros.append(val_f1_micro)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_f1_micros.append(test_f1_micro)
            test_aucs.append(test_auc)
        ############ 效率统计
        train_times.append(train_time)
        infer_times.append(infer_time)
        flops.append(flop)
        params.append(param)
        ############ 效率统计
    else:
        for seed in seeds:
            fusion_model = Transformer(
                num_modalities,
                args.num_patches,
                args.hidden_dim,
                n_labels,
                args.num_layers_fus,
                args.num_layers_pred,
                args.num_experts,
                args.num_routers,
                args.top_k,
                args.num_heads,
                args.dropout,
                args.fusion_sparse,
                args.gate,
            ).to(device)

            if args.data == "mosi_regression":
                (
                    val_loss,
                    val_acc,
                    test_acc,
                    test_mae,
                    train_time,
                    infer_time,
                    flop,
                    param,
                ) = train_and_evaluate_isimoe(
                    args, args.seed, fusion_model, "transformer"
                )
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                test_accs.append(test_acc)
                test_maes.append(test_mae)

            else:

                (
                    val_acc,
                    val_f1,
                    val_f1_micro,
                    val_auc,
                    test_acc,
                    test_f1,
                    test_f1_micro,
                    test_auc,
                    train_time,
                    infer_time,
                    flop,
                    param,
                ) = train_and_evaluate_isimoe(args, seed, fusion_model, "transformer")

                val_accs.append(val_acc)
                val_f1s.append(val_f1)
                val_aucs.append(val_auc)
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
                test_f1_micros.append(test_f1_micro)
                test_aucs.append(test_auc)
            ############ 效率统计
            train_times.append(train_time)
            infer_times.append(infer_time)
            flops.append(flop)
            params.append(param)
            ############ 效率统计

    ############ 效率统计
    mean_train_time = np.mean(train_times)
    variance_train_time = np.var(train_times)
    mean_infer_time = np.mean(infer_times)
    variance_infer_time = np.var(infer_times)
    mean_flop = np.mean(flops)
    variance_flop = np.var(flops)
    mean_gflop = np.mean(np.array(flops) / 1e9)
    variance_gflop = np.var(np.array(flops) / 1e9)
    mean_param = np.mean(params)
    variance_param = np.var(params)

    log_summary += "\n"
    log_summary += (
        f"Train one epoch time: {mean_train_time:.2f} ± {variance_train_time:.2f} "
    )
    log_summary += "\n"
    log_summary += (
        f"Inference one epoch time: {mean_infer_time:.2f} ± {variance_infer_time:.2f} "
    )
    log_summary += "\n"
    log_summary += f"flops: {mean_flop:,.0f} ± {variance_flop:,.0f} "
    log_summary += "\n"
    log_summary += f"gflops: {mean_gflop:.2f} ± {variance_gflop:.2f} "
    log_summary += "\n"
    log_summary += f"param: {mean_param:,.0f} ± {variance_param:,.0f} "
    log_summary += "\n"
    ############ 效率统计

    if args.data == "mosi_regression":
        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_loss = np.mean(val_losses)
        val_std_loss = np.std(val_losses)
        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_mae = np.mean(test_maes) * 100
        test_std_mae = np.std(test_maes) * 100

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f} "
        log_summary += (
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}  "
        )
        log_summary += (
            f"[Test] Mean Absolute Error: {test_avg_mae:.2f} ± {test_std_mae:.2f}  "
        )

        print(model_kwargs)
        print(
            f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f} "
        )
        print(f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} ")

    else:

        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_f1 = np.mean(val_f1s) * 100
        val_std_f1 = np.std(val_f1s) * 100
        val_avg_f1_micro = np.mean(val_f1_micros) * 100
        val_std_f1_micro = np.std(val_f1_micros) * 100
        val_avg_auc = np.mean(val_aucs) * 100
        val_std_auc = np.std(val_aucs) * 100

        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_f1 = np.mean(test_f1s) * 100
        test_std_f1 = np.std(test_f1s) * 100
        test_avg_f1_micro = np.mean(test_f1_micros) * 100
        test_std_f1_micro = np.std(test_f1_micros) * 100
        test_avg_auc = np.mean(test_aucs) * 100
        test_std_auc = np.std(test_aucs) * 100

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average F1 Score (Macro): {val_avg_f1:.2f} ± {val_std_f1:.2f} "
        log_summary += f"[Val] Average F1 Score (Micro): {val_avg_f1_micro:.2f} ± {val_std_f1_micro:.2f} "
        log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / "
        log_summary += (
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} "
        )
        log_summary += (
            f"[Test] Average F1 (Macro) Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} "
        )
        log_summary += f"[Test] Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f} "
        log_summary += f"[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} "

        print(model_kwargs)
        print(
            f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score (Macro): {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average F1 Score (Micro): {val_avg_f1_micro:.2f} ± {val_std_f1_micro:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}  "
        )
        print(
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score (Macro): {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}  "
        )

    logger.info(log_summary)


if __name__ == "__main__":
    main()
