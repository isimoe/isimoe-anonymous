import torch
import csv
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from copy import deepcopy
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)
from src.common.datasets.MultiModalDataset import create_loaders

from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    plot_interaction_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
    visualize_expert_logits_distribution,
    set_style,
)

from src.isimoe.InteractionMoE import  SpecializedInteractionMoE
from src.isimoe.InteractionMoERegression import  SpecializedInteractionMoERegression

set_style()


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _modality_display_names(args, num_modalities):
    if args.data == "enrico" and args.modality.upper() == "SW":
        return ["Screenshot", "Wireframe"]
    return [f"Modality {i + 1}" for i in range(num_modalities)]


def _save_mir_curve(args, mir_records, seed, fusion, num_modalities):
    if not mir_records:
        return None, None

    out_dir = Path(getattr(args, "mir_log_dir", "./outputs/mir_curves"))
    out_dir = out_dir / "isimoe" / fusion / args.data
    out_dir.mkdir(exist_ok=True, parents=True)
    ldiv_level = getattr(args, "ldiv_level", "feature")
    run_tag = (
        f"seed_{seed}_modality_{args.modality}_ldiv_{ldiv_level}_epochs_{args.train_epochs}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    csv_path = out_dir / f"{run_tag}_mir_curve.csv"
    epoch_csv_path = out_dir / f"{run_tag}_mir_epoch_curve.csv"

    fieldnames = ["epoch", "batch", "global_step", "label_entropy"]
    for i in range(num_modalities):
        fieldnames.extend([f"mir_m{i}", f"rho_m{i}", f"score_m{i}"])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in mir_records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    epochs = np.array(sorted({int(record["epoch"]) for record in mir_records}), dtype=float)
    epoch_records = []
    for epoch in epochs:
        epoch_batch_records = [
            record for record in mir_records if int(record["epoch"]) == int(epoch)
        ]
        epoch_record = {"epoch": int(epoch)}
        for i in range(num_modalities):
            epoch_values = np.array(
                [record[f"mir_m{i}"] for record in epoch_batch_records],
                dtype=float,
            )
            epoch_record[f"mir_m{i}"] = float(np.mean(epoch_values))
        epoch_records.append(epoch_record)

    epoch_fieldnames = ["epoch"] + [f"mir_m{i}" for i in range(num_modalities)]
    with epoch_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
        writer.writeheader()
        for record in epoch_records:
            writer.writerow(record)

    modality_names = _modality_display_names(args, num_modalities)
    line_styles = [
        {"color": "#C94F4F", "marker": "o", "linestyle": "-"},
        {"color": "#4F73B8", "marker": "s", "linestyle": "--"},
        {"color": "#6B9D6E", "marker": "^", "linestyle": "-."},
        {"color": "#8A6BBE", "marker": "D", "linestyle": ":"},
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 2.15), constrained_layout=True)
    ax.axhline(0.5, color="#B7BCC4", linestyle="--", linewidth=0.9, zorder=0)

    all_values = []
    for i in range(num_modalities):
        values = np.array([record[f"mir_m{i}"] for record in epoch_records], dtype=float)
        all_values.extend(values.tolist())
        style = line_styles[i % len(line_styles)]
        ax.plot(
            epochs,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.15,
            marker=style["marker"],
            markersize=3.0,
            markeredgewidth=0.0,
            label=modality_names[i],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MIR share")
    ax.grid(axis="y", color="#D9DDE3", linestyle=":", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", width=0.8, length=3.0)
    if all_values:
        y_min = min(min(all_values), 0.5)
        y_max = max(max(all_values), 0.5)
        margin = max((y_max - y_min) * 0.18, 0.012)
        ax.set_ylim(max(0.0, y_min - margin), min(1.0, y_max + margin))
    ax.legend(
        frameon=False,
        loc="best",
        handlelength=2.0,
        handletextpad=0.6,
        borderaxespad=0.2,
    )

    fig_path = out_dir / f"{run_tag}_mir_curve.pdf"
    png_path = out_dir / f"{run_tag}_mir_curve.png"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return csv_path, fig_path


def train_and_evaluate_isimoe(args, seed, fusion_model, fusion):
    """Train and evaluate interaction MoE.

    Args:
        args (argparser.args): argument
        seed (int): random seed
        ensemble_model (nn.Module): ensemble model
        fusion (str): name of fusion method

    Raises:
        ValueError

    Returns:
        tuple: (best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc)
    """
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
    num_modalities = len(args.modality)

    # ... [数据加载部分保持不变] ...
    if args.data == "adni":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_adni(args)
    elif args.data == "mosi":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mosi(args)
    elif args.data == "sarcasm":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_sarcasm(args)
    elif args.data == "humor":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_humor(args)
    elif args.data == "enrico":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_enrico(args)
    elif args.data == "mmimdb":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mmimdb(args)
    elif args.data == "mosi_regression":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mosi_regression(args)

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.preprocessed,
        args.use_common_ids,
        dataset=args.data,
    )
    if args.data == "mosi_regression" and getattr(args, "enable_r_path", False):
        raise ValueError(
            "--enable_r_path is currently implemented for SpecializedInteractionMoE "
            "classification runs only; mosi_regression uses SpecializedInteractionMoERegression."
        )
    # Optionally use the specialized MI+Fisher rebalancing / AMSS InteractionMoE (only for 2-modality case)
    if  args.data != "mosi_regression":
        ensemble_model = SpecializedInteractionMoE(
            num_modalities=num_modalities,
            fusion_model=deepcopy(fusion_model),
            fusion_sparse=args.fusion_sparse,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
            topk_ratio=getattr(args, "topk_ratio", 0.2),
            amss_enabled=getattr(args, "amss_enabled", False),
            amss_tau=getattr(args, "amss_tau", 1.0),
            amss_momentum=getattr(args, "amss_momentum", 0.9),
            scale_factor=getattr(args, "scale_factor", 1.0),
            use_interaction=getattr(args, "use_interaction", True),
            enable_r_path=getattr(args, "enable_r_path", False),
            data=args.data,
        ).to(device)



    if args.data == "mosi_regression":
        ensemble_model = SpecializedInteractionMoERegression(
            num_modalities=num_modalities,  # 传入实际模态数 (例如 3)
            fusion_model=deepcopy(fusion_model),
            fusion_sparse=args.fusion_sparse,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
            topk_ratio=getattr(args, "topk_ratio", 0.2),
            amss_enabled=getattr(args, "amss_enabled", False),
            amss_tau=getattr(args, "amss_tau", 1.0),
            amss_momentum=getattr(args, "amss_momentum", 0.9),
            scale_factor=getattr(args, "scale_factor", 1.0),
            use_interaction=getattr(args, "use_interaction", True),
        ).to(device)


    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.data in ["adni", "enrico", "mosi", "sarcasm", "humor"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mosi_regression":
        criterion = torch.nn.SmoothL1Loss()  # Regression
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()

    if args.data == "mosi_regression":
        best_val_loss = 100000
    elif args.data == "mmimdb":
        best_val_f1 = 0
    else:
        best_val_acc = 0.0

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "interaction": [], "gate": []}
    else:
        plotting_total_losses = {"task": [], "interaction": []}

    plotting_interaction_losses = {}
    for i in range(len(args.modality)):
        plotting_interaction_losses[f"uni_{i + 1}"] = []
    plotting_interaction_losses[f"syn"] = []
    plotting_interaction_losses[f"red"] = []

    ############ efficiency
    train_time = 0
    ############ efficiency
    track_mir_curve = getattr(args, "track_mir_curve", False)
    mir_records = []
    global_step = 0

    for epoch in trange(args.train_epochs):
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        ensemble_model.train()

        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []
        batch_interaction_losses = []

        # 对于 Specialized 模型，experts 数量 = num_modalities + 1 (shared)
        # 兼容旧代码的 uniqueness/synergy/redundancy 统计逻辑可能不完全适用，但我们保留以防报错
        num_interaction_experts = len(args.modality) + 1 + 1  # 预留足够空间
        interaction_loss_sums = [0] * (num_interaction_experts + 5)
        minibatch_count = len(train_loader)

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            if args.fusion_sparse:
                expert_outputs, _, outputs, interaction_losses, gate_losses = ensemble_model(
                    fusion_input
                )
            else:
                expert_outputs, _, outputs, interaction_losses = ensemble_model(fusion_input)

            if args.data == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)

            interaction_loss = sum(interaction_losses) / max(1, len(interaction_losses))

            # AMSS auxiliary supervision and orthogonality
            amss_aux = 0.0
            amss_orth = 0.0
            ldiv_sources = expert_outputs
            if getattr(args, "ldiv_level", "feature") == "feature":
                cached_features = getattr(ensemble_model, "expert_features", None)
                if cached_features is not None and len(cached_features) >= num_modalities + 1:
                    ldiv_sources = cached_features

            # [关键修改 2]: 通用化 AMSS 损失计算，支持 3 模态 (N 模态)
            # expert_outputs 结构: [Exp_1, Exp_2, ..., Exp_N, Exp_Shared]
            if len(expert_outputs) >= num_modalities + 1:

                # --- 1. 计算 AMSS 辅助损失 (对前 N 个单模态专家) ---
                if getattr(args, "amss_aux_weight", 0.0) > 0.0:
                    aux_criterion = criterion  # 复用主任务的 criterion (Regression用SmoothL1)

                    for i in range(num_modalities):
                        raw_logits = expert_outputs[i]
                        # 兼容性处理：Specialized 模型可能返回 [[tensor]] 结构
                        if isinstance(raw_logits, (list, tuple)):
                            raw_logits = raw_logits[0]

                        if args.data == "mosi_regression":
                            # Regression 需要 unsqueeze label
                            curr_loss = aux_criterion(raw_logits, batch_labels.unsqueeze(1))
                        elif args.data == "mmimdb":
                            curr_loss = aux_criterion(raw_logits, batch_labels.float())
                        else:
                            curr_loss = aux_criterion(raw_logits, batch_labels.long())

                        amss_aux += curr_loss

                # --- 2. 计算正交损失 (Shared Expert 与单模态专家正交) ---
                if getattr(args, "amss_orth_weight", 0.0) > 0.0:
                    # Shared expert is after the N single-modality experts; optional R is appended after it.
                    raw_shared = ldiv_sources[num_modalities]
                    if isinstance(raw_shared, (list, tuple)):
                        raw_shared = raw_shared[0]

                    # 只有当维度合适时才计算 (必须是 [B, D])
                    if raw_shared is not None and raw_shared.dim() == 2:
                        shared_norm = F.normalize(raw_shared, p=2, dim=1)
                        total_cos = 0.0
                        valid_pairs = 0

                        for i in range(num_modalities):
                            raw_spec = ldiv_sources[i]
                            if isinstance(raw_spec, (list, tuple)):
                                raw_spec = raw_spec[0]

                            if (
                                raw_spec is not None
                                and raw_spec.dim() == 2
                                and raw_spec.shape == raw_shared.shape
                            ):
                                spec_norm = F.normalize(raw_spec, p=2, dim=1)
                                # 计算 cosine 相似度的绝对值
                                cos_sim = torch.sum(torch.abs(torch.sum(spec_norm * shared_norm, dim=1)))
                                total_cos += cos_sim / (spec_norm.size(0) + 1e-8)
                                valid_pairs += 1

                        if valid_pairs > 0:
                            amss_orth = total_cos / valid_pairs

            if args.fusion_sparse:
                gate_loss = torch.stack([
                    g.to(device) if isinstance(g, torch.Tensor) else torch.as_tensor(g, device=device)
                    for g in gate_losses
                ]).mean()
                loss = (
                        task_loss
                        + args.interaction_loss_weight * interaction_loss
                        + args.gate_loss_weight * gate_loss
                )
            else:
                loss = task_loss + args.interaction_loss_weight * interaction_loss

            # add AMSS auxiliary and orth losses if configured
            if getattr(args, "amss_aux_weight", 0.0) > 0.0:
                loss = loss + args.amss_aux_weight * amss_aux
            if getattr(args, "amss_orth_weight", 0.0) > 0.0:
                loss = loss + args.amss_orth_weight * amss_orth

            loss.backward()

            # If enabled, apply MI+Fisher freezing or AMSS before optimizer step
            if getattr(args, "amss_enabled", False) and hasattr(
                    ensemble_model, "apply_mi_fisher_after_backward"):

                stats = ensemble_model.apply_mi_fisher_after_backward(
                    fusion_input, batch_labels)
                if track_mir_curve:
                    record = {
                        "epoch": epoch + 1,
                        "batch": global_step % max(1, minibatch_count),
                        "global_step": global_step,
                        "label_entropy": _to_float(stats.get("label_entropy", 0.0)),
                    }
                    for i in range(num_modalities):
                        record[f"mir_m{i}"] = _to_float(stats.get(f"mir_m{i}", 0.0))
                        record[f"rho_m{i}"] = _to_float(stats.get(f"rho_m{i}", 0.0))
                        record[f"score_m{i}"] = _to_float(stats.get(f"score_m{i}", 0.0))
                    mir_records.append(record)

            optimizer.step()
            global_step += 1

            batch_task_losses.append(task_loss.item())
            batch_interaction_losses.append(interaction_loss.item())
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())

            # 更新 plotting loss (防止 Specialized 模型 loss 个数不匹配报错)
            for idx, loss in enumerate(interaction_losses):
                if idx < len(interaction_loss_sums):
                    interaction_loss_sums[idx] += loss.item()

            if args.data == "enrico":
                torch.nn.utils.clip_grad_norm_(params, 1.0)

        ############ efficiency
        epoch_end_time = time.time()
        train_epoch_time = epoch_end_time - epoch_start_time
        train_time += train_epoch_time
        ############ efficiency

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        plotting_total_losses["interaction"].append(np.mean(batch_interaction_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))

        for i in range(len(args.modality)):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i + 1}"].append(avg_loss)

        # For syn and red interaction losses (安全访问)
        if len(interaction_losses) >= 2:
            plotting_interaction_losses["syn"].append(
                interaction_loss_sums[len(interaction_losses) - 2] / minibatch_count
            )
            plotting_interaction_losses["red"].append(
                interaction_loss_sums[len(interaction_losses) - 1] / minibatch_count
            )
        else:
            plotting_interaction_losses["syn"].append(0)
            plotting_interaction_losses["red"].append(0)

        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

        with torch.no_grad():
            for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                batch_samples = {
                    k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
                }
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_mcs = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)
                optimizer.zero_grad()

                fusion_input = []
                for i, (modality, samples) in enumerate(batch_samples.items()):
                    encoded_samples = encoder_dict[modality](samples)
                    fusion_input.append(encoded_samples)


                _, _, outputs = ensemble_model.inference(fusion_input)

                if args.data == "mosi_regression":
                    # if False:
                    val_loss = criterion(outputs, batch_labels.unsqueeze(1))
                    val_losses.append(val_loss.item())
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

                else:
                    if args.data == "mmimdb":
                        val_loss = criterion(outputs, batch_labels.float())
                    else:
                        val_loss = criterion(outputs, batch_labels)
                    val_losses.append(val_loss.item())
                    if args.data == "mmimdb":
                        preds = torch.sigmoid(outputs).round()
                    else:
                        _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    if args.data in ["mosi", "sarcasm", "humor"]:
                        all_probs.extend(
                            torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                            .cpu()
                            .numpy()
                        )
                    else:
                        probs = (
                            torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        )
                        all_probs.extend(probs)
                        if (
                                probs.shape[1] != n_labels
                        ):  # n_labels is the number of classes
                            raise ValueError("Incorrect output shape from the model")
        if args.data == "mosi_regression":
            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(
                (np.array(all_preds) > 0), (np.array(all_labels) > 0)
            )
            print(
                f"[Seed {seed}/{args.n_runs - 1}] [Epoch {epoch + 1}/{args.train_epochs}] Task Loss: {np.mean(val_losses):.2f} / Val Loss: {val_loss:.2f}, Val Acc: {val_acc * 100:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc

                print(
                    f"[(**Best**) [Epoch {epoch + 1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc * 100:.2f}"
                )

                best_model_fus = deepcopy(ensemble_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                # Move the models to CPU for saving (only state_dict)
                if args.save:
                    best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                    best_model_enc_cpu = {
                        modality: {k: v.cpu() for k, v in enc_state.items()}
                        for modality, enc_state in best_model_enc.items()
                    }

        else:
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            val_f1_micro = f1_score(all_labels, all_preds, average="micro")
            if args.data == "enrico":
                val_auc = roc_auc_score(
                    np.array(all_labels),
                    np.array(all_probs),
                    multi_class="ovo",
                    labels=list(range(n_labels)),
                )
            elif args.data in ["mosi", "sarcasm", "humor"]:
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0
            elif args.data == "adni":
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            print(
                f"[Seed {seed}/{args.n_runs - 1}] [Epoch {epoch + 1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc * 100:.2f}, Val F1: {val_f1 * 100:.2f}, Val AUC: {val_auc * 100:.2f}"
            )
            best_val_f1_micro=0.0
            if args.data == "mmimdb":
                # if False:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_f1_micro = val_f1_micro
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    print(
                        f" [(**Best**) Epoch {epoch + 1}/{args.train_epochs}] Val Acc: {val_acc * 100:.2f}, Val F1: {val_f1 * 100:.2f}, Val AUC: {val_auc * 100:.2f}"
                    )

                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }

                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
            else:
                if val_acc > best_val_acc:
                    print(
                        f" [(**Best**) Epoch {epoch + 1}/{args.train_epochs}] Val Acc: {val_acc * 100:.2f}, Val F1: {val_f1 * 100:.2f}, Val AUC: {val_auc * 100:.2f}"
                    )
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_f1_micro = val_f1_micro
                    best_val_auc = val_auc
                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }
                    # Move the models to CPU for saving (only state_dict)
                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
    ############ efficiency
    total_param = parameter_count(ensemble_model)[""]
    # flop = FlopCountAnalysis(ensemble_model, fusion_input)
    total_flop = 0
    ############ efficiency

    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="isimoe",
        fusion=fusion,
    )

    plot_interaction_loss_curves(
        args,
        plotting_interaction_losses=plotting_interaction_losses,
        framework="isimoe",
        fusion=fusion,
    )
    if track_mir_curve:
        mir_csv_path, mir_fig_path = _save_mir_curve(
            args, mir_records, seed, fusion, num_modalities
        )
        if mir_csv_path is not None:
            print(f"MIR curve values saved to {mir_csv_path}")
            print(f"MIR curve figure saved to {mir_fig_path}")
    # Save the best model
    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/isimoe/{fusion}/{args.data}").mkdir(exist_ok=True, parents=True)

        if args.data == "mmimdb":
            save_path = f"./saves/isimoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
        elif args.data == "mosi_regression":
            save_path = f"./saves/isimoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_loss_{best_val_loss:.2f}.pth"
        else:
            save_path = f"./saves/isimoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc:.2f}.pth"
        torch.save(
            {"ensemble_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            save_path,
        )

        print(f"Best model saved to {save_path}")

    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    test_losses = []
    all_routing_weights = []
    num_experts = len(args.modality) + 1  # 对于 Specialized 模型，N+1 个专家

    # 兼容性处理：如果用的是普通 InteractionMoE，可能专家数量不一样
    # 简单起见，动态扩展列表
    all_expert_outputs = [[] for _ in range(num_experts + 5)]

    ############ efficiency
    infer_time = 0
    ############ efficiency

    with torch.no_grad():
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        for (
                batch_samples,
                batch_ids,
                batch_labels,
                batch_mcs,
                batch_observed,
        ) in test_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            expert_outputs, routing_weights, outputs = ensemble_model.inference(
                fusion_input
            )

            # 记录 Expert Outputs，注意动态长度
            for expert_idx in range(len(expert_outputs)):
                if expert_idx < len(all_expert_outputs):
                    # 处理可能存在的嵌套 list
                    val = expert_outputs[expert_idx]
                    if isinstance(val, (list, tuple)): val = val[0]
                    all_expert_outputs[expert_idx].extend(
                        val.cpu().numpy()
                    )

            all_routing_weights.extend(routing_weights.cpu().numpy())

            if args.data == "mosi_regression":
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

            else:
                if args.data == "mmimdb":
                    preds = torch.sigmoid(outputs).round()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_ids.extend(batch_ids.cpu().numpy())

                if args.data in ["mosi", "sarcasm", "humor"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
                else:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )

    ############ efficiency
    epoch_end_time = time.time()
    infer_epoch_time = epoch_end_time - epoch_start_time
    infer_time += infer_epoch_time
    ############ efficiency

    # 截断 all_expert_outputs 到实际使用的专家数量，避免可视化空列表报错
    actual_num_experts = len(expert_outputs)
    visualize_expert_logits(
        expert_outputs, routing_weights, outputs, args, framework="isimoe", fusion=fusion
    )

    visualize_expert_logits_distribution(
        all_expert_outputs[:actual_num_experts], args, framework="isimoe", fusion=fusion
    )

    visualize_sample_weights(all_routing_weights, args, framework="isimoe", fusion=fusion)

    if args.data == "mosi_regression":
        all_binary_preds = np.array(all_preds) > 0
        all_labels = np.array(all_labels) > 0
        test_acc = accuracy_score(all_binary_preds, all_labels)
        test_mae = mean_absolute_error(all_preds, all_labels)

        now = datetime.now()
        save_dir = Path(
            f"./outputs/isimoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        # 保存时使用实际长度
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs[:actual_num_experts]))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (
            best_val_loss,
            best_val_acc,
            test_acc,
            test_mae,
            train_time / args.train_epochs,
            infer_time,
            total_flop,
            total_param,
        )
    else:
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
        test_f1_micro = f1_score(all_labels, all_preds, average="micro")
        if args.data == "enrico":
            test_auc = roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        elif args.data in ["mosi", "sarcasm", "humor"]:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "mmimdb":
            test_auc = 0
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

        now = datetime.now()
        save_dir = Path(
            f"./outputs/isimoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs[:actual_num_experts]))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (
            best_val_acc,
            best_val_f1,
            best_val_f1_micro,
            best_val_auc,
            test_acc,
            test_f1,
            test_f1_micro,
            test_auc,
            train_time / args.train_epochs,
            infer_time,
            total_flop,
            total_param,
        )
