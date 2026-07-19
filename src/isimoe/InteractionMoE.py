import os
import sys
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.common.modules.common import MLP, AttentionBRFF



# Sparse cross-modal interaction
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSSControlledSparseDeltaCrossAttention(nn.Module):
    def __init__(self,
                 num_modalities,
                 d_model,
                 num_heads=4,
                 dropout=0.1,
                 use_proj_q=True,
                 amss_tau=1.0,
                 amss_momentum=0.9,
                 initial_topk_ratio=0.5,
                 data="adni",
                 ):
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_proj_q = use_proj_q
        self.amss_tau = amss_tau
        self.amss_momentum = amss_momentum
        self.initial_topk_ratio = initial_topk_ratio
        self.data = data
        # Cache the previous batch's modality ratios.
        # Use neutral defaults for the first forward pass.
        self.register_buffer("cached_rho", torch.ones(num_modalities) * 0.5)


        # Shared cross-attention projections.
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)])

        # Modality-specific query projections.
        if self.use_proj_q:
            self.q_projs = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_modalities)
            ])

        # 4. Token-wise Post-Gate
        self.post_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])

        self.norm = nn.LayerNorm(d_model)

    # Cache current ratios for the next batch.
    def update_rho_cache(self, rho_list, ema_coeff=0.1):
        """
        由主模型在apply_mi_fisher_after_backward中调用，更新rho缓存
        Args:
            rho_list: list/tensor，长度=num_modalities（2个），值范围0.1~1.0
        """

        """使用EMA，让rho缓存更新更平滑"""
        if isinstance(rho_list, list):
            rho_tensor = torch.tensor(rho_list, dtype=torch.float32, device=self.cached_rho.device)
        else:
            rho_tensor = rho_list.to(self.cached_rho.device)
        rho_tensor = rho_tensor.clamp(0.1, 1.0)

        # Smooth ratio updates with an exponential moving average.
        self.cached_rho = self.cached_rho * ema_coeff + rho_tensor * (1 - ema_coeff)

    def get_topk_ratio_per_modality(self):
        """前向时：直接用缓存的rho（上一批）计算Top-k比例"""
        return self.cached_rho.clamp(0.1, 0.9)

    def forward(self, inputs, modal_ratios=None):
        """前向逻辑：全程用缓存的rho（上一批）"""
        assert len(inputs) == self.num_modalities
        device = inputs[0].device

        # Derive token budgets from the previous batch's ratios.
        topk_ratios = self.get_topk_ratio_per_modality()
        topk_ratios_normalized =torch.softmax(topk_ratios / 0.1, dim=0)

        # Match the budget list to the modality count.
        if(self.data == "adni"):
            target_sum = 2
        elif (self.data == "mosi"):
            target_sum = 1.5
        else:
            target_sum = 1

        topk_ratios_scaled = topk_ratios_normalized * target_sum
        topk_ratios_scaled = torch.clamp(topk_ratios_scaled, min=0.1, max=1.0)
        topk_ratios=topk_ratios_scaled


        # Normalize input shapes before cross-modal attention.
        processed = []
        is_2d = []
        for x in inputs:
            is_2d.append(x.dim() == 2)
            if x.dim() == 2:
                processed.append(x.unsqueeze(1))
            else:
                processed.append(x)

        enhanced_outputs = []

        for i in range(self.num_modalities):
            q = processed[i]
            B, Lq, D = q.shape

            context = torch.cat([processed[j] for j in range(self.num_modalities) if j != i], dim=1)
            if context.numel() == 0:
                out = q.squeeze(1) if is_2d[i] else q
                enhanced_outputs.append(out)
                continue
            # Context contains all tokens from the other modalities.
            Lctx = context.shape[1]

            # Convert the cached ratio into a top-k token budget.
            _, attn_weights = self.cross_attns[i](q, context, context, need_weights=True)
            # A larger ratio preserves more tokens for a weaker modality.
            k = max(1, min(int(topk_ratios[i].item() * Lctx), Lctx))
            # =========================================================

            # Select the most relevant context tokens.
            topk_vals, topk_idx = torch.topk(attn_weights, k=k, dim=-1)
            context_flat = context.reshape(-1, D)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2).expand(B, Lq, k)
            topk_idx_flat = batch_idx * Lctx + topk_idx
            topk_idx_flat = topk_idx_flat.reshape(-1)
            topk_context_flat = context_flat[topk_idx_flat]
            topk_context = topk_context_flat.reshape(B, Lq, k, D)
            topk_weights = F.softmax(topk_vals, dim=-1)
            sparse_attn_out = torch.einsum('blk,blkd->bld', topk_weights, topk_context)

            # Inject the attended residual through a learned gate.
            if self.use_proj_q:
                proj_q = self.q_projs[i](q)
                delta = sparse_attn_out - proj_q
            else:
                delta = sparse_attn_out - q
            gate = self.post_gates[i](delta)
            enhanced_q = q + gate * delta

            enhanced_q = self.norm(enhanced_q)
            if is_2d[i]:
                enhanced_q = enhanced_q.squeeze(1)
            enhanced_outputs.append(enhanced_q)

        return enhanced_outputs



# Expert components
class MLPReWeighting(nn.Module):
    """Use MLP to re-weight all interaction experts."""

    def __init__(
            self,
            num_modalities,
            num_branches,
            hidden_dim=256,
            hidden_dim_rw=256,
            num_layers=2,
            temperature=1,
    ):
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=1)

    def forward(self, inputs):
        if inputs[0].dim() == 3:
            x = [item.mean(dim=1) for item in inputs]
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat(inputs, dim=1)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)


class InteractionExpert(nn.Module):
    """Interaction Expert."""

    def __init__(self, fusion_model, fusion_sparse):
        super(InteractionExpert, self).__init__()
        self.fusion_model = fusion_model
        self.fusion_sparse = fusion_sparse
        self.last_latent = None
        self.supports_return_latent = self._supports_return_latent()

    def _supports_return_latent(self):
        try:
            params = inspect.signature(self.fusion_model.forward).parameters
        except (TypeError, ValueError):
            return False
        has_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        return "return_latent" in params or has_kwargs

    def forward(self, inputs):
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_with_replacement(self, inputs, replace_index):
        return self._forward_with_replacement(inputs, replace_index=replace_index)

    def _forward_with_replacement(self, inputs, replace_index=None):
        if replace_index is not None:
            random_vector = torch.randn_like(inputs[replace_index])
            inputs = inputs[:replace_index] + [random_vector] + inputs[replace_index + 1:]

        self.last_latent = None
        if self.supports_return_latent:
            x = self.fusion_model(inputs, return_latent=True)
        else:
            x = self.fusion_model(inputs)
        if isinstance(x, tuple) and len(x) == 2:
            x, self.last_latent = x
        if self.fusion_sparse:
            return x, self.fusion_model.gate_loss() if hasattr(self.fusion_model, 'gate_loss') else 0.0
        return x

    def forward_multiple(self, inputs):
        outputs = []
        gate_losses = [] if self.fusion_sparse else None

        if self.fusion_sparse:
            output, gate_loss = self.forward(inputs)
            outputs.append(output)
            gate_losses.append(gate_loss)
        else:
            outputs.append(self.forward(inputs))

        for i in range(len(inputs)):
            if self.fusion_sparse:
                output, gate_loss = self.forward_with_replacement(inputs, replace_index=i)
                outputs.append(output)
                gate_losses.append(gate_loss)
            else:
                outputs.append(self.forward_with_replacement(inputs, replace_index=i))

        if self.fusion_sparse:
            return outputs, gate_losses
        return outputs


# Specialized interaction mixture of experts
class SpecializedInteractionMoE(nn.Module):
    """
    完整整合版：
    - 替换原有交互模块为 AMSSControlledSparseDeltaCrossAttention
    - 保留原有 MIR/AMSS/Fisher 逻辑
    - 实现动态 topk_ratio + Delta 注入 + Token-wise Post-Gate
    """

    def __init__(
            self,
            num_modalities=2,
            fusion_model=None,
            fusion_sparse=True,
            hidden_dim=256,
            hidden_dim_rw=256,
            num_layer_rw=2,
            temperature_rw=1,
            topk_ratio=0.2,
            amss_enabled=True,
            amss_tau=1.0,
            amss_momentum=0.9,
            scale_factor=1.0,
            use_interaction=True,
            # Sparse interaction settings.
            delta_attn_num_heads=4,
            delta_attn_dropout=0.1,
            use_proj_q=True,
            initial_topk_ratio=0.5,
            enable_r_path=False,
            data="adni",
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.topk_ratio = topk_ratio
        self.amss_enabled = amss_enabled
        self.amss_tau = amss_tau
        self.amss_momentum = amss_momentum
        self.scale_factor = scale_factor
        self.use_interaction = use_interaction
        self.enable_r_path = enable_r_path

        # Track one momentum value per unimodal expert plus the shared expert.
        self.register_buffer("u_mom", torch.zeros(num_modalities + 1))

        # Enhance each modality with sparse cross-modal context.
        if self.use_interaction:
            self.interaction_module = AMSSControlledSparseDeltaCrossAttention(
                num_modalities=num_modalities,
                d_model=hidden_dim,
                num_heads=delta_attn_num_heads,
                dropout=delta_attn_dropout,
                use_proj_q=use_proj_q,
                amss_tau=amss_tau,
                amss_momentum=amss_momentum,
                initial_topk_ratio=initial_topk_ratio,
                data=data,
            )
        else:
            self.interaction_module = None

        # Reweight expert predictions with an MLP.
        self.num_branches = num_modalities + 1 + int(self.enable_r_path)
        self.reweight = MLPReWeighting(
            num_modalities,
            self.num_branches,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layers=num_layer_rw,
            temperature=temperature_rw,
        )

        # Build unimodal and shared experts.
        self.specialized_experts = nn.ModuleList()
        for _ in range(num_modalities):
            spec_model = deepcopy(fusion_model)
            # Adapt positional embeddings and prediction heads.
            if hasattr(spec_model, "pos_embed") and spec_model.pos_embed is not None:
                total_len = spec_model.pos_embed.shape[1]
                patches_per_modality = total_len // num_modalities
                target_len = patches_per_modality * 2
                if target_len != total_len:
                    new_pos = spec_model.pos_embed.data[:, :target_len, :]
                    spec_model.pos_embed = nn.Parameter(new_pos)
            if hasattr(spec_model, "network") and len(spec_model.network) > 0:
                predictor = spec_model.network[-1]
                if hasattr(predictor, "fc") and isinstance(predictor.fc, nn.Linear):
                    old_in = predictor.fc.in_features
                    old_out = predictor.fc.out_features
                    if old_in % num_modalities == 0:
                        new_in = int((old_in // num_modalities) * 2)
                        predictor.fc = nn.Linear(new_in, old_out)
                        if predictor.fc.bias is not None:
                            nn.init.constant_(predictor.fc.bias, 0)
            self.specialized_experts.append(InteractionExpert(spec_model, fusion_sparse))

        self.shared_expert = InteractionExpert(deepcopy(fusion_model), fusion_sparse)
        if self.enable_r_path:
            self.redundancy_expert = InteractionExpert(deepcopy(fusion_model), fusion_sparse)
        self.fusion_sparse = fusion_sparse
        self.expert_outputs = None  # Retain expert outputs for MIR estimation.

    def redundancy_loss(self, anchor, positives):
        total_redundancy_loss = 0
        if anchor.dim() > 2:
            anchor = anchor.flatten(start_dim=1)
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for positive in positives:
            if positive.dim() > 2:
                positive = positive.flatten(start_dim=1)
            positive_normalized = F.normalize(positive, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * positive_normalized, dim=1)
            total_redundancy_loss += torch.mean(1 - cosine_sim)
        return total_redundancy_loss / len(positives)

    def forward(self, inputs, labels=None):
        assert len(inputs) == self.num_modalities
        device = inputs[0].device

        # Compute cross-modally enhanced features.
        enhanced_features = []
        if self.use_interaction and self.interaction_module is not None:

            # Apply sparse cross-modal interaction.
            enhanced_features = self.interaction_module(inputs)
        else:
            enhanced_features = [torch.zeros_like(x) for x in inputs]

        # Run unimodal and shared experts.
        expert_outputs = []
        expert_features = []
        gate_losses = []
        # Unimodal experts.
        for i in range(self.num_modalities):
            curr_feat = inputs[i]
            feat_enhanced = enhanced_features[i]
            spec_input = [curr_feat, feat_enhanced]
            if self.fusion_sparse:
                out, g_loss = self.specialized_experts[i].forward(spec_input)
                gate_losses.append(g_loss)
            else:
                out = self.specialized_experts[i].forward(spec_input)
            expert_outputs.append(out)
            expert_features.append(self.specialized_experts[i].last_latent)
        # Shared expert.
        if self.fusion_sparse:
            out_s, g_loss_s = self.shared_expert.forward(inputs)
            gate_losses.append(g_loss_s)
        else:
            out_s = self.shared_expert.forward(inputs)
        expert_outputs.append(out_s)
        expert_features.append(self.shared_expert.last_latent)

        redundancy_loss = torch.tensor(0.0, device=device)
        if self.enable_r_path:
            if self.fusion_sparse:
                redundancy_outputs, redundancy_gate_losses = self.redundancy_expert.forward_multiple(inputs)
                gate_losses.append(torch.stack([
                    g.to(device) if isinstance(g, torch.Tensor) else torch.as_tensor(g, device=device)
                    for g in redundancy_gate_losses
                ]).mean())
            else:
                redundancy_outputs = self.redundancy_expert.forward_multiple(inputs)
            redundancy_anchor = redundancy_outputs[0]
            redundancy_positives = redundancy_outputs[1:]
            redundancy_loss = self.redundancy_loss(redundancy_anchor, redundancy_positives)
            expert_outputs.append(redundancy_anchor)
            expert_features.append(self.redundancy_expert.last_latent)
        self.expert_outputs = expert_outputs
        self.expert_features = expert_features

        # Reweight expert predictions.
        interaction_weights = self.reweight(inputs)
        all_preds = torch.stack(expert_outputs, dim=1)
        weighted_preds = (all_preds * interaction_weights.unsqueeze(2)).sum(dim=1)
        wrapped_outputs = [[out] for out in expert_outputs]
        interaction_losses = [torch.tensor(0.0, device=device) for _ in range(self.num_modalities + 1)]
        if self.enable_r_path:
            interaction_losses.append(redundancy_loss)

        # Return fused predictions and auxiliary outputs.
        if self.fusion_sparse:
            return wrapped_outputs, interaction_weights, weighted_preds, interaction_losses, gate_losses
        return wrapped_outputs, interaction_weights, weighted_preds, interaction_losses

    def apply_mi_fisher_after_backward(self, inputs, labels):
        """计算rho后，更新交互模块的rho缓存（供下一批前向使用）"""
        if not self.amss_enabled:
            return {}


        from src.isimoe.mi_fisher_utils import modal_caculate_multi_mi


        # Estimate per-modality MIR scores.
        expert_outputs = self.expert_outputs
        expert_inputs = inputs + [inputs[0]]
        scores, ratios, hy = modal_caculate_multi_mi(
            expert_outputs=expert_outputs,
            expert_inputs=expert_inputs,
            labels=labels,
            temperature=self.amss_tau
        )
        # Keep the unimodal expert ratios.
        if isinstance(ratios, (list, tuple)):
            unique_ratios = ratios[:self.num_modalities]
        elif isinstance(ratios, torch.Tensor):
            unique_ratios = ratios[:self.num_modalities]
        if len(unique_ratios) < self.num_modalities:
            pad_val = unique_ratios.mean() if len(unique_ratios) > 0 else 0.0
            if isinstance(unique_ratios, list):
                unique_ratios += [pad_val] * (self.num_modalities - len(unique_ratios))
            else:
                unique_ratios = torch.cat([
                    unique_ratios,
                    torch.full((self.num_modalities - len(unique_ratios),), pad_val,
                               device=unique_ratios.device, dtype=unique_ratios.dtype)
                ])
        # Update the model-level momentum statistics.
        shared_ratio = ratios[-1] if len(ratios) > 0 else 0.0
        if isinstance(unique_ratios, list):
            current_ratios_list = unique_ratios + [shared_ratio]
            current_ratios_tensor = torch.as_tensor(current_ratios_list, device=self.u_mom.device)
        else:
            current_ratios_tensor = torch.cat([
                unique_ratios,
                torch.tensor([shared_ratio], device=unique_ratios.device)
            ], dim=0)
        m = self.amss_momentum
        current_val = current_ratios_tensor.clone().detach().to(self.u_mom.device)
        self.u_mom.mul_(m).add_((1.0 - m) * current_val)
        s = torch.softmax(self.u_mom / self.amss_tau, dim=0)

        # Compute the current modality ratios.
        rho_list = []
        stats = {}
        for i in range(self.num_modalities):
            rho=1.0-ratios[i]
            rho = max(0.1, min(rho, 1.0))
            rho_list.append(rho)  # Collect the current batch ratio.
            stats[f"mir_m{i}"] = ratios[i]
            stats[f"rho_m{i}"] = rho
            stats[f"score_m{i}"] = scores[i] if i < len(scores) else 0.0
        # Make current ratios available to the next batch.
        if self.use_interaction and self.interaction_module is not None:
            # Update the interaction cache.
            self.interaction_module.update_rho_cache(rho_list)

        # ================================================================

        # Update shared-expert statistics.
        idx_shared = self.num_modalities
        rho_shared = float(1.0 - s[idx_shared] * self.scale_factor) if idx_shared < len(s) else 0.5
        rho_shared = max(0.1, min(rho_shared, 1.0))
        stats["rho_shared"] = rho_shared
        stats["score_shared"] = scores[-1] if len(scores) > 0 else 0.0
        stats["label_entropy"] = hy

        return stats

    def inference(self, inputs):
        """推理阶段使用默认参数"""
        if self.use_interaction and self.interaction_module is not None:
            enhanced_features = self.interaction_module(inputs, modal_ratios=None)
        else:
            enhanced_features = [torch.zeros_like(x) for x in inputs]

        expert_outputs = []
        # Unimodal experts.
        for i in range(self.num_modalities):
            curr_feat = inputs[i]
            feat_enhanced = enhanced_features[i]
            spec_input = [curr_feat, feat_enhanced]
            out = self.specialized_experts[i].forward(spec_input)
            if isinstance(out, (tuple, list)):
                out = out[0]
            expert_outputs.append(out)
        # Shared expert.
        out_s = self.shared_expert.forward(inputs)
        if isinstance(out_s, (tuple, list)):
            out_s = out_s[0]
        expert_outputs.append(out_s)

        if self.enable_r_path:
            out_r = self.redundancy_expert.forward(inputs)
            if isinstance(out_r, (tuple, list)):
                out_r = out_r[0]
            expert_outputs.append(out_r)

        # Reweight expert predictions.
        interaction_weights = self.reweight(inputs)
        all_preds = torch.stack(expert_outputs, dim=1)
        weighted_preds = (all_preds * interaction_weights.unsqueeze(2)).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_preds
