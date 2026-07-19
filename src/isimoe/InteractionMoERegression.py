import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.common.modules.common import MLP


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
    """
    Interaction Expert.
    """

    def __init__(self, fusion_model, fusion_sparse):
        super(InteractionExpert, self).__init__()
        self.fusion_model = fusion_model
        self.fusion_sparse = fusion_sparse

    def forward(self, inputs):
        """
        Forward pass with all modalities present.
        """
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_with_replacement(self, inputs, replace_index):
        """
        Forward pass with one modality replaced by a random vector.

        Args:
            inputs (list of tensors): List of modality inputs.
            replace_index (int): Index of the modality to replace. If None, no modality is replaced.
        """
        return self._forward_with_replacement(inputs, replace_index=replace_index)

    def random_masking(self, x, ratio=0.80):
        """
        Randomly mask tokens in the input tensor.
        The ratio controls the fraction of tokens to replace.
        """
        N, L, E = x.shape
        dim_feature = 2
        len_keep = int(L * (1 - ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(dim_feature)
        return mask

    def _forward_with_replacement(self, inputs, replace_index=None):
        """
        Internal function to handle forward pass with optional modality replacement.
        """
        if replace_index is not None:
            mask = self.random_masking(inputs[replace_index])
            mask_tokens = torch.zeros(
                1, 1, inputs[replace_index].size(-1), device=inputs[replace_index].device
            )
            random_vector = inputs[replace_index] * mask
            inputs = (
                inputs[:replace_index] + [random_vector] + inputs[replace_index + 1:]
            )

        x = self.fusion_model(inputs)
        if self.fusion_sparse:
            return x, self.fusion_model.gate_loss()

        return x

    def forward_multiple(self, inputs):
        """
        Perform (1 + n) forward passes: one with all modalities and one for each modality replaced.

        Args:
            inputs (list of tensors): List of modality inputs.

        Returns:
            List of outputs from the forward passes.
        """
        outputs = []
        if self.fusion_sparse:
            gate_losses = []

            output, gate_loss = self.forward(inputs)
            outputs.append(output)
            gate_losses.append(gate_loss)

            for i in range(len(inputs)):
                output, gate_loss = self.forward_with_replacement(
                    inputs, replace_index=i
                )
                outputs.append(output)
                gate_losses.append(gate_loss)

            return outputs, gate_losses
        else:
            outputs.append(self.forward(inputs))

        for i in range(len(inputs)):
            outputs.append(self.forward_with_replacement(inputs, replace_index=i))

        return outputs


# Sparse cross-modal interaction
class AMSSControlledSparseDeltaCrossAttention(nn.Module):
    def __init__(self,
                 num_modalities,
                 d_model,
                 num_heads=4,
                 dropout=0.1,
                 use_proj_q=True,
                 amss_tau=1.0,
                 amss_momentum=0.9,
                 initial_topk_ratio=0.5):
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_proj_q = use_proj_q
        self.amss_tau = amss_tau
        self.amss_momentum = amss_momentum
        self.initial_topk_ratio = initial_topk_ratio

        self.register_buffer("cached_rho", torch.ones(num_modalities) * 0.5)
        self.register_buffer("u_mom", torch.zeros(num_modalities))

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)
        ])

        if self.use_proj_q:
            self.q_projs = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_modalities)
            ])

        self.post_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])

        self.norm = nn.LayerNorm(d_model)

    def update_rho_cache(self, rho_list, ema_coeff=0.1):
        if isinstance(rho_list, list):
            rho_tensor = torch.tensor(rho_list, dtype=torch.float32, device=self.cached_rho.device)
        else:
            rho_tensor = rho_list.to(self.cached_rho.device)
        rho_tensor = rho_tensor.clamp(0.1, 1.0)
        self.cached_rho = self.cached_rho * ema_coeff + rho_tensor * (1 - ema_coeff)

    def get_topk_ratio_per_modality(self):
        return self.cached_rho.clamp(0.1, 0.9)

    def forward(self, inputs, modal_ratios=None):
        assert len(inputs) == self.num_modalities
        device = inputs[0].device

        # Derive token budgets from the previous batch's ratios.
        topk_ratios = self.get_topk_ratio_per_modality()
        topk_ratios_normalized = torch.softmax(topk_ratios / 0.1, dim=0)


        target_sum = 1.5
        topk_ratios_scaled = topk_ratios_normalized * target_sum
        topk_ratios_scaled = torch.clamp(topk_ratios_scaled, min=0.1, max=1.0)
        topk_ratios = topk_ratios_scaled

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
            Lctx = context.shape[1]

            _, attn_weights = self.cross_attns[i](q, context, context, need_weights=True)
            k = max(1, min(int(topk_ratios[i].item() * Lctx), Lctx))

            topk_vals, topk_idx = torch.topk(attn_weights, k=k, dim=-1)
            context_flat = context.reshape(-1, D)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2).expand(B, Lq, k)
            topk_idx_flat = batch_idx * Lctx + topk_idx
            topk_idx_flat = topk_idx_flat.reshape(-1)
            topk_context_flat = context_flat[topk_idx_flat]
            topk_context = topk_context_flat.reshape(B, Lq, k, D)
            topk_weights = F.softmax(topk_vals, dim=-1)
            sparse_attn_out = torch.einsum('blk,blkd->bld', topk_weights, topk_context)

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

    def update_amss_momentum(self, ratios, suppress_warning=False):
        if isinstance(ratios, (list, tuple)):
            ratios = torch.tensor(ratios, dtype=torch.float32, device=self.u_mom.device)
        elif not isinstance(ratios, torch.Tensor):
            raise TypeError(f"ratios必须是列表/元组/张量，当前类型：{type(ratios)}")

        if ratios.dim() == 0:
            ratios = ratios.unsqueeze(0)
        elif ratios.dim() > 1:
            ratios = ratios.flatten()

        if len(ratios) > self.num_modalities:
            if not suppress_warning:
                print(f"提示：自动截取前{self.num_modalities}个独特专家比例")
            ratios = ratios[:self.num_modalities]
        elif len(ratios) < self.num_modalities:
            if not suppress_warning:
                print(f"警告：ratios长度{len(ratios)}不足，自动补齐到{self.num_modalities}")
            pad_len = self.num_modalities - len(ratios)
            pad_val = ratios.mean() if len(ratios) > 0 else 0.0
            ratios = torch.cat([
                ratios,
                torch.full((pad_len,), pad_val, device=ratios.device, dtype=ratios.dtype)
            ])

        current_val = ratios.clone().detach()
        self.u_mom.mul_(self.amss_momentum).add_((1.0 - self.amss_momentum) * current_val)


# Specialized interaction mixture of experts for regression
class SpecializedInteractionMoERegression(nn.Module):
    def __init__(
            self,
            num_modalities=3,
            fusion_model=None,
            fusion_sparse=True,
            hidden_dim=256,
            hidden_dim_rw=256,
            num_layer_rw=2,
            temperature_rw=1,
            topk_ratio=0.2,
            amss_enabled=False,
            amss_tau=1.0,
            amss_momentum=0.9,
            scale_factor=1.0,
            use_interaction=True,
            # Sparse interaction settings.
            delta_attn_num_heads=4,
            delta_attn_dropout=0.1,
            use_proj_q=True,
            initial_topk_ratio=0.5
    ):
        super(SpecializedInteractionMoERegression, self).__init__()
        self.num_modalities = num_modalities
        self.amss_enabled = amss_enabled
        self.amss_tau = amss_tau
        self.amss_tau = amss_tau
        self.amss_momentum = amss_momentum
        self.scale_factor = scale_factor
        self.use_interaction = use_interaction

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
                initial_topk_ratio=initial_topk_ratio
            )
        else:
            self.interaction_module = None

        # Gate expert contributions.
        num_branches = num_modalities + 1
        self.reweight = MLPReWeighting(
            num_modalities,
            num_branches,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layers=num_layer_rw,
            temperature=temperature_rw,
        )

        # Build unimodal and shared experts.
        self.specialized_experts = nn.ModuleList()
        for _ in range(num_modalities):
            spec_model = deepcopy(fusion_model)

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

            self.specialized_experts.append(
                InteractionExpert(spec_model, fusion_sparse)
            )

        self.shared_expert = InteractionExpert(deepcopy(fusion_model), fusion_sparse)
        self.fusion_sparse = fusion_sparse
        self.expert_outputs = None

    def forward(self, inputs):
        assert len(inputs) == self.num_modalities
        device = inputs[0].device

        # Compute cross-modally enhanced features.
        enhanced_features = []
        if self.use_interaction and self.interaction_module is not None:
            modal_ratios = None
            if self.amss_enabled and hasattr(self, 'u_mom'):
                modal_ratios = self.u_mom[:self.num_modalities].cpu().tolist()
            enhanced_features = self.interaction_module(inputs, modal_ratios)
        else:
            enhanced_features = [torch.zeros_like(x) for x in inputs]

        # Run unimodal and shared experts.
        expert_outputs = []
        gate_losses = []
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

        if self.fusion_sparse:
            out_s, g_loss_s = self.shared_expert.forward(inputs)
            gate_losses.append(g_loss_s)
        else:
            out_s = self.shared_expert.forward(inputs)
        expert_outputs.append(out_s)
        self.expert_outputs = expert_outputs

        # Fuse expert predictions.
        interaction_losses = [torch.tensor(0.0, device=device) for _ in range(self.num_modalities + 1)]
        all_preds = torch.stack(expert_outputs, dim=1)
        interaction_weights = self.reweight(inputs)
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_preds = (all_preds * weights_transposed).sum(dim=1)
        wrapped_outputs = [[out] for out in expert_outputs]

        if self.fusion_sparse:
            return (
                wrapped_outputs,
                interaction_weights,
                weighted_preds,
                interaction_losses,
                gate_losses,
            )

        return (
            wrapped_outputs,
            interaction_weights,
            weighted_preds,
            interaction_losses
        )

    def apply_mi_fisher_after_backward(self, inputs, labels):
        """Apply regression-aware Fisher masking and update cached ratios."""
        if not self.amss_enabled:
            return {}

        from src.isimoe.mi_fisher_utils import apply_fisher_freeze, modal_caculate_multi_mi_regression

        # Collect expert predictions.
        expert_outputs = self.expert_outputs
        expert_inputs = inputs + [inputs[0]]

        # Estimate regression significance.
        coeffs, ratios = modal_caculate_multi_mi_regression(
            expert_outputs=expert_outputs,
            labels=labels,
            temperature=self.amss_tau
        )

        # Update momentum statistics.
        m = self.amss_momentum
        unique_ratios = ratios[:self.num_modalities]
        shared_ratio = ratios[-1]

        if isinstance(unique_ratios, list):
            all_ratios = unique_ratios + [shared_ratio]
            current_val = torch.tensor(all_ratios).clone().detach().to(self.u_mom.device)
        else:
            if isinstance(shared_ratio, torch.Tensor):
                shared_ratio_t = shared_ratio.unsqueeze(0) if shared_ratio.dim() == 0 else shared_ratio
            else:
                shared_ratio_t = torch.tensor([shared_ratio], device=unique_ratios.device)
            current_val = torch.cat([unique_ratios, shared_ratio_t]).clone().detach().to(self.u_mom.device)

        if self.u_mom.numel() != current_val.numel():
            self.u_mom = torch.zeros(current_val.numel(), device=self.u_mom.device)

        self.u_mom.mul_(m).add_((1.0 - m) * current_val)
        s = torch.softmax(self.u_mom / self.amss_tau, dim=0)

        # Compute modality ratios and refresh the interaction cache.
        rho_list = []
        stats = {}
        for i in range(self.num_modalities):
            rho = float(1.0 - s[i] * self.scale_factor)
            rho = max(0.1, min(rho, 1.0))
            rho_list.append(rho)

            expert = self.specialized_experts[i]
            attention_params = []
            for name, param in expert.named_parameters():
                if param.requires_grad and 'attention' in name:
                    attention_params.append(param)
            n_attn_param = sum(p.numel() for p in attention_params)
            p_keep = int(rho * n_attn_param)
            p_freeze = max(0, n_attn_param - p_keep)
            frozen_count = apply_fisher_freeze(attention_params, p_freeze)

            stats[f"rho_m{i}"] = rho
            stats[f"ratio_m{i}"] = float(unique_ratios[i])
            stats[f"frozen_m{i}"] = frozen_count

        # Cache ratios for the next batch.
        if self.use_interaction and self.interaction_module is not None:
            self.interaction_module.update_rho_cache(rho_list)
            self.interaction_module.update_amss_momentum(unique_ratios, suppress_warning=True)
        # =========================================================

        # Update shared-expert statistics.
        idx_shared = self.num_modalities
        rho_shared = float(1.0 - s[idx_shared] * self.scale_factor)
        rho_shared = max(0.1, min(rho_shared, 1.0))

        shared_expert = self.shared_expert
        shared_attn_params = []
        for name, param in shared_expert.named_parameters():
            if param.requires_grad and 'attention' in name:
                shared_attn_params.append(param)
        n_shared_param = sum(p.numel() for p in shared_attn_params)
        p_keep_shared = int(rho_shared * n_shared_param)
        p_freeze_shared = max(0, n_shared_param - p_keep_shared)
        frozen_shared = apply_fisher_freeze(shared_attn_params, p_freeze_shared)

        stats["rho_shared"] = rho_shared
        stats["frozen_shared"] = frozen_shared

        return stats

    def inference(self, inputs):
        if self.use_interaction and self.interaction_module is not None:
            enhanced_features = self.interaction_module(inputs, modal_ratios=None)
        else:
            enhanced_features = [torch.zeros_like(x) for x in inputs]

        expert_outputs = []
        for i in range(self.num_modalities):
            curr_feat = inputs[i]
            feat_enhanced = enhanced_features[i]
            spec_input = [curr_feat, feat_enhanced]
            out = self.specialized_experts[i].forward(spec_input)
            if isinstance(out, (tuple, list)):
                out = out[0]
            expert_outputs.append(out)

        out_s = self.shared_expert.forward(inputs)
        if isinstance(out_s, (tuple, list)):
            out_s = out_s[0]
        expert_outputs.append(out_s)

        all_preds = torch.stack(expert_outputs, dim=1)
        interaction_weights = self.reweight(inputs)
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_preds = (all_preds * weights_transposed).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_preds
