import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def modal_caculate_multi_mi(expert_outputs, expert_inputs, labels, temperature=1.0):
    """Estimate multimodal mutual-information coefficients.

    Args:
        expert_outputs: Expert logits used to estimate predictive scores.
        expert_inputs: Modality features shaped [batch, dim] or
            [batch, sequence, dim].
        labels: Targets shaped [batch, classes] or [batch].
        temperature: Global AMSS temperature.

    Returns:
        Modality coefficients, MIR values, and estimated label entropy.
    """
    num_modalities = len(expert_outputs)-1
    device = expert_outputs[0].device

    def limit_number(num):
        if torch.isnan(num):
            return torch.tensor(0.5, device=device)
        return torch.clamp(num, 0.1, 0.9)

    # Normalize labels and estimate label entropy.
    if labels.dim() > 1:
        label_single = torch.argmax(labels, dim=1)
    else:
        label_single = labels

    # Compute entropy from the empirical label distribution.
    label_counts = torch.bincount(label_single)
    label_probs = label_counts.float() / len(label_single)
    label_probs = torch.clamp(label_probs, min=1e-8)
    criterion = nn.CrossEntropyLoss()
    HY = criterion(label_probs.unsqueeze(0), label_probs.unsqueeze(0))

    # Estimate predictive scores and feature entropy per modality.
    scores = []       # Accuracy-based mutual-information proxy.
    entropies = []    # Feature-uncertainty proxy.
    ratios = []       # Per-modality MIR values.
    exp_t_list = []   # Temperature-scaled MIR values.

    for i in range(num_modalities):
        # Use prediction accuracy as the modality score.
        score_i = -criterion(expert_outputs[i], label_single)
        scores.append(score_i)

        # Estimate modality entropy from expert inputs.
        # Reduce sequence features to one vector per sample.
        input_feat = expert_inputs[i]
        if input_feat.dim() == 3:  # [batch, sequence, dim] -> [batch, dim]
            # Mean-pool sequence tokens.
            input_feat = torch.mean(input_feat, dim=1)
        # Flatten any remaining feature dimensions.
        assert input_feat.dim() == 2, f"模态{i}的输入特征需为2D/3D，当前维度：{input_feat.dim()}"

        # Convert features into a normalized distribution.
        # A softmax alternative is retained for logit-like features.
        # feat_norm = F.normalize(input_feat, p=2, dim=1)
        # probs = F.softmax(feat_norm, dim=1)

        # Min-max scale generic features before normalization.
        feat_min = input_feat.min(dim=1, keepdim=True)[0]
        feat_max = input_feat.max(dim=1, keepdim=True)[0]
        feat_norm = (input_feat - feat_min) / (feat_max - feat_min + 1e-8)  # Avoid division by zero.
        probs = feat_norm / torch.sum(feat_norm, dim=1, keepdim=True)  # Normalize each sample.
        probs = torch.clamp(probs, min=1e-8)  # Avoid log(0).

        # Compute feature entropy directly for arbitrary dimensions.
        hi = -torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        entropies.append(hi)

    # Detect a modality with near-random predictions.
    invalid_flag = False
    for i in range(num_modalities):
        if (scores[i] + HY + 0.5) <= 0:
            invalid_flag = True
            break

    # Convert MIR values into suppression coefficients.
    if invalid_flag:
        # Fall back to equal weights for degenerate predictions.
        coeffs = torch.full((num_modalities,), 0.5, device=device)
        ratios = [torch.tensor(1.0, device=device) for _ in range(num_modalities)]
    else:
        # Compute temperature-scaled MIR values.
        for i in range(num_modalities):
            ratio_i = (scores[i] + HY + 0.5) / entropies[i]
            ratios.append(ratio_i)
            t_i = ratio_i / temperature
            exp_t_list.append(torch.exp(t_i))

        # Invert the weights to suppress dominant modalities.
        sum_exp_all = torch.sum(torch.stack(exp_t_list))
        coeffs = []
        for i in range(num_modalities):
            sum_exp_others = sum_exp_all - exp_t_list[i]
            c_i = sum_exp_others / sum_exp_all
            c_i = limit_number(c_i)
            coeffs.append(c_i)
        coeffs = torch.stack(coeffs)

    # Optional diagnostics.
    return coeffs, ratios, HY

def modal_caculate_multi_mi_regression(expert_outputs, labels, temperature=1.0):
    """Estimate regression significance using negative and inverse MSE."""
    num_modalities = len(expert_outputs)
    device = expert_outputs[0].device

    # Normalize targets and predictions to [batch, 1].
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    criterion = nn.MSELoss()
    scores = []
    ratios = []

    for i in range(num_modalities):
        out = expert_outputs[i]
        if out.dim() == 1:
            out = out.unsqueeze(1)

        # Compute per-expert mean squared error.
        mse = criterion(out, labels)
        # Lower error yields a higher score.
        score_i = -mse
        scores.append(score_i)

        # Approximate MIR by the inverse error.
        ratio_i = 1.0 / (mse + 1e-6)
        ratios.append(ratio_i)

    # Invert weights so stronger modalities receive more suppression.
    exp_t = torch.exp(torch.stack(ratios) / temperature)
    sum_exp = torch.sum(exp_t)

    # Avoid division by zero for a single modality.
    denom = sum_exp * (num_modalities - 1) if num_modalities > 1 else sum_exp
    coeffs = (sum_exp - exp_t) / (denom + 1e-8)
    return coeffs, ratios

def apply_fisher_freeze(attention_params, p_freeze):
    """Zero gradients with the largest Fisher proxies.

    Args:
        attention_params: Selected trainable attention parameters.
        p_freeze: Number of gradient elements to suppress.

    Returns:
        Number of gradient elements set to zero.
    """
    # Collect squared gradients from the selected attention parameters.
    grads = []
    valid_params = []  # Retain only parameters with gradients.
    for p in attention_params:
        if p.grad is None:
            continue
        grads.append((p.grad.detach() ** 2).view(-1))
        valid_params.append(p)

    if len(grads) == 0:
        return 0

    all_grads = torch.cat(grads)
    total = all_grads.numel()
    p_freeze = min(max(0, int(p_freeze)), total)
    if p_freeze == 0:
        return 0

    # Select the largest-gradient elements to freeze.
    _, idx = torch.topk(all_grads, p_freeze)

    # Zero the selected gradient entries in each parameter tensor.
    cur = 0
    frozen = 0
    for p in valid_params:
        g = p.grad.view(-1)
        n = g.numel()
        # Convert global indices into offsets for this tensor.
        rel_idx = idx[(idx >= cur) & (idx < cur + n)] - cur
        if rel_idx.numel() > 0:
            g[rel_idx] = 0.0
            frozen += rel_idx.numel()
        cur += n

    return frozen
