import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def modal_caculate_multi_mi(expert_outputs, expert_inputs, labels, temperature=1.0):
    """
    计算多模态互信息系数 (Multi-modal MI Coefficients)。
    核心修改：将原expert_raw_logits替换为expert_inputs，基于输入特征计算模态熵。

    Args:
        expert_outputs (List[Tensor]): 经过温度缩放或其他处理后的专家输出 logits (用于计算 Loss)。
        expert_inputs (List[Tensor]): 模态的原始输入特征 (用于计算 Entropy)，支持形状：
                                      - 2D: [Batch, Dim]
                                      - 3D: [Batch, SeqLen, Dim]
        labels (Tensor): 真实标签 [Batch, Classes] 或 [Batch]。
        temperature (float): AMSS 的全局温度系数。

    Returns:
        coeffs (Tensor): 每个模态的系数 [num_modalities]。
        ratios (List[Tensor]): 每个模态的 MIR 比例 (Tensor类型，保持梯度)。
        HY (Tensor): 标签的熵值
    """
    num_modalities = len(expert_outputs)-1
    device = expert_outputs[0].device

    def limit_number(num):
        if torch.isnan(num):
            return torch.tensor(0.5, device=device)
        return torch.clamp(num, 0.1, 0.9)

    # 1. 处理标签和标签熵 (H(Y))
    if labels.dim() > 1:
        label_single = torch.argmax(labels, dim=1)
    else:
        label_single = labels

    # 计算标签分布的熵
    label_counts = torch.bincount(label_single)
    label_probs = label_counts.float() / len(label_single)
    label_probs = torch.clamp(label_probs, min=1e-8)
    criterion = nn.CrossEntropyLoss()
    HY = criterion(label_probs.unsqueeze(0), label_probs.unsqueeze(0))

    # 2. 循环计算每个模态的分数和熵
    scores = []       # 每个模态的预测准确度分数 (I(X;Y)近似)
    entropies = []    # 每个模态的不确定性熵 (H(X)近似)
    ratios = []       # 每个模态的MIR比例
    exp_t_list = []   # 存储每个模态的 exp(ratio_i / temperature)

    for i in range(num_modalities):
        # 计算Score：衡量预测准确度（逻辑不变）
        score_i = -criterion(expert_outputs[i], label_single)
        scores.append(score_i)

        # ========== 核心修改：用expert_inputs计算模态熵 ==========
        # 步骤1：处理输入特征维度（兼容2D/3D特征）
        input_feat = expert_inputs[i]
        if input_feat.dim() == 3:  # [Batch, SeqLen, Dim] → 降维为 [Batch, Dim]
            # 可选策略：取序列均值（也可根据需求用max/last token）
            input_feat = torch.mean(input_feat, dim=1)
        # 确保最终是2D特征 [Batch, Dim]
        assert input_feat.dim() == 2, f"模态{i}的输入特征需为2D/3D，当前维度：{input_feat.dim()}"

        # 步骤2：计算输入特征的概率分布（用于熵计算）
        # 方式1：对特征做L2归一化后softmax（适用于特征维度为类别数的场景）
        # feat_norm = F.normalize(input_feat, p=2, dim=1)
        # probs = F.softmax(feat_norm, dim=1)

        # 方式2：对特征做均值归一化（通用场景，避免维度不匹配）
        # 先将特征缩放到0~1范围，再归一化
        feat_min = input_feat.min(dim=1, keepdim=True)[0]
        feat_max = input_feat.max(dim=1, keepdim=True)[0]
        feat_norm = (input_feat - feat_min) / (feat_max - feat_min + 1e-8)  # 避免除0
        probs = feat_norm / torch.sum(feat_norm, dim=1, keepdim=True)  # 归一化到和为1
        probs = torch.clamp(probs, min=1e-8)  # 避免log(0)

        # 步骤3：计算模态熵 H(X) = -sum(p * log(p))
        # 手动计算熵（比criterion更通用，适配任意维度特征）
        hi = -torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        entropies.append(hi)

    # 3. 检查是否有模态预测极差（逻辑不变）
    invalid_flag = False
    for i in range(num_modalities):
        if (scores[i] + HY + 0.5) <= 0:
            invalid_flag = True
            break

    # 4. 计算MIR比例和系数（逻辑不变）
    if invalid_flag:
        # 异常情况：平分权重
        coeffs = torch.full((num_modalities,), 0.5, device=device)
        ratios = [torch.tensor(1.0, device=device) for _ in range(num_modalities)]
    else:
        # 正常情况：计算ratio和exp(t)
        for i in range(num_modalities):
            ratio_i = (scores[i] + HY + 0.5) / entropies[i]
            ratios.append(ratio_i)
            t_i = ratio_i / temperature
            exp_t_list.append(torch.exp(t_i))

        # 反向抑制系数计算
        sum_exp_all = torch.sum(torch.stack(exp_t_list))
        coeffs = []
        for i in range(num_modalities):
            sum_exp_others = sum_exp_all - exp_t_list[i]
            c_i = sum_exp_others / sum_exp_all
            c_i = limit_number(c_i)
            coeffs.append(c_i)
        coeffs = torch.stack(coeffs)

    # 调试打印
    return coeffs, ratios, HY

def modal_caculate_multi_mi_regression(expert_outputs, labels, temperature=1.0):
    """
    回归任务专用：计算多模态显著性得分。
    由于回归没有信息熵，我们使用负 MSE 作为 Score，并用误差的倒数作为 Ratio 的近似。
    """
    num_modalities = len(expert_outputs)
    device = expert_outputs[0].device

    # 统一维度为 [Batch, 1]
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    criterion = nn.MSELoss()
    scores = []
    ratios = []

    for i in range(num_modalities):
        out = expert_outputs[i]
        if out.dim() == 1:
            out = out.unsqueeze(1)

        # 计算 MSE
        mse = criterion(out, labels)
        # Score 越接近 0 越好
        score_i = -mse
        scores.append(score_i)

        # Ratio 近似：Score 越高（误差越小），Ratio 越高
        # 加 1e-6 防止除以 0
        ratio_i = 1.0 / (mse + 1e-6)
        ratios.append(ratio_i)

    # 计算权重系数 (反向抑制逻辑)
    # 误差越小的模态，exp_t 越大，最终得到的 coeff 越小（从而 rho 越小，冻结越多）
    exp_t = torch.exp(torch.stack(ratios) / temperature)
    sum_exp = torch.sum(exp_t)

    # 防止单模态情况下除以 0
    denom = sum_exp * (num_modalities - 1) if num_modalities > 1 else sum_exp
    coeffs = (sum_exp - exp_t) / (denom + 1e-8)
    return coeffs, ratios

def apply_fisher_freeze(attention_params, p_freeze):
    """根据梯度 Fisher 估计，将前 ``p_freeze`` 个参数的梯度置零。

    参数：
        attention_params (list): 筛选后的注意力层可训练参数列表。
        p_freeze (int): 需要冻结的参数元素数量。

    返回：
        frozen (int): 实际被置零的梯度元素数量。
    """
    # 收集梯度平方：直接遍历传入的注意力参数列表，而不是整个模块
    grads = []
    valid_params = []  # 仅保留有梯度的Attention参数
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

    # 选取梯度平方最大的 p_freeze 个参数元素
    _, idx = torch.topk(all_grads, p_freeze)

    # 在各注意力参数张量中将对应梯度置零
    cur = 0
    frozen = 0
    for p in valid_params:
        g = p.grad.view(-1)
        n = g.numel()
        # 当前参数张量内需要置零的相对索引
        rel_idx = idx[(idx >= cur) & (idx < cur + n)] - cur
        if rel_idx.numel() > 0:
            g[rel_idx] = 0.0
            frozen += rel_idx.numel()
        cur += n

    return frozen
