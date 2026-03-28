import torch
import torch.nn.functional as F
import torch.nn as nn

def soft_entropy(input, target, reduction='mean'):
    """
    Cross entropy that accepts soft targets, returns scalar or per-sample loss.

    Args:
        input (Tensor): [B, C] logits
        target (Tensor): [B, C] soft targets
        reduction (str): 'mean', 'sum', or 'none'

    Returns:
        Tensor: loss (scalar if 'mean' or 'sum'; [B] if 'none')
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    log_probs = logsoftmax(input)         # [B, C]
    per_sample_loss = -(target * log_probs).sum(dim=1)  # [B]

    if reduction == 'mean':
        return per_sample_loss.mean()
    elif reduction == 'sum':
        return per_sample_loss.sum()
    elif reduction == 'none':
        return per_sample_loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")

import torch
import torch.nn.functional as F

def gate_loss_lbl(alpha, label, label_dis, gamma=1.0):
    """
    Label-aware balancing loss (improved version, Float32 safe).
    Args:
        alpha (Tensor): gating output after sigmoid, shape [B] or [B,1], float32
        label (Tensor): one-hot labels, shape [B, C]
        label_dis (Tensor): class frequencies, shape [C]
        gamma (float): scaling factor
    """
    device = alpha.device
    dtype = torch.float32

    if alpha.ndim > 1:
        alpha = alpha.view(-1)

    alpha = alpha.to(dtype)

    # 类别先验（归一化）
    label_dis = torch.tensor(label_dis, dtype=torch.float)
    pi = label_dis.to(device=device, dtype=dtype)
    pi = pi / (pi.sum() + 1e-12)

    # 均值频率
    pi_mean = pi.mean()

    # 每个类的目标 alpha*
    target_alpha_per_class = torch.sigmoid(
        gamma * (torch.log(pi + 1e-12) - torch.log(pi_mean + 1e-12))
    )

    # 从 one-hot 标签中选出对应类别的 alpha*
    target_alpha = (label.to(device=device, dtype=dtype) * target_alpha_per_class.unsqueeze(0)).sum(dim=1).detach()

    # L2 损失
    loss = F.mse_loss(alpha, target_alpha)
    return loss


def mix_outputs_old(outputs, labels, balance=False, label_dis=None):
    logits_rank = outputs[0].unsqueeze(1)
    for i in range(len(outputs) - 1):
        logits_rank = torch.cat(
            (logits_rank, outputs[i+1].unsqueeze(1)), dim=1)

    max_tea, max_idx = torch.max(logits_rank, dim=1)
    # min_tea, min_idx = torch.min(logits_rank, dim=1)

    non_target_labels = torch.ones_like(labels) - labels

    avg_logits = torch.sum(logits_rank, dim=1) / len(outputs)
    non_target_logits = (-30 * labels) + avg_logits * non_target_labels

    _hardest_nt, hn_idx = torch.max(non_target_logits, dim=1)

    hardest_idx = torch.zeros_like(labels)
    hardest_idx.scatter_(1, hn_idx.data.view(-1, 1), 1)
    hardest_logit = non_target_logits * hardest_idx

    rest_nt_logits = max_tea * (1 - hardest_idx) * (1 - labels)
    reformed_nt = rest_nt_logits + hardest_logit

    preds = [F.softmax(logits) for logits in outputs]

    reformed_non_targets = []
    for i in range(len(preds)):
        target_preds = preds[i] * labels

        target_preds = torch.sum(target_preds, dim=-1, keepdim=True)
        target_min = -30 * labels
        target_excluded_preds = F.softmax(
            outputs[i] * (1 - labels) + target_min)
        reformed_non_targets.append(target_excluded_preds)

    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0
    if balance == True:
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)
    else:
        #===================new added==========================
        moe_ce = []
        #======================================================
        for i in range(len(outputs)):
            # base ce
            loss += soft_entropy(outputs[i], labels)
            #===================new added==========================
            moe_ce.append(soft_entropy(outputs[i], labels))
            #======================================================
            # hardest negative suppression
            loss += 10.0 * \
                F.kl_div(
                    torch.log(reformed_non_targets[i]), F.softmax(reformed_nt))
            # mutual distillation loss
            for j in range(len(outputs)):
                if i != j:
                    loss += F.kl_div(F.log_softmax(outputs[i]),
                                     F.softmax(outputs[j]))
        #===================new added==========================
        # fuse_outputs_parallel
        reciprocal_sum = 0.0
    
        # 累加每个张量的倒数
        for ce in moe_ce:
            reciprocal_sum += 1 / (ce + 1e-8)  # 加平滑项避免除以零
        
        # 计算并联融合输出
        fused_output = 1/(reciprocal_sum) *len(outputs)
        
        loss += fused_output
        #======================================================

    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output


def mix_outputs(outputs, labels, balance=False, label_dis=None):
    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0
    loss_distillation = 0.0
    loss_parallel = 0.0
    if balance == True:
        moe_ce = []
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)
            # moe_ce.append(soft_entropy(outputs[i] + label_dis.log(), labels))
            # for j in range(len(outputs)):
            #     if i != j:
            #         loss_distillation += F.kl_div(F.log_softmax(outputs[i]),
            #                         F.softmax(outputs[j]))
    else:
        #===================new added==========================
        moe_ce = []
        #======================================================
        for i in range(len(outputs)):
            # NEW
            # base ce
            loss += soft_entropy(outputs[i], labels) #TODO:change it back(+ label_dis.log())
            #===================new added==========================
            moe_ce.append(soft_entropy(outputs[i], labels))
            #======================================================
            # distillation loss
            for j in range(len(outputs)):
                if i != j:
                    loss_distillation += F.kl_div(F.log_softmax(outputs[i]),
                                    F.softmax(outputs[j]))
        #===================new added==========================
        # fuse_outputs_parallel
        reciprocal_sum = 0.0

        # 累加每个张量的倒数
        for ce in moe_ce:
            reciprocal_sum += 1 / (ce + 1e-8)  # 加平滑项避免除以零
        
        # 计算并联融合输出
        fused_output = 1/(reciprocal_sum) #
        
        loss_parallel = fused_output * len(outputs)
        #======================================================
        
        # print(loss, loss_distillation, loss_parallel)
        # loss =  loss_distillation + loss_parallel # + loss
    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output, loss_distillation, loss_parallel


def mix_outputs_teacher(outputs, labels, balance=False, label_dis=None):
    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0
    loss_distillation = 0.0
    loss_parallel = 0.0
    if balance == True:
        moe_ce = []
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)
    else:
        #===================new added==========================
        moe_ce = []
        #======================================================
        teacher_logits = smooth_label_logit_with_label_dis(
            outputs, labels)
        for i in range(len(outputs)):
            # base ce
            loss += soft_entropy(outputs[i], labels)
            #===================new added==========================
            moe_ce.append(soft_entropy(outputs[i], labels))
        # weights = compute_confusion_weights(outputs)
        # for i in range(len(outputs)):
        #     # base ce
        #     weighted_ce = (soft_entropy(outputs[i], labels,'none') * weights).sum() / weights.sum()
        #     loss += weighted_ce
        #     #===================new added==========================
        #     moe_ce.append(weighted_ce)
            #======================================================

            loss_distillation += F.kl_div(F.log_softmax(outputs[i]),
                                    F.softmax(teacher_logits))
         # #===================new added==========================
        # fuse_outputs_parallel
        reciprocal_sum = 0.0

        # 累加每个张量的倒数
        for ce in moe_ce:
            reciprocal_sum += 1 / (ce + 1e-8)  # 加平滑项避免除以零
        
        # 计算并联融合输出
        fused_output = 1/(reciprocal_sum) #
        
        loss_parallel = fused_output * len(outputs)
        #======================================================
        
        # print(loss, loss_distillation, loss_parallel)
        # loss =  loss_distillation + loss_parallel # + loss
    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output, loss_distillation, loss_parallel


# 这个其实还是shike的hnm，根本没有改
def smooth_label_logit_with_label_dis(outputs, labels, label_dis=None, temperature=1.0, min_weight=0.1, use_label_dis_weight=False):
    """
    Use expert outputs to construct a smoother logit for distillation, 
    with improved handling of non-target classes and label distribution (label_dis).
    
    Args:
        outputs (List[Tensor]): List of expert logits [B, C]
        labels (Tensor): One-hot ground-truth labels [B, C]
        label_dis (Tensor, optional): Distribution of class sample counts [C], shape [C]
        temperature (float): Temperature for softening logits
        min_weight (float): Minimum weight for non-target logits enhancement
        use_label_dis_weight (bool): Whether to use label distribution for weighting
        
    Returns:
        smooth_logits (Tensor): The smooth logits [B, C]
    """
    num_experts = len(outputs)
    B, C = outputs[0].shape

    # Step 1: Calculate the average logits across experts using stack
    logits_rank = torch.stack(outputs, dim=1)  # [B, M, C], where M is the number of experts
    avg_logits = torch.mean(logits_rank, dim=1)  # [B, C] (average across all experts)

    # Step 2: Calculate non-target logits (1 - labels)
    non_target_labels = 1 - labels
    non_target_logits = avg_logits * non_target_labels  # Weight non-target logits more gently

    if use_label_dis_weight and label_dis is not None:
        # Step 3: Adjust logits based on label distribution
        # We use label_dis to scale the logits based on class frequency
        # Smaller classes (tail) get higher weight
        class_weights = 1 / (label_dis + 1e-6)  # Avoid division by zero
        class_weights = class_weights / class_weights.sum()  # Normalize

        # Apply the class weights to non-target logits
        non_target_logits = non_target_logits * class_weights[None, :]  # [B, C], element-wise weight

    # Step 4: Rebuild the teacher logit by enhancing non-targets and combining
    max_teacher_logits, _ = torch.max(non_target_logits, dim=1)  # [B], hardest non-target logits

    # Expand max_teacher_logits to [B, 1] so it can be added to [B, C]
    max_teacher_logits = max_teacher_logits.unsqueeze(1)  # [B, 1]

    # Now we can safely combine the logits
    reformed_non_target_logits = max_teacher_logits * (1 - labels) +  labels * avg_logits # * 1.1  # [B, C]

    # Step 5: Apply temperature and softmax to obtain final teacher logits
    teacher_logits = F.softmax(reformed_non_target_logits / temperature, dim=1)  # [B, C]

    return teacher_logits


def compute_confusion_weights_jsd(outputs, labels, cls_freq, min_weight=0.3, eps=1e-8):
    """
    Compute sample weights using Jensen-Shannon Divergence + class frequency reweighting.

    Args:
        outputs (List[Tensor]): List of expert logits, each [B, C]
        labels (Tensor): Ground truth class indices, shape [B]
        cls_freq (Tensor): Class frequencies, shape [C]
        min_weight (float): Minimum weight
        eps (float): Small value to avoid division by zero

    Returns:
        weights (Tensor): [B], values in [min_weight, 1.0]
    """
    num_experts = len(outputs)
    B, C = outputs[0].shape
    device = outputs[0].device
    labels = labels.argmax(dim=1)

    # softmax 概率 [M, B, C]
    probs = [torch.softmax(o, dim=1) for o in outputs]
    probs = torch.stack(probs, dim=0)

    # 平均分布 [B, C]
    mean_prob = probs.mean(dim=0)

    # JSD 计算
    jsd = torch.zeros(B, device=device)
    for m in range(num_experts):
        kl = (probs[m] * (torch.log(probs[m] + eps) - torch.log(mean_prob + eps))).sum(dim=1)
        jsd += kl
    jsd = jsd / num_experts  # [B]

    # 归一化到 [0, 1]
    jsd_min, jsd_max = jsd.min(), jsd.max()
    norm_jsd = (jsd - jsd_min) / (jsd_max - jsd_min + eps)

    # 基础权重 [min_weight, 1]
    conf_weight = norm_jsd * (1 - min_weight) + min_weight

    # 类别频率修正
    # freq = cls_freq.to(device).float()  # [C]
    # freq_factor = 1.0 / torch.sqrt(freq[labels] + 1.0)  # [B]

    # # 融合
    # final_weight = conf_weight * freq_factor
    final_weight = conf_weight

    # 再次归一化到 [min_weight, 1]
    fw_min, fw_max = final_weight.min(), final_weight.max()
    final_weight = (final_weight - fw_min) / (fw_max - fw_min + eps)
    final_weight = final_weight * (1 - min_weight) + min_weight

    return final_weight


def mix_outputs_cw(outputs, labels, balance=False, label_dis=None):
    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    weights = compute_confusion_weights(outputs=outputs, labels=labels, cls_freq=label_dis)
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0
    loss_distillation = 0.0
    loss_parallel = 0.0
    if balance == True:
        moe_ce = []
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)
    else:
        #===================new added==========================
        moe_ce = []
        #======================================================
        
        teacher_logit = smooth_label_logit_with_label_dis(
            outputs, labels)
        for i in range(len(outputs)):
            # base ce
            weighted_ce = (soft_entropy(outputs[i], labels,'none') * weights).sum() / weights.sum()
            loss += weighted_ce
            #===================new added==========================
            moe_ce.append(weighted_ce)
            #======================================================
            # distillation loss
            # loss_distillation += F.kl_div(F.log_softmax(outputs[i]),
            #                         F.softmax(teacher_logit))
            for j in range(len(outputs)):
                if i != j:
                    loss_distillation += F.kl_div(F.log_softmax(outputs[i]),
                                    F.softmax(outputs[j]))
         # #===================new added==========================
        # fuse_outputs_parallel
        reciprocal_sum = 0.0

        # 累加每个张量的倒数
        for ce in moe_ce:
            reciprocal_sum += 1 / (ce + 1e-8)  # 加平滑项避免除以零
        
        # 计算并联融合输出
        fused_output = 1/(reciprocal_sum) #
        
        loss_parallel = fused_output * len(outputs)
        #======================================================
        
        # print(loss, loss_distillation, loss_parallel)
        # loss =  loss_distillation + loss_parallel # + loss
    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output, loss_distillation, loss_parallel

def best_teacher_student_weighted_kl(outputs, moe_ce, T=1.0):
    """
    Implements 'Teacher-from-best + Student-aware Weighting' distillation.

    Args:
        outputs (List[Tensor]): List of expert logits, each of shape [B, C]
        moe_ce (List[float]): Cross-entropy loss for each expert (length M)
        T (float): Temperature for distillation

    Returns:
        loss_distill (Tensor): Weighted KL-based distillation loss (scalar)
    """
    device = outputs[0].device
    num_experts = len(outputs)
    loss_distill = 0.0

    # Convert CE list to tensor
    moe_ce_tensor = torch.tensor(moe_ce, device=device)
    
    # Find teacher index
    teacher_idx = torch.argmin(moe_ce_tensor)

    # Normalize CE loss to [0, 1] for student weights
    min_ce = moe_ce_tensor.min()
    max_ce = moe_ce_tensor.max()
    norm_ce = (moe_ce_tensor - min_ce) / (max_ce - min_ce + 1e-6)  # [M]

    # Compute distillation loss
    for i in range(num_experts):
        if i == teacher_idx:
            continue
        weight = norm_ce[i]  # worse student -> higher weight

        kl = F.kl_div(
            F.log_softmax(outputs[i] / T, dim=1),
            F.softmax(outputs[teacher_idx].detach() / T, dim=1),
            reduction='batchmean'
        )
        loss_distill += weight * kl * (T ** 2)

    return loss_distill

def compute_confusion_consistency(outputs, threshold=1, mode='logits', reduction='mean'):
    """
    Compute consistency loss on samples where expert predictions disagree.

    Args:
        outputs (List[Tensor]): List of expert outputs (logits), each of shape [B, C]
        threshold (int): Minimum number of disagreements to consider a sample as 'confused'
        mode (str): 'logits' for logits-based consistency loss
        reduction (str): 'mean' or 'sum'

    Returns:
        loss_confusion (Tensor): Consistency loss on confused samples
        confusion_mask (Tensor): Boolean mask of confused samples, shape [B]
    """
    num_experts = len(outputs)
    device = outputs[0].device
    B, C = outputs[0].shape

    # Compute predictions from each expert
    preds = [o.argmax(dim=1) for o in outputs]  # List[Tensor], each [B]
    preds_stack = torch.stack(preds, dim=0)  # [M, B]

    # Compute mode (majority vote) along experts
    majority_vote = preds_stack.mode(dim=0).values  # [B]

    # Confusion: number of experts that disagree with majority
    disagreements = (preds_stack != majority_vote.unsqueeze(0)).sum(dim=0)  # [B]
    confusion_mask = disagreements > threshold  # [B], bool

    if confusion_mask.sum() == 0:
        return torch.tensor(0.0, device=device), confusion_mask  # No confused samples

    # Compute consistency loss only on confused samples
    loss_confusion = 0.0
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            x_i = outputs[i][confusion_mask]
            x_j = outputs[j][confusion_mask]
            if mode == 'logits':
                pairwise_loss = F.mse_loss(x_i, x_j, reduction=reduction)
            else:
                raise NotImplementedError(f"Mode '{mode}' not supported")
            loss_confusion += pairwise_loss

    return loss_confusion, confusion_mask

def compute_confusion_weights(outputs,labels=None,cls_freq=None, min_weight=0.3):
    """
    Compute smooth confusion-based sample weights using entropy of expert predictions.

    Args:
        outputs (List[Tensor]): List of expert logits, each of shape [B, C]
        min_weight (float): Minimum weight assigned to samples with no disagreement

    Returns:
        weights (Tensor): Tensor of shape [B], values in [min_weight, 1.0]
    """
    num_experts = len(outputs)
    B = outputs[0].shape[0]
    num_classes = outputs[0].shape[1]
    device = outputs[0].device
    labels = labels.argmax(dim=1) if labels is not None else None

    # Collect predictions from all experts: [M, B]
    preds = torch.stack([o.argmax(dim=1) for o in outputs], dim=0)  # [M, B]

    # Count votes for each class per sample: [B, C]
    vote_hist = torch.zeros((B, num_classes), device=outputs[0].device)
    for b in range(B):
        vote_hist[b].scatter_add_(0, preds[:, b], torch.ones(num_experts, device=outputs[0].device))

    # Convert to probabilities
    probs = vote_hist / vote_hist.sum(dim=1, keepdim=True)  # [B, C]

    # Entropy per sample
    entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1)  # [B]

    # Normalize entropy to [0, 1]
    max_entropy = torch.log(torch.tensor(num_classes, device=outputs[0].device))
    norm_entropy = entropy / max_entropy  # [B]

    # Map to [min_weight, 1.0]
    weights = norm_entropy * (1.0 - min_weight) + min_weight  # [B]


    # ---- 归一化 + min_weight 映射 ----
    w_min, w_max = weights.min(), weights.max()
    weights = (weights - w_min) / (w_max - w_min + 1e-6)
    weights = weights * (1.0 - min_weight) + min_weight


    return weights


def mix_outputs_weight(outputs, labels, balance=False, label_dis=None):
    label_dis_ori = label_dis
    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0
    loss_distillation = 0.0
    loss_parallel = 0.0
    if balance == True:
        moe_ce = []
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)

    else:
        moe_ce = []
        loss_parallel = 0.0
        loss_distillation = 0.0
        teacher_index = -1  # To store the index of the teacher (best expert)
        for i in range(len(outputs)):
            # Calculate base cross-entropy loss for each expert
            loss += soft_entropy(outputs[i], labels)
            #===================new added==========================
            # Append each expert's loss for further processing
            moe_ce.append(soft_entropy(outputs[i], labels))

        # Calculate the reciprocal sum to compute fused output
        reciprocal_sum = 0.0

        # Calculate reciprocal sum for each expert's loss
        for ce in moe_ce:
            reciprocal_sum += 1 / (ce + 1e-8)  # Add smoothing term to avoid division by zero

        # Calculate fused output using parallel loss
        fused_output = 1 / (reciprocal_sum)
        loss_parallel = fused_output * len(outputs)
        #======================================================

        class_counts = torch.tensor(label_dis_ori).to('cuda')
        class_weights = torch.log(1 + torch.sum(class_counts) / (class_counts + 1e-6))  # 防止除以0
        class_weights = class_weights / class_weights.sum()  # 归一化

        moe_ce = torch.tensor(moe_ce)
        loss_distillation = 0.0
        T = 2.0
        labels = labels.argmax(dim=1)  # 获取每个样本的类别索引
        for i in range(len(outputs)):
            for j in range(len(outputs)):
                if i == j:
                    continue
                # 获取当前 batch 每个样本的类别权重
                sample_weights = class_weights[labels]  # shape: [batch_size]

                # KL散度：逐样本计算（逐行 KL），乘以权重后再做均值
                kl = F.kl_div(
                    F.log_softmax(outputs[i] / T, dim=1),
                    F.softmax(outputs[j].detach() / T, dim=1),
                    reduction='none'  # [batch_size, num_classes]
                ).sum(dim=1)  # sum over classes → [batch_size]
 

                weighted_kl = (kl * sample_weights).mean()
                loss_distillation += weighted_kl * (T ** 2)
    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output, loss_distillation, loss_parallel

def moe_ce_loss(outputs, labels,label_dis=None):
    """
    计算交叉熵损失
    Args:
        outputs: 每个专家的输出 logits
        labels: 真实标签
    Returns:
        ce_loss: 交叉熵损失
    """
    loss = []
    if label_dis:
        label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
        label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
        for i in range(len(outputs)):
            loss.append(soft_entropy(outputs[i] + label_dis.log(), labels))
    else:
        for i in range(len(outputs)):
            loss.append(soft_entropy(outputs[i], labels))
    return loss


def distillation_loss(outputs,moe_ce):
    loss = 0.0
    for i in range(len(outputs)):
        for j in range(len(outputs)):
            if i != j:
                loss += F.kl_div(F.log_softmax(outputs[i]),
                                    F.softmax(outputs[j]))
    return loss

def distillation_loss_my(outputs, moe_ce):
    """
    计算知识蒸馏损失
    Args:
        outputs: 每个专家的输出 logits
        moe_ce: 存储每个专家交叉熵损失的列表
    Returns:
        distillation_loss: 知识蒸馏损失
    """
    distillation_loss = 0.0
    # 找到效果最好的专家
    best_expert_idx = torch.argmin(torch.stack(moe_ce))  # 最小的交叉熵损失对应的专家
    for i in range(len(outputs)):
        if i != best_expert_idx:  # 如果不是最好的专家，作为学生
            teacher_output = F.softmax(outputs[best_expert_idx], dim=1)  # 最好的专家输出
            student_output = F.softmax(outputs[i], dim=1)  # 当前专家的输出

            # 计算学生和教师之间的KL散度，作为蒸馏损失
            distillation_loss += F.kl_div(F.log_softmax(outputs[i], dim=1), teacher_output, reduction='batchmean')
    
    return distillation_loss

def parallel_loss(moe_ce):
    """
    计算并行损失，鼓励专家间的多样性
    Args:
        moe_ce: 存储每个专家交叉熵损失的列表
    Returns:
        parallel_loss: 并行损失
    """
    reciprocal_sum = 0.0
    
    # 累加每个张量的倒数
    for ce in moe_ce:
        reciprocal_sum += 1 / (ce + 1e-8)  # 加平滑项避免除以零
    
    # 计算并联融合输出
    fused_output = 1 / (reciprocal_sum) * len(moe_ce)
    return fused_output



def mixup_features(features, labels, alpha=0.2):
    """
    对同一类别的样本进行Mixup操作。将多个样本进行加权平均得到通用特征。
    
    :param features: 形状为(N, D)的张量，N是样本数，D是特征维度
    :param labels: 形状为(N,)的张量，包含每个样本的标签
    :param alpha: Mixup的超参数，控制加权的程度
    :return: 混合后的特征
    """
    # 获取所有唯一的标签
    unique_labels = torch.unique(labels)
    mixed_features = []

    # 对每个类别进行Mixup
    for label in unique_labels:
        # 获取当前类别的样本索引
        idx = torch.where(labels == label)[0]
        class_features = features[idx]

        # 对同类样本进行Mixup
        if len(class_features) > 1:  # 至少需要两个样本
            lambda_ = torch.distributions.Beta(alpha, alpha).sample()
            lambda_ = lambda_.to(features.device)

            # 对样本进行加权平均，得到通用特征
            mixup_feature = lambda_ * class_features[0] + (1 - lambda_) * class_features[1]
            mixed_features.append(mixup_feature)
        else:
            mixed_features.append(class_features[0])

    return torch.stack(mixed_features)


def hardest_negative(logits, labels):
    """
    选择logits中不是目标类别（标签）但预测值最大的类别作为最难负类。
    
    :param logits: 模型输出的logits，形状为(N, C)，N是样本数，C是类别数
    :param labels: 目标标签，形状为(N,)
    :return: 最难负类的类别索引
    """
    # 计算每个样本的预测类别（最大logit值对应的类别）
    max_logits, pred_classes = torch.max(logits, dim=1)

    # 创建一个mask来排除目标类别
    mask = pred_classes != labels

    # 对排除目标类别的logits进行过滤，选择最大值
    hardest_negatives = []
    for i in range(logits.size(0)):
        # 如果预测类别不是目标类别，获取最大的预测logits的类别
        if mask[i]:
            hardest_negatives.append(pred_classes[i])
        else:
            # 如果预测类别是目标类别，则跳过
            hardest_negatives.append(-1)  # 代表无效值，或者可以按需处理
    
    return torch.tensor(hardest_negatives).to(logits.device)


def contrastive_loss(logits, labels, temperature=0.07):
    """
    基于监督对比学习计算损失，采用logits和标签进行计算。
    
    :param logits: 模型输出的logits，形状为(N, C)，N是样本数，C是类别数
    :param labels: 目标标签，形状为(N,)
    :param temperature: 对比损失的温度参数
    :return: 对比损失
    """
    labels = labels.contiguous().view(-1, 1)  # 转换为列向量

    # 使用Softmax函数将logits转换为概率分布
    logits = logits / temperature  # 温度缩放
    logits = logits - logits.max(dim=1, keepdim=True)[0]  # 防止数值溢出
    probs = F.softmax(logits, dim=1)

    # 对比损失公式：负样本的概率
    loss = F.cross_entropy(logits, labels.view(-1), reduction='mean')
    
    return loss


class DiversityLoss(nn.Module):
    def __init__(self, num_experts, device='cuda'):
        """
        :param num_experts: 总共的专家数量
        :param device: 计算设备
        """
        super(DiversityLoss, self).__init__()
        self.num_experts = num_experts
        self.device = device

    def forward(self, expert_feature,label):
        """
        计算多个专家的原型之间的多样性损失
        :param expert_prototypes: 一个列表，每个元素包含一个专家的原型矩阵 (num_classes, feature_dim)
        :return: 多样性损失
        """
        diversity_loss = 0
        # 计算每对专家的原型之间的欧几里得距离
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                # 获取两个专家的原型
                proto_i = expert_feature[i]  # 形状: (num_classes, feature_dim)
                proto_j = expert_feature[j]  # 形状: (num_classes, feature_dim)

                # 计算余弦相似度矩阵
                # similarity_matrix  = F.cosine_similarity(proto_i.unsqueeze(1), proto_j.unsqueeze(0), dim=2)
                similarity_matrix  = F.kl_div(F.log_softmax(proto_i), F.softmax(proto_j), reduction='batchmean')

                 # 使用负的相似度作为损失，鼓励多样性，减小相似度
                diversity_loss +=  similarity_matrix.mean()

        diversity_loss /= (self.num_experts * (self.num_experts - 1)) / 2

        return -1/diversity_loss


def feature_separation_loss(features, lambda_fs=1e-2):
    """
    通过计算专家之间的特征分离来鼓励每个专家学习多样化的特征。
    - features: (batch_size, num_experts, feature_dim)
    - lambda_fs: 特征分离损失的权重超参数
    """
    features = torch.stack(features, dim=1)  # (batch_size, num_experts, feature_dim)
    batch_size, num_experts, feature_dim = features.shape
    
    # 计算专家之间的余弦相似度
    features = features.view(batch_size, num_experts, -1)  # Flatten features for pairwise similarity
    feature_norms = features.norm(p=2, dim=-1, keepdim=True)
    normalized_features = features / (feature_norms + 1e-8)  # Normalize features
    
    # 计算所有专家对之间的余弦相似度矩阵
    similarity_matrix = torch.bmm(normalized_features, normalized_features.transpose(1, 2))
    mask = torch.eye(similarity_matrix.shape[1], device=similarity_matrix.device)  # 创建 3x3 的单位矩阵
    similarity_matrix = similarity_matrix - mask.unsqueeze(0)  # 对角线置 0
    separation_loss = similarity_matrix.mean()  # 通过减少相似度来增加特征多样性
    
    return lambda_fs * separation_loss

def feature_diversity_loss(features, lambda_reg=0.01):
    """
    features: (batch_size, num_experts, feature_dim)
    lambda_reg: 正则化系数
    增加专家之间的特征多样性
    """
    features = torch.stack(features, dim=1)  # (batch_size, num_experts, feature_dim)
    batch_size, num_experts, feature_dim = features.shape
    
    # 计算不同专家之间的欧氏距离
    feature_pairwise_distances = torch.cdist(features.view(batch_size * num_experts, feature_dim),
                                             features.view(batch_size * num_experts, feature_dim))
    
    # 忽略对角线元素（同一专家的距离）
    mask = torch.eye(batch_size * num_experts).to(features.device)
    feature_pairwise_distances = feature_pairwise_distances.masked_select(mask == 0).view(batch_size * num_experts, -1)
    
    # 计算特征多样性损失（最大化专家间的距离）
    loss = feature_pairwise_distances.mean()
    
    return lambda_reg * loss

def attn_diversity_loss(attns, lambda_div=1):
    """
    attention_matrices: List of [1, 64, 8, 8] learned attention matrices
    lambda_div: 正则化权重
    """
    num_experts = len(attns)
    loss = 0.0

    # 计算两两专家的注意力矩阵差异
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            diff = attns[i] - attns[j]
            loss += torch.norm(diff, p=2)  # L2 距离
    
    # 归一化
    loss /= num_experts * (num_experts - 1) / 2
    return lambda_div * loss

class DiversityLossProto(nn.Module):
    def __init__(self, num_experts, num_classes=100, feature_dim=64, device='cuda'):
        """
        :param num_experts: 总共的专家数量
        :param num_classes: 类别数
        :param feature_dim: 特征维度
        :param device: 计算设备
        """
        super(DiversityLossProto, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device

    def forward(self, expert_features, labels):
        """
        计算多个专家之间的多样性损失
        :param expert_features: 一个列表，每个元素包含一个专家的特征矩阵 (batch_size, feature_dim)
        :param labels: 一个形状为 (batch_size, num_classes) 的 one-hot 编码标签
        :return: 多样性损失
        """
        # 初始化存储每个专家的原型
        prototypes = []
        
        for i in range(self.num_experts):
            expert_feature = expert_features[i]  # 形状: (batch_size, feature_dim)
            
            # 计算每个类别的原型，仅使用标签中为1的类别
            batch_size = expert_feature.size(0)
            class_prototype = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
            for c in range(self.num_classes):
                class_mask = labels[:, c] == 1  # 获取该类别的掩码
                if class_mask.sum() > 0:
                    # 对于该类别，计算特征均值
                    class_prototype[c] = expert_feature[class_mask].mean(dim=0)
            prototypes.append(class_prototype)

        # 将多个专家的原型堆叠成一个大的张量，形状: (num_experts, num_classes, feature_dim)
        prototypes = torch.stack(prototypes, dim=0)  # 形状: (num_experts, num_classes, feature_dim)

        # 计算原型之间的余弦相似度
        cos_sim = self.compute_cosine_similarity(prototypes)

        # 计算多样性损失（最大化原型之间的相似度）
        diversity_loss = cos_sim.mean()

        return diversity_loss

    def compute_cosine_similarity(self, prototypes):
        """
        计算原型之间的余弦相似度
        :param prototypes: (num_experts, num_classes, feature_dim)
        :return: (num_experts, num_experts)
        """
        # 计算两个原型之间的余弦相似度，首先将它们进行标准化
        normalized_prototypes = F.normalize(prototypes, p=2, dim=2)
        
        # 计算余弦相似度，结果是 (num_experts, num_classes, num_classes)
        cos_sim = torch.matmul(normalized_prototypes, normalized_prototypes.transpose(1, 2))  # (num_experts, num_classes, num_classes)
        
        # 取每个专家之间的平均余弦相似度
        cos_sim = cos_sim.mean(dim=1)  # (num_experts, num_experts)
        
        return cos_sim

    # def compute_prototypes(self, features, labels):
    #     """
    #     计算每个类别的原型（均值特征）
    #     :param features: [batch_size, feature_dim]
    #     :param labels: [batch_size, num_classes], one-hot 标签
    #     :return: [num_classes, feature_dim], 每个类别的原型
    #     """
    #     batch_size, feature_dim = features.size()
    #     num_classes = labels.size(1)

    #     # 使用 one-hot 标签来选择属于每个类别的特征
    #     class_prototypes = torch.zeros(num_classes, feature_dim, device=self.device)

    #     for i in range(num_classes):
    #         # 选择属于当前类别的特征
    #         class_mask = labels[:, i].unsqueeze(1)  # [batch_size, 1]
    #         class_features = features[class_mask.squeeze() == 1]  # [num_samples_for_class, feature_dim]
            
    #         if class_features.size(0) > 0:
    #             # 计算当前类别的原型（均值）
    #             class_prototypes[i] = class_features.mean(dim=0)

    #     return class_prototypes

    # def compute_diversity_loss(self, expert_prototypes):
    #     """
    #     计算多个专家之间的多样性损失
    #     :param expert_prototypes: [num_experts, num_classes, feature_dim], 每个专家的原型矩阵
    #     :return: 多样性损失
    #     """
    #     num_experts, num_classes, feature_dim = expert_prototypes.size()
        
    #     # 计算所有专家的原型之间的相似度（可以使用欧氏距离或余弦相似度）
    #     diversity_loss = 0
    #     for i in range(num_classes):
    #         class_proto = expert_prototypes[:, i, :]  # [num_experts, feature_dim]
    #         # 计算当前类别的专家之间的欧氏距离或余弦相似度
    #         similarity_matrix = F.cosine_similarity(class_proto.unsqueeze(1), class_proto.unsqueeze(0),dim=2)  # 欧氏距离
    #         similarity_matrix = (similarity_matrix.sum() - expert_prototypes.shape[0]) / 2 # 求和，作为多样性损失的一部分

    #         diversity_loss += similarity_matrix

    #     return diversity_loss

    # def forward(self, expert_features, labels):
    #     """
    #     计算多样性损失
    #     :param expert_features: 一个列表，每个元素包含一个专家的特征矩阵
    #     :param labels: 一个形状为 [batch_size, num_classes] 的 one-hot 标签
    #     :return: 多样性损失
    #     """
    #     # 计算每个专家的原型
    #     expert_prototypes = []
    #     for features in expert_features:
    #         class_prototypes = self.compute_prototypes(features, labels)
    #         expert_prototypes.append(class_prototypes)

    #     # 将所有专家的原型堆叠成一个张量 [num_experts, num_classes, feature_dim]
    #     expert_prototypes = torch.stack(expert_prototypes, dim=0)

    #     # 计算并返回多样性损失
    #     diversity_loss = self.compute_diversity_loss(expert_prototypes)
    #     return diversity_loss