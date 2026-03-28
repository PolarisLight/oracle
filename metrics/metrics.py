import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.dim()==2:
            target = target.argmax(dim=1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def dis_2_score(dis, return_numpy=True):
    """
    convert distance to score
    :param dis: distance
    :return: score
    """
    w = torch.linspace(1, 10, 10).to(dis.device)
    w_batch = w.repeat(dis.shape[0], 1)
    score = (dis * w_batch).sum(dim=1)
    if return_numpy:
        return score.cpu().numpy()
    else:
        return score

class ACC(torch.nn.Module):
    def __init__(self):
        super(ACC, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)

class ACCAVA(torch.nn.Module):
    def __init__(self):
        super(ACCAVA, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        # if score>5, then the prediction is correct, otherwise, it is wrong
        y_pred = np.where(y_pred>5, 1, 0)
        y_true = np.where(y_true>5, 1, 0)
        return accuracy_score(y_true, y_pred)
    
class Pearson(torch.nn.Module):
    def __init__(self):
        super(Pearson, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        return pearsonr(y_true, y_pred)[0]
    
class Spearman(torch.nn.Module):
    def __init__(self):
        super(Spearman, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        return spearmanr(y_true, y_pred)[0]
    

import torch

# 把compute_hmt_acc函数变成一个类，class freq作为初始化参数
class HMTAccuracy:
    def __init__(self, class_freq, head_thr=128, mid_thr=65):
        """
        计算 Head/Mid/Tail 的准确率 (macro average)

        Args:
            class_freq (list or Tensor): 每个类别的样本频率（来自训练集统计）
            head_thr (int): Head 阈值 (≥ head_thr)
            mid_thr (int): Mid 阈值 (mid_thr ≤ count < head_thr)
                           Tail 阈值 (count < mid_thr)
        """
        self.class_freq = torch.tensor(class_freq)
        self.head_thr = head_thr
        self.mid_thr = mid_thr

    def __call__(self, pred, target):
        return compute_hmt_acc(pred, target, self.class_freq, self.head_thr, self.mid_thr, num_classes=len(self.class_freq))

def compute_hmt_acc(pred, target, class_freq, head_thr=128, mid_thr=65, num_classes=None):
    """
    计算 Head/Mid/Tail 的准确率 (macro average)

    Args:
        pred (Tensor): 模型预测 logits 或类别索引, shape (N, C) 或 (N,)
        target (Tensor): 真实标签, shape (N,)
        class_freq (list or Tensor): 每个类别的样本频率（来自训练集统计）
        head_thr (int): Head 阈值 (≥ head_thr)
        mid_thr (int): Mid 阈值 (mid_thr ≤ count < head_thr)
                       Tail 阈值 (count < mid_thr)
        num_classes (int): 类别数，若 pred 是 logits 时必须提供

    Returns:
        dict: {"Head": acc, "Mid": acc, "Tail": acc}
    """
    if pred.dim() > 1:  # logits
        assert num_classes is not None, "Need num_classes when pred is logits"
        pred_labels = pred.argmax(dim=1)
    else:  # already indices
        pred_labels = pred

    if target.dim() == 2:  # one-hot
        target = target.argmax(dim=1)

    # 转为 CPU，方便统计
    pred_labels = pred_labels.cpu()
    target = target.cpu()
    class_freq = torch.tensor(class_freq)

    # 每类正确数/总数
    correct = torch.zeros(len(class_freq))
    total = torch.zeros(len(class_freq))

    for t, p in zip(target, pred_labels):
        total[t] += 1
        if p == t:
            correct[t] += 1

    class_acc = torch.zeros(len(class_freq))
    mask = total > 0
    class_acc[mask] = correct[mask] / total[mask]

    # 分段索引
    head_idx = (class_freq >= head_thr).nonzero(as_tuple=True)[0]
    mid_idx = ((class_freq >= mid_thr) & (class_freq < head_thr)).nonzero(as_tuple=True)[0]
    tail_idx = (class_freq < mid_thr).nonzero(as_tuple=True)[0]

    res = {}
    if len(head_idx) > 0:
        res["Head"] = class_acc[head_idx].mean().item()
    if len(mid_idx) > 0:
        res["Mid"] = class_acc[mid_idx].mean().item()
    if len(tail_idx) > 0:
        res["Tail"] = class_acc[tail_idx].mean().item()

    return res

if __name__ == "__main__":
    # ==== 测试 ====
    logits = torch.tensor([[2.0, 1.0, 0.5],
                           [0.5, 2.0, 1.0],
                           [1.0, 0.5, 2.0],
                           [2.0, 1.0, 0.5],
                           [0.5, 2.0, 1.0],
                           [1.0, 0.5, 2.0]])
    target = torch.tensor([0, 1, 2, 0, 1, 1])
    class_freq = [3, 2, 1]  # 类别频率

    res = compute_hmt_acc(logits, target, class_freq, head_thr=3, mid_thr=2, num_classes=3)
    print(res)  # {'Head': ..., 'Mid': ..., 'Tail': ...}
    # ==== 示例 ====
    logits = torch.randn(100, 10)      # 模型输出
    targets = torch.randint(0, 10, (100,))
    class_freq = [200,180,150,90,70,50,40,30,20,15]  # 训练集频率

    res = compute_hmt_acc(logits, targets, class_freq, head_thr=128, mid_thr=65, num_classes=10)
    print(res)  # {'Head': ..., 'Mid': ..., 'Tail': ...}
