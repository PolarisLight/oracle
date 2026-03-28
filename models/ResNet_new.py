import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.metrics import accuracy
from models.Resnet import BasicBlock


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _labels_to_index(label: torch.Tensor) -> torch.Tensor:
    if label.ndim == 2:
        return label.argmax(dim=1).long()
    return label.long()


class ResNet_AmbiProto(nn.Module):
    """
    Multi-prototype ambiguity-aware classifier for oracle bone characters.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = args.train.device
        self.num_classes = args.model.num_classes

        block = BasicBlock
        num_blocks = [5, 5, 5]
        self.in_planes = 16
        self.next_in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.in_planes = 32
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.feature_dim = 64

        self.proto_k = getattr(args.model, "proto_k", 4)
        self.tau = getattr(args.model, "tau", 0.10)
        self.beta = getattr(args.model, "ambi_beta", 5.0)
        self.m0 = getattr(args.model, "ambi_m0", 0.0)
        self.kappa = getattr(args.model, "entropy_floor", min(math.log(self.num_classes) * 0.55, 3.2))
        self.lambda_g = getattr(args.model, "lambda_g", 0.5)
        self.lambda_t = getattr(args.model, "lambda_t", 0.3)
        self.lambda_p = getattr(args.model, "lambda_p", 0.05)
        self.tol_eps = getattr(args.model, "tol_eps", 0.08)
        self.tol_m = getattr(args.model, "tol_M", 5)
        self.conf_ema = getattr(args.model, "conf_ema", 0.95)
        self.proto_margin = getattr(args.model, "proto_margin", 0.20)
        self.aa_warmup = getattr(args.model, "aa_warmup", 20)
        self.aa_ramp = max(1, getattr(args.model, "aa_ramp", 40))
        self.tol_warmup = getattr(args.model, "tol_warmup", 30)
        self.conf_warmup = getattr(args.model, "conf_warmup", 20)

        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.proto_k, self.feature_dim) * 0.02
        )
        self.ce_loss = nn.CrossEntropyLoss()

        self.register_buffer("conf_mat", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("conf_ready", torch.zeros(1, dtype=torch.bool))

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride_item in strides:
            layers.append(block(self.next_in_planes, planes, stride_item))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = F.avg_pool2d(out, out.size(3)).view(out.size(0), -1)
        return feat

    def forward_logits(self, feat: torch.Tensor) -> torch.Tensor:
        feat_norm = F.normalize(feat, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=2)
        sim = torch.einsum("bd,ckd->bck", feat_norm, proto_norm) / max(self.tau, 1e-6)
        return torch.logsumexp(sim, dim=2)

    def ambiguity_score(self, logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
        return torch.sigmoid(self.beta * (self.m0 - gap))

    def prototype_diversity_loss(self) -> torch.Tensor:
        if self.proto_k <= 1:
            return self.prototypes.new_zeros(())
        proto_norm = F.normalize(self.prototypes, dim=2)
        sim = torch.einsum("ckd,cld->ckl", proto_norm, proto_norm)
        eye = torch.eye(self.proto_k, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        penalty = F.relu(sim - self.proto_margin) * (1.0 - eye)
        denom = float(self.num_classes * self.proto_k * (self.proto_k - 1))
        return penalty.sum() / max(denom, 1.0)

    @torch.no_grad()
    def update_confusion_ema(self, target: torch.Tensor, pred: torch.Tensor) -> None:
        batch_conf = torch.zeros_like(self.conf_mat)
        ones = torch.ones_like(target, dtype=batch_conf.dtype)
        batch_conf.index_put_((target, pred), ones, accumulate=True)
        row_sum = batch_conf.sum(dim=1, keepdim=True).clamp_min(1.0)
        batch_conf = batch_conf / row_sum
        if not bool(self.conf_ready.item()):
            self.conf_mat.copy_(batch_conf)
            self.conf_ready.fill_(True)
            return
        self.conf_mat.mul_(self.conf_ema).add_(batch_conf * (1.0 - self.conf_ema))

    @torch.no_grad()
    def get_accept_set(self, target: torch.Tensor) -> torch.Tensor:
        if not bool(self.conf_ready.item()) or self.tol_m <= 0 or self.num_classes <= 1:
            return torch.empty(target.size(0), 0, device=target.device, dtype=torch.long)
        top_m = min(self.tol_m, self.num_classes - 1)
        conf_rows = self.conf_mat[target].to(target.device)
        conf_rows = conf_rows.clone()
        conf_rows.scatter_(1, target.unsqueeze(1), float("-inf"))
        return torch.topk(conf_rows, k=top_m, dim=1).indices

    def build_tolerant_targets(self, target: torch.Tensor, accept_set: torch.Tensor) -> torch.Tensor:
        batch_size = target.size(0)
        q = torch.zeros(batch_size, self.num_classes, device=target.device, dtype=torch.float32)
        q.scatter_(1, target.unsqueeze(1), 1.0 - self.tol_eps)
        if accept_set.numel() > 0:
            neighbor_mass = self.tol_eps / accept_set.size(1)
            q.scatter_add_(
                1,
                accept_set,
                torch.full(
                    (batch_size, accept_set.size(1)),
                    neighbor_mass,
                    device=target.device,
                    dtype=q.dtype,
                ),
            )
        return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feat = self.forward_features(x)
        logits = self.forward_logits(feat)
        prob = F.softmax(logits, dim=1).clamp_min(1e-12)
        entropy = -(prob * prob.log()).sum(dim=1)
        ambiguity = self.ambiguity_score(logits)
        self.feature = feat
        aux = {
            "feat": feat,
            "prob": prob,
            "entropy": entropy,
            "ambiguity": ambiguity,
        }
        return logits, aux

    def curriculum_scale(self, epoch: int) -> float:
        if epoch < self.aa_warmup:
            return 0.0
        return min(1.0, float(epoch - self.aa_warmup + 1) / float(self.aa_ramp))

    def train_step(self, data, rt=None, epoch: int = 0):
        x = data["image"].to(self.device)
        y = _labels_to_index(data["label"].to(self.device))

        logits, aux = self(x)
        ambiguity = aux["ambiguity"]
        entropy = aux["entropy"]
        schedule = self.curriculum_scale(epoch)
        effective_ambiguity = ambiguity * schedule

        with torch.no_grad():
            if epoch >= self.conf_warmup:
                accept_set = self.get_accept_set(y)
            else:
                accept_set = torch.empty(y.size(0), 0, device=y.device, dtype=torch.long)

        log_prob = F.log_softmax(logits, dim=1)
        hard_ce = -log_prob.gather(1, y.unsqueeze(1)).squeeze(1)
        guard = F.relu(self.kappa - entropy).pow(2)
        loss_aa = (
            (1.0 - effective_ambiguity) * hard_ce
            + effective_ambiguity * (self.lambda_g * schedule) * guard
        ).mean()

        if accept_set.numel() > 0 and epoch >= self.tol_warmup:
            tolerant_targets = self.build_tolerant_targets(y, accept_set)
            tolerant_ce = -(tolerant_targets * log_prob).sum(dim=1)
            loss_t = (effective_ambiguity * self.lambda_t * tolerant_ce).mean()
        else:
            loss_t = logits.new_zeros(())

        loss_proto = self.prototype_diversity_loss()
        loss = loss_aa + loss_t + self.lambda_p * loss_proto

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            self.update_confusion_ema(y, pred)

        return {
            "loss": loss,
            "loss_AA": loss_aa.detach(),
            "loss_tol": loss_t.detach(),
            "loss_proto": loss_proto.detach(),
            "ambiguity_mean": ambiguity.mean().detach(),
            "schedule": torch.tensor(schedule, device=logits.device),
            "entropy_mean": entropy.mean().detach(),
        }

    @torch.no_grad()
    def eval_step(self, data, rt=None):
        x = data["image"].to(self.device)
        y = _labels_to_index(data["label"].to(self.device))
        logits, aux = self(x)
        loss = self.ce_loss(logits, y)
        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        return {
            "pred": logits,
            "loss": loss,
            "acc1": acc1,
            "acc5": acc5,
            "feature": self.feature,
            "label": y,
            "ambiguity": aux["ambiguity"],
            "entropy": aux["entropy"],
        }

    def on_eval_end(self):
        return


ResNet_ALG_AmbiLatent = ResNet_AmbiProto
