'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
This MoE design is based on the implementation of Yerlan Idelbayev.
'''

from collections import OrderedDict
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from loss.moe_loss import *
from metrics.metrics import accuracy
from utils.utils import core_module,compute_class_accuracies
from loss.HNM import HardNegativeMining_Proto as HardNegativeMining


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, 
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class StridedConv(nn.Module):
    """
    downsampling conv layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out


class SkipPooling(nn.Module):
    """
    shallow features alignment wrt. depth
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(SkipPooling, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(depth)]))

    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

@core_module
class ResNet_MoE(nn.Module):

    def __init__(self, args)->None:
        """
        """
        super(ResNet_MoE, self).__init__()
        self.args = args
        block = BasicBlock
        num_blocks =  [5, 5, 5]
        num_experts = args.model.num_experts if hasattr(args.model,'num_experts') else 3
        num_classes = args.model.num_classes
        use_norm=True
        self.s = 30
        self.device = args.train.device
        # self.label_dis = args.label_dis #TODO:open this line

        self.num_experts = num_experts
        self.in_planes = 16
        self.next_in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        if num_experts:
            layer3_output_dim = 64
            self.in_planes = 32
            self.layer3s = nn.ModuleList([self._make_layer(
                block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(self.num_experts)])
            self.in_planes = self.next_in_planes
            if use_norm:
                # self.s = 30
                self.classifiers = nn.ModuleList(
                    [NormedLinear(64, num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(64, num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList(
                    [nn.Linear(64, num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = NormedLinear(64, num_classes) if use_norm else nn.Linear(
                64, num_classes, bias=True)

        self.apply(_weights_init)
        self.depth = list(
            reversed([i + 1 for i in range(len(num_blocks) - 1)]))  # [2, 1]
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(
            self.num_experts)]  # [2, 1, 2]
        feat_dim = 16
        self.shallow_exps = nn.ModuleList([SkipPooling(
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])
        

        self.expert_hnm = [HardNegativeMining(num_classes=num_classes,feature_dim=64,k=args.core_params.hnm_k,momentum=args.core_params.hnm_momentum,temperature=args.core_params.hnm_temperature) for _ in range(num_experts)]

        self.diversity_loss = feature_diversity_loss
        from loss.la_loss import LogitAdjustmentLoss
        self.la_loss = LogitAdjustmentLoss(class_freq=args.label_dis)
    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, crt=False):

        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]

            exp_outs = [out3 * exp(shallow_outs[i % len(shallow_outs)]) for i, (out3, exp) in enumerate(zip(out3s, self.shallow_exps))]

            # exp_outs = out3s

            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]
            self.feature = exp_outs
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            outs = self.linear(out)

        return outs

    
    def train_step(self, data:dict[str,torch.Tensor],rt:bool=False,epoch:int=0)->dict[str,torch.Tensor]:
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x,crt=rt)
        features_list = self.feature

        loss,_= mix_outputs_old(pred,y,balance=rt,label_dis=self.args.label_dis)
        # ===================================
        # if epoch < self.args.core_params.conf_start_epoch:
        #     loss,_ ,loss_distillation,loss_parallel= mix_outputs(pred,y,balance=rt,label_dis=self.args.label_dis)
        # else:
        #     loss,_ ,loss_distillation,loss_parallel= mix_outputs_cw(pred,y,balance=rt,label_dis=self.args.label_dis)
        # exp_loss = []
        # for i in range(len(pred)):
        #     # base ce
        #     exp_loss.append(soft_entropy(pred[i], y))

        
        # loss_diversity = 0.0
        # if not rt:
        #     loss = self.args.core_params.distill_weight * loss_distillation \
        #             + self.args.core_params.parallel_weight * loss_parallel \
        # =======================================
                    # + (1-self.args.core_params.parallel_weight) * loss
        # #     if epoch <= self.args.core_params.diversity_epoch:
        #         loss_diversity = self.diversity_loss(features_list,self.args.core_params.diversity_weight)
        #         loss += loss_diversity
            # if hasattr(self.args.core_params,'hnm_start_epoch') and epoch>=self.args.core_params.hnm_start_epoch:
            #     expert_losses = []
            #     for i, expert_features in enumerate(features_list):
            #         hnm = self.expert_hnm[i] 

            #         predicted =  torch.argmax(pred[i], dim=1)

            #         hnm.update_prototypes(expert_features, y)
            #         hnm.update_confusion_matrix(predicted, y)
            #         contrastive_loss = hnm.compute_contrastive_loss(expert_features, y)

            #         expert_losses.append(contrastive_loss)
                
            #     loss = sum(expert_losses) * self.args.core_params.hnm_weight + (1 - self.args.core_params.hnm_weight) * loss 

                


        return {'loss':loss}
    
    def eval_step(self, data:dict[str,torch.Tensor],rt:bool=False)->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        x = data['image'].to(self.device)
        y = data['label'].to(self.device)
        pred = self(x,crt=rt)
        output = sum(pred)/len(pred)
        loss = F.cross_entropy(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        exp_acc = []
        for exp_pred in pred:
            exp_acc.append(accuracy(exp_pred, y, topk=(1, 5))[0].item())
        exp_acc = torch.tensor(exp_acc)
        return {"pred": output, "loss": loss, "acc1": acc1, "acc5": acc5,'exp_acc':exp_acc,'pred':pred,'label':y,'feature':self.feature}
    def on_eval_end(self,):
        for hnm in self.expert_hnm:
            hnm.apply_epoch_momentum()
    
    def extract_features(self,x):
        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]

            exp_outs = [out3s[i] * shallow_expe_outs[i]
                        for i in range(self.num_experts)]


            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]
            return exp_outs
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            return out
        
    def classify(self,feature,crt=True):
        if self.num_experts:
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]
                        (feature[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (feature[i]) for i in range(self.num_experts)]
        else:
            outs = self.linear(feature)
        
        return outs
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.la_loss import LogitAdjustmentLoss

class ResNet_MoE_Gated(nn.Module):
    """
    双专家 MoE (CE + LA)，带标量 gating 融合
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        block = BasicBlock
        num_blocks = [5, 5, 5]
        num_classes = args.model.num_classes
        self.s = 30
        self.device = args.train.device

        # --------- Backbone ---------
        self.in_planes = 16
        self.next_in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.in_planes = 32
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # --------- Experts ---------
        self.classifier_ce = NormedLinear(64, num_classes)
        self.classifier_la = NormedLinear(64, num_classes)

        # --------- Gating ---------
        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),     # scalar gate
            # nn.Sigmoid()          # α ∈ [0,1]
        )

        # --------- Loss ---------
        self.ce_loss = nn.CrossEntropyLoss()
        self.la_loss = LogitAdjustmentLoss(class_freq=args.label_dis)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = F.avg_pool2d(out, out.size()[3]).view(out.size(0), -1)  # [B,64]
        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        self.feature = feat
        # expert outputs
        logit_ce = self.classifier_ce(feat) * self.s
        logit_la = self.classifier_la(feat) * self.s

        # gating
        alpha_logit = self.gate(feat)            # [B,1]
        alpha = torch.sigmoid(alpha_logit)       # [B,1], α ∈ [0,1]
        logit_mix = alpha * logit_ce + (1 - alpha) * logit_la
        # logit_mix = (logit_ce + logit_la)/2  # 简单平均,消融
        return logit_mix, logit_ce, logit_la, alpha_logit

    def train_step(self, data,rt=None, epoch=0):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logit_mix, logit_ce, logit_la, alpha_logit = self(x)

        # 主 loss：混合输出的 CE
        loss_main = self.ce_loss(logit_mix, y)

        # 专家辅助 loss
        # loss_ce = self.ce_loss(logit_ce, y)
        # loss_la = self.la_loss(logit_la, y, 0.5 + epoch / self.args.train.epochs)
        loss_ce = soft_entropy(logit_ce, y)
        label_dis = torch.tensor(
            self.args.label_dis, dtype=torch.float, requires_grad=False).cuda()
        label_dis = label_dis.unsqueeze(0).expand(y.shape[0], -1)
        loss_pla = soft_entropy(logit_la+(0.5 + epoch / self.args.train.epochs)*label_dis.log()
                               , y)
        
        loss_la = soft_entropy(logit_la+label_dis.log()
                               , y)
        loss_lbl = gate_loss_lbl(alpha_logit, y, self.args.label_dis)
        
        # 总 loss
        loss = loss_main + 0.5 * (loss_ce + loss_pla) + 1 * loss_lbl
        # loss = loss_ce
        alpha = torch.sigmoid(alpha_logit)
        return {
            "loss": loss,
            "loss_main": loss_main.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_la": loss_pla.detach(),
            "alpha_mean": alpha.mean().item()
        }

    def eval_step(self, data,rt=None):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logit_mix, _, _, alpha_logit = self(x)
        alpha = torch.sigmoid(alpha_logit)
        loss = self.ce_loss(logit_mix, y)
        acc1, acc5 = accuracy(logit_mix, y, topk=(1, 5))
        return {"pred": logit_mix, "loss": loss, "acc1": acc1, "acc5": acc5,
                'feature':self.feature,'label':y,'alpha':alpha.detach()}

    def on_eval_end(self):
        pass


# @core_module
class ResNet_GatingMoE(nn.Module):
    def __init__(self, args)->None:
        super(ResNet_GatingMoE, self).__init__()
        self.args = args
        block = BasicBlock
        num_blocks = [5, 5, 5]
        self.num_experts = getattr(args.model, 'num_experts', 3)
        self.num_classes = 100 if args.dataset.name == 'IMBALANCECIFAR100' else 10
        use_norm = True
        self.s = 30
        self.device = args.train.device

        # ---- stem & shared ----
        self.in_planes = 16
        self.next_in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        # ---- experts ----
        if self.num_experts:
            layer3_output_dim = 64
            self.in_planes = 32
            self.layer3s = nn.ModuleList([
                self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2)
                for _ in range(self.num_experts)
            ])
            self.in_planes = self.next_in_planes
            if use_norm:
                self.classifiers = nn.ModuleList([NormedLinear(64, self.num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList([NormedLinear(64, self.num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(64, self.num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = NormedLinear(64, self.num_classes) if use_norm else nn.Linear(64, self.num_classes, bias=True)

        # ---- gating（轻量）----
        gating_in_dim = 32 * getattr(block, "expansion", 1)
        self.gate = nn.Linear(gating_in_dim, self.num_experts)

        cp = getattr(args, "core_params", args)
        self.gate_top_m      = getattr(cp, "gate_top_m", 2)
        self.gate_q_temp     = getattr(cp, "gate_q_temp", 1.0)     # q 的温度（非门控温度）
        self.gate_kl_weight  = getattr(cp, "gate_kl_weight", 0.5)  # KL(q||p) 权重
        self.gate_ce_weight  = getattr(cp, "gate_ce_weight", 0.3)  # 用 p 混合的 CE 权重
        self.gate_balance_w  = getattr(cp, "gate_balance_weight", 1e-3)  # 使用均衡正则
        self.eps = 1e-8

        self.apply(_weights_init)

    # ----------------- helpers -----------------
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for s in strides:
            layers.append(block(self.next_in_planes, planes, s))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def _topk_mask(scores: torch.Tensor, m: int) -> torch.Tensor:
        B, K = scores.shape
        m = min(m, K)
        if m >= K:
            return torch.ones_like(scores, dtype=torch.float32)
        idx = scores.topk(k=m, dim=1).indices
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(1, idx, 1.0)
        return mask

    def _masked_log_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        very_neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(mask < 1, very_neg)
        return F.log_softmax(masked, dim=1)

    def _is_one_hot(self, y: torch.Tensor) -> bool:
        return (y.dim()==2) and (y.size(1)==self.num_classes) and (y.dtype.is_floating_point)

    def _to_index(self, y: torch.Tensor) -> torch.Tensor:
        """one-hot -> index；index 直接返回"""
        if self._is_one_hot(y):
            return y.argmax(dim=1)
        return y.long()

    def _to_one_hot(self, y: torch.Tensor) -> torch.Tensor:
        """index -> one-hot；one-hot 直接返回"""
        if self._is_one_hot(y):
            return y
        idx = y.long()
        oh = torch.zeros(idx.size(0), self.num_classes, device=idx.device, dtype=torch.float32)
        oh.scatter_(1, idx.view(-1,1), 1.0)
        return oh

    def _soft_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        支持软/硬标签：
        - target float [B,C] -> soft CE
        - target long  [B]   -> 普通 CE
        """
        if target.dtype.is_floating_point:
            logp = F.log_softmax(logits, dim=-1)
            return -(target * logp).sum(dim=1).mean()
        else:
            return F.cross_entropy(logits, target)

    # ----------------- forward -----------------
    def forward(self, x, crt:bool=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)

        # gate 特征（layer2 的 GAP）
        self.gate_feat = F.adaptive_avg_pool2d(out2, 1).view(out2.size(0), -1)  # [B, 32*exp]

        if self.num_experts:
            out3s = [self.layer3s[i](out2) for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(o3, o3.size()[3]).view(o3.size(0), -1) for o3 in out3s]  # [B,64] * K
            self.feature = exp_outs
            if crt:
                outs = [self.s * self.rt_classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            outs = self.linear(out)
        return outs  # list[K]*[B,C]（有专家）或 [B,C]（无专家）

    # ----------------- gating targets & probs -----------------
    def _gate_targets_and_probs(self, outs_tensor: torch.Tensor, y: torch.Tensor):
        """
        由专家 logits 构造 q（教师）；由 gate logits 构造 p（学生）。
        y 支持 one-hot 或 index。
        返回：q(detached)、logq(detached)、p、logp、mask
        """
        B, K, C = outs_tensor.shape
        y_idx = self._to_index(y)               # [B]
        y_oh  = self._to_one_hot(y)             # [B,C]

        # 专家效用 u：真类 log-prob；兼容 one-hot
        logp_exp = F.log_softmax(outs_tensor, dim=-1)                 # [B,K,C]
        # u = sum_c onehot[c]*logp_exp[..., c]
        u = (logp_exp * y_oh.unsqueeze(1)).sum(dim=2)                 # [B,K]

        # 组内标准化优势 a（样本内、跨专家）
        mu = u.mean(1, keepdim=True)
        sigma = torch.clamp(u.std(1, keepdim=True), min=1e-3)         # 稳定
        a = (u - mu) / sigma                                          # [B,K]

        # Top-m 掩码：用优势选择支持集，q/p 共用
        mask = self._topk_mask(a, self.gate_top_m)

        # 教师 q：softmax(a/T) on mask
        logq = self._masked_log_softmax(a / self.gate_q_temp, mask)   # [B,K]
        q = logq.exp().detach()                                       # 作为教师，detach

        # 学生 p：gate logits on same mask
        gate_logits = self.gate(self.gate_feat)                       # [B,K]
        logp = self._masked_log_softmax(gate_logits, mask)            # [B,K]
        p = logp.exp()

        return q, logq.detach(), p, logp, mask, y_idx, y_oh

    # ----------------- train / eval -----------------
    def train_step(self, data:dict[str,torch.Tensor], rt:bool=False, epoch:int=0)->dict[str,torch.Tensor]:
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x, crt=rt)                              # list[K]*[B,C] 或 [B,C]

        # —— 主 CE：使用 gating 的混合（让 gate 真正“用起来”）——
        if isinstance(pred, list):
            outs_tensor = torch.stack(pred, dim=1)          # [B,K,C]
            gate_logits = self.gate(self.gate_feat)
            mask_gate = self._topk_mask(gate_logits, self.gate_top_m)
            logp_gate = self._masked_log_softmax(gate_logits, mask_gate)
            p_gate = logp_gate.exp()                        # [B,K]
            mix_logits = (outs_tensor * p_gate.unsqueeze(-1)).sum(1)  # [B,C]
        else:
            mix_logits = pred                               # K=1 时

        # y 兼容 one-hot & index
        ce_main = self._soft_ce(mix_logits, y)
        
        for i, p in enumerate(pred):
            if isinstance(p, torch.Tensor):
                ce_main += self._soft_ce(p, y) * self.gate_ce_weight
            else:
                ce_main += sum(self._soft_ce(p_, y) for p_ in p) / len(p)

        loss = ce_main

        # —— KL(q||p)：把 gate 拉向“组内相对优势”的教师分布 q —— 
        if self.num_experts and self.gate_kl_weight > 0:
            outs_tensor = torch.stack(pred, dim=1) if isinstance(pred, list) else pred.unsqueeze(1)
            q, logq, p, logp, mask, y_idx, y_oh = self._gate_targets_and_probs(outs_tensor, y)
            kl = (q * (logq - logp)).sum(1).mean()         # forward-KL on mask
            loss = loss + self.gate_kl_weight * kl

        # —— 使用均衡正则：防止单专家长期独占（很小系数）——
        if self.num_experts and self.gate_ce_weight > 0 and self.gate_balance_w > 0:
            p_bar = p_gate.mean(0)                         # [K]
            balance = ((p_bar - 1.0/self.num_experts) ** 2).sum()
            loss = loss + self.gate_balance_w * balance

        return {'loss': loss}

    @torch.no_grad()
    def eval_step(self, data:dict[str,torch.Tensor], rt:bool=False)->dict[str,torch.Tensor]:
        x = data['image'].to(self.device)
        y = data['label'].to(self.device)
        pred = self(x, crt=rt)
        y_idx = self._to_index(y)

        if isinstance(pred, list):
            outs_tensor = torch.stack(pred, dim=1)         # [B,K,C]
            gate_logits = self.gate(self.gate_feat)
            mask_gate = self._topk_mask(gate_logits, self.gate_top_m)
            logp_gate = self._masked_log_softmax(gate_logits, mask_gate)
            p_gate = logp_gate.exp()
            output = (outs_tensor * p_gate.unsqueeze(-1)).sum(1)      # gate 混合作为主输出
            avg_output = outs_tensor.mean(1)                          # 对照：等权平均
        else:
            output = pred
            avg_output = pred

        loss = F.cross_entropy(output, y_idx)                          # 评估用硬标签
        acc1, acc5 = accuracy(output, y_idx, topk=(1, 5))

        # 专家各自 acc（观测多样性）
        if isinstance(pred, list):
            exp_acc = torch.tensor([accuracy(p, y_idx, topk=(1,5))[0].item() for p in pred])
            avg_acc1, _ = accuracy(avg_output, y_idx, topk=(1,5))
        else:
            exp_acc = torch.tensor([])
            avg_acc1 = acc1

        return {"pred": output, "loss": loss, "acc1": acc1, "acc5": acc5,
                "avg_acc1": avg_acc1, "exp_acc": exp_acc, "pred_list": pred,
                "label": y, "feature": getattr(self, "feature", None)}

    # 兼容原接口
    def on_eval_end(self): 
        return

    def extract_features(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        if self.num_experts:
            out3s = [self.layer3s[i](out2) for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(o3, o3.size()[3]).view(o3.size(0), -1) for o3 in out3s]
            return exp_outs
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            return out
        
    def classify(self,feature,crt=True):
        if self.num_experts:
            if crt: outs = [self.s * self.rt_classifiers[i](feature[i]) for i in range(self.num_experts)]
            else:   outs = [self.s * self.classifiers[i](feature[i])     for i in range(self.num_experts)]
        else:
            outs = self.linear(feature)
        return outs


@core_module
class ResNet_SSM(nn.Module):

    def __init__(self, args)->None:
        """
        """
        super(ResNet_SSM, self).__init__()
        self.args = args
        block = BasicBlock
        num_blocks =  [5, 5, 5]
        num_experts = args.model.num_experts if hasattr(args.model,'num_experts') else 1
        num_classes = 100 if args.dataset.name == 'IMBALANCECIFAR100' else 10
        use_norm=True
        self.s = 30
        self.device = args.train.device
        # self.label_dis = args.label_dis #TODO:open this line

        self.num_experts = num_experts
        self.in_planes = 16
        self.next_in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3_output_dim = 64
        self.in_planes = 32
        self.layer3 = self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) 
        self.in_planes = self.next_in_planes

        self.classifiers = NormedLinear(64, num_classes) 
        self.rt_classifiers = NormedLinear(64, num_classes)

        self.apply(_weights_init)

        from models.ssm import SSMLogitAdjustment
        self.ssm = SSMLogitAdjustment(num_classes=num_classes)

    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, label=None,crt=False, ssm_weight=1.0):

        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        
        out3 = self.layer3(out2)
        out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
        self.feature = out
        if crt:
            outs = self.rt_classifiers(out)*self.s
        else:
            outs = self.classifiers(out)*self.s

        outs = self.ssm(outs, label, ssm_weight)

        return outs

    
    def train_step(self, data:dict[str,torch.Tensor],rt:bool=False,epoch:int=0)->dict[str,torch.Tensor]:
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        # ssm_weight 随着 epoch增加而增加
        ssm_weight = max(0, float(epoch / self.args.train.epochs))
        pred = self(x, label=y, crt=rt, ssm_weight=ssm_weight)
        # if epoch >= self.args.core_params.ssm_start_epoch:
        #     pred = self(x, label=y, crt=rt, use_ssm=True)
        # else:
        #     pred = self(x, crt=rt)

        if not rt:
            loss_ce = soft_entropy(pred, y)
            loss = loss_ce #+ 2*(self.ssm.A.norm(2)+self.ssm.B.norm(2))
        else:
            label_dis = torch.tensor(self.args.label_dis,dtype=torch.float, requires_grad=False,device=self.device)
            loss_ad = soft_entropy(pred+label_dis.log(), y)
            loss = loss_ad

        return {'loss':loss}
    
    def eval_step(self, data:dict[str,torch.Tensor],rt:bool=False)->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        x = data['image'].to(self.device)
        y = data['label'].to(self.device)
        pred = self(x,crt=rt)
        loss = F.cross_entropy(pred, y)
        acc1, acc5 = accuracy(pred, y, topk=(1, 5))
        return {"pred": pred, "loss": loss, "acc1": acc1, "acc5": acc5,'pred':pred,'label':y,'feature':self.feature}
    def on_eval_end(self,test_logits, test_targets):
        """
        Called at the end of evaluation
        test_logits: logits from the model
        test_targets: true labels
        """
        acc_per_class = compute_class_accuracies(test_logits, test_targets)
        self.ssm.update_accuracies(acc_per_class)
    
    def extract_features(self,x):
        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]

            exp_outs = [out3s[i] * shallow_expe_outs[i]
                        for i in range(self.num_experts)]


            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]
            return exp_outs
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            return out
        
    def classify(self,feature,crt=True):
        if self.num_experts:
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]
                        (feature[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (feature[i]) for i in range(self.num_experts)]
        else:
            outs = self.linear(feature)
        
        return outs


@core_module
class ResNet_SSMloss(nn.Module):

    def __init__(self, args)->None:
        """
        """
        super(ResNet_SSMloss, self).__init__()
        self.args = args
        block = BasicBlock
        num_blocks =  [5, 5, 5]
        num_experts = args.model.num_experts if hasattr(args.model,'num_experts') else 1
        num_classes = 100 if args.dataset.name == 'IMBALANCECIFAR100' else 10
        use_norm=True
        self.s = 30
        self.device = args.train.device
        # self.label_dis = args.label_dis #TODO:open this line

        self.num_experts = num_experts
        self.in_planes = 16
        self.next_in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        layer3_output_dim = 64
        self.in_planes = 32
        self.layer3 = self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) 
        self.in_planes = self.next_in_planes

        self.classifiers = NormedLinear(64, num_classes) 
        self.rt_classifiers = NormedLinear(64, num_classes)

        self.apply(_weights_init)

        from loss.SSM_loss import MambaSSMLoss, RebalancedCELoss, FocalLoss
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = MambaSSMLoss(num_classes=args.model.num_classes,
        #                          alpha=args.core_params.mamba_alpha,
        #                          label_dis=args.label_dis)
        # self.loss = RebalancedCELoss(label_distribution=torch.tensor(args.label_dis, dtype=torch.float32, device=self.device))
        # self.loss = FocalLoss(gamma=2.0, alpha=None, reduction='mean')
        from loss.grpo_loss import DynamicClassWeightLoss, GroupRelativeLossMem,GIBALoss,GIBALoss_best
        # self.loss = DynamicClassWeightLoss(num_classes=args.model.num_classes,
        #                                    alpha=args.core_params.mamba_alpha,)
        self.giba_loss = GIBALoss(class_freq=torch.tensor(args.label_dis, dtype=torch.float32, device=self.device),
                             tail_classes= [i for i in range(66,100)],)
        self.loss = nn.CrossEntropyLoss()

    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, crt=False):

        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        
        out3 = self.layer3(out2)
        out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
        self.feature = out
        if crt:
            outs = self.rt_classifiers(out) *self.s
        else:
            outs = self.classifiers(out) *self.s

        # outs = self.ssm(outs, label, ssm_weight)

        return outs

    
    def train_step(self, data:dict[str,torch.Tensor],rt:bool=False,epoch:int=0)->dict[str,torch.Tensor]:
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        # ssm_weight 随着 epoch增加而增加
        pred = self(x, crt=rt)
        # if epoch >= self.args.core_params.ssm_start_epoch:
        #     pred = self(x, label=y, crt=rt, use_ssm=True)
        # else:
        #     pred = self(x, crt=rt)

        if not rt:
            loss_ce = self.loss(pred, y)
            loss_giba = self.giba_loss(pred, self.feature,self.classifiers,y,epoch)
            mix = epoch/self.args.train.epochs
            loss = loss_ce*(1-mix)+loss_giba*mix
            # loss = loss_giba
        else:
            label_dis = torch.tensor(self.args.label_dis,dtype=torch.float, requires_grad=False,device=self.device)
            loss_ad = soft_entropy(pred+label_dis.log(), y)
            loss = loss_ad

        return {'loss':loss}
    
    def eval_step(self, data:dict[str,torch.Tensor],rt:bool=False)->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        x = data['image'].to(self.device)
        y = data['label'].to(self.device)
        pred = self(x,crt=rt)
        loss = F.cross_entropy(pred, y)
        acc1, acc5 = accuracy(pred, y, topk=(1, 5))
        return {"pred": pred, "loss": loss, "acc1": acc1, "acc5": acc5,'pred':pred,'label':y,'feature':self.feature}
    def on_eval_end(self,):
        """
        Called at the end of evaluation
        test_logits: logits from the model
        test_targets: true labels
        """
        # acc_per_class = compute_class_accuracies(kargs['test_logits'],kargs['test_targets'])
        pass
    
    def extract_features(self,x):
        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]

            exp_outs = [out3s[i] * shallow_expe_outs[i]
                        for i in range(self.num_experts)]


            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]
            return exp_outs
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            return out
        
    def classify(self,feature,crt=True):
        if self.num_experts:
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]
                        (feature[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (feature[i]) for i in range(self.num_experts)]
        else:
            outs = self.linear(feature)
        
        return outs

# for Cifar100-LT use
def resnet32(num_classes=100, use_norm=False, num_exps=None):
    return ResNet_MoE(BasicBlock, [5, 5, 5], num_experts=num_exps, num_classes=num_classes, use_norm=use_norm)
