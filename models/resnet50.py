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

# ------------------------
# 替换 backbone block
# ------------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ------------------------
# ResNet50-MoE
# ------------------------
@core_module
class ResNet_MoE(nn.Module):
    def __init__(self, args)->None:
        super(ResNet_MoE, self).__init__()
        self.args = args
        block = Bottleneck
        num_blocks = [3, 4, 6, 3]   # ResNet50 配置
        num_experts = args.model.num_experts if hasattr(args.model,'num_experts') else 3
        num_classes = args.model.num_classes
        use_norm = True
        self.s = 30
        self.device = args.train.device

        self.num_experts = num_experts
        self.in_planes = 64
        self.next_in_planes = 64

        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # shared layers
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)   # 输出 256
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)   # 输出 512
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)   # 输出 1024

        # experts: only layer4
        self.layer4s = nn.ModuleList()
        for _ in range(self.num_experts):
            self.in_planes = 1024   # 共享 layer3 的输出
            l4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 输出 2048
            self.layer4s.append(l4)
        feat_dim = 512 * block.expansion  # 2048
        if use_norm:
            self.classifiers = nn.ModuleList([NormedLinear(feat_dim, num_classes) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList([NormedLinear(feat_dim, num_classes) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList([nn.Linear(feat_dim, num_classes) for _ in range(self.num_experts)])

        self.apply(_weights_init)

        # 保留你原来的 HNM / LA Loss 配置
        self.expert_hnm = [
            HardNegativeMining(num_classes=num_classes, feature_dim=feat_dim,
                               k=args.core_params.hnm_k, momentum=args.core_params.hnm_momentum,
                               temperature=args.core_params.hnm_temperature)
            for _ in range(num_experts)
        ]
        from loss.la_loss import LogitAdjustmentLoss
        self.la_loss = LogitAdjustmentLoss(class_freq=args.label_dis)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, crt=False):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        outs, feats = [], []
        for i in range(self.num_experts):
            o = self.layer4s[i](out)
            o = F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1)
            feats.append(o)
            if crt:
                outs.append(self.s * self.rt_classifiers[i](o))
            else:
                outs.append(self.s * self.classifiers[i](o))

        self.feature = feats
        return outs
    
    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}
    
    
    def train_step(self, data:dict[str,torch.Tensor],rt:bool=False,epoch:int=0)->dict[str,torch.Tensor]:
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x,crt=rt)
        features_list = self.feature
        exp_loss = []
        for i in range(len(pred)):
            # base ce
            exp_loss.append(soft_entropy(pred[i], y))

        loss,_ ,loss_distillation,loss_parallel= mix_outputs(pred,y,balance=rt,label_dis=self.args.label_dis)


        
        if epoch < self.args.core_params.conf_start_epoch:
            loss,_ ,loss_distillation,loss_parallel= mix_outputs(pred,y,balance=rt,label_dis=self.args.label_dis)
        else:
            loss,_ ,loss_distillation,loss_parallel= mix_outputs_cw(pred,y,balance=rt,label_dis=self.args.label_dis)
       
        
        loss_diversity = 0.0
        if not rt:
            loss = self.args.core_params.distill_weight * loss_distillation \
                    + self.args.core_params.parallel_weight * loss_parallel \
        #             + (1-self.args.core_params.parallel_weight) * loss
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

                


        return {'loss':loss,'exp_loss':exp_loss}
    
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
    

