import torch.nn as nn
import torch
import torch.nn.init as init
from metrics.metrics import accuracy
from loss.HNM import HardNegativeMining_Proto_Enhanced
from loss.moe_loss import soft_entropy
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


import torch.nn.functional as F

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Backbone(nn.Module):

    def __init__(self, block=BasicBlock_s, num_blocks=[5,5,5], nf=64):
        super(Backbone, self).__init__()
        self.in_planes = nf

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.out_dim = 4 * nf * block.expansion

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        self.feature = feature
        return feature



class ResNet_modify(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100, nf=64):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.out_dim = 4 * nf * block.expansion

        self.fc = nn.Linear(self.out_dim, num_classes)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        hidden_dim = 128
        self.rt_classifiers = nn.Linear(self.out_dim, num_classes)
        # self.contrast_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        # self.projection_head = nn.Sequential(
        #     nn.Linear(self.out_dim, hidden_dim),
        # )
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        self.feature = feature
        if train is True:
            out = self.fc(feature)
            return out # z, p
        else:
            out = self.rt_classifiers(feature)

            return out
        
    def classify(self,feature):
        out = self.fc(feature)
        return out


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_class=100):
        super().__init__()
        self.in_channels = 64
        self.num_class = num_class

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # from torch.nn import w

        self.fc = nn.Linear(512 * block.expansion, num_class)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        hidden_dim=256
        self.fc_cb = nn.Linear(512 * block.expansion, num_class)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(512 * block.expansion, hidden_dim),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        feature = output.view(output.size(0), -1)
        self.feature = feature
        if train is True:
            out = self.fc(feature)
            out_cb = self.fc_cb(feature)
            z = self.projection_head(feature)
            p = self.contrast_head(z)
            return out
        else:
            out = self.fc_cb(feature)
            return out

from utils.utils import core_module

@core_module
class ResNet_GLMC(nn.Module):

    def __init__(self,args)->None:
        super().__init__()
        self.model = ResNet_modify(BasicBlock_s, [5, 5, 5], num_classes=args.model.num_classes)
        self.device = args.train.device

    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}
    
    def forward(self, x,train):
        return self.model(x, train=train)
    
    def train_step(self, data:dict[str,torch.Tensor],rt,epoch):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x,train=rt)
        loss = F.cross_entropy(pred,y)


        return {'loss':loss}

    def eval_step(self, data:dict[str,torch.Tensor],rt)->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        x = data['image'].to(self.device)
        y = data['label'].to(self.device)
        pred = self(x, train=rt)
        output = pred
        loss = F.cross_entropy(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        
        return {"pred": output, "loss": loss, "acc1": acc1, "acc5": acc5,'pred':pred,'label':y}
    

    def on_eval_end(self,):
        pass

class IncrementalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=100):
        super(IncrementalClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 初始化分类头权重
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_dim) * 0.01)

    def forward(self, x):
        # 始终输出 num_classes 个类别的 logits
        return torch.matmul(x, self.weights.T)  # [batch_size, num_classes]

    def get_weight_matrix(self):
        return self.weights
    
def zero_space_projection(W_base: torch.Tensor, W_rt: torch.Tensor):
    """
    将 W_rt 投影到 W_base 的零空间
    """
    W_base = W_base.float()
    W_rt = W_rt.float()
    d = W_base.shape[1]

    # 构造零空间投影矩阵 P
    K = W_base.T
    KTK_inv = torch.linalg.pinv(K @ K.T)
    P = torch.eye(d, device=W_base.device) - K @ K.T @ KTK_inv
    return (P @ W_rt.T).T  # 保留 [C_t, d] 结构

@core_module
class ResNet_zero(nn.Module):

    def __init__(self,args)->None:
        super().__init__()
        self.args = args
        self.backbone = Backbone()
        self.classifier = IncrementalClassifier(self.backbone.out_dim, num_classes=args.model.num_classes)

        self.device = args.train.device

    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}
    
    def forward(self, x,train):
        return self.model(x, train=train)
    
    def train_step(self, data:dict[str,torch.Tensor],rt,epoch):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x,train=rt)
        feature = self.model.feature
        if rt:
            # use reweight loss
            label_dis = torch.tensor(
                self.args.label_dis, dtype=torch.float, requires_grad=False).cuda()
            loss_ce = soft_entropy(pred + label_dis.log(), y)
            # ========rt时的基于原型的增强========
            # aug_feature = self.proto.prototype_interpolate(feature, y)
            # aug_pred = self.model.classify(aug_feature)
            # loss_ce += soft_entropy(aug_pred + label_dis.log(), y)
            loss = loss_ce

        else:
            #============特征提取器训练阶段的原型学习与对比学习============
            self.proto.update_prototypes(feature, y)
            self.proto.update_confusion_matrix(pred, y)
            if hasattr(self.args.core_params,'hnm_start_epoch') and epoch>=self.args.core_params.hnm_start_epoch:
                adjusted_pred = self.proto.correct_logits(pred,feature, y)
                # use normal loss
                loss_ce = F.cross_entropy(adjusted_pred, y)
            else:
                loss_ce = F.cross_entropy(pred, y)
            loss = loss_ce
            
            if hasattr(self.args.core_params,'hnm_start_epoch') and epoch>=self.args.core_params.hnm_start_epoch:
                loss_pcl = self.proto.compute_dynamic_proto_contrastive_loss(feature, y,
                                                                             epoch=epoch,
                                                                             warmup=self.args.core_params.hnm_start_epoch)
                loss += self.args.core_params.pcl_weight * loss_pcl
            


        return {'loss':loss}

    def eval_step(self, data:dict[str,torch.Tensor],rt)->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        x = data['image'].to(self.device)
        y = data['label'].to(self.device)
        pred = self(x, train=rt)
        output = pred
        loss = F.cross_entropy(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        
        return {"pred": output, "loss": loss, "acc1": acc1, "acc5": acc5,'pred':pred,'label':y}
    

    def on_eval_end(self,):
        self.proto.apply_epoch_momentum(self.args.core_params.momentum)

def resnet18(num_class=100):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class=num_class)

def resnet32(num_class=10):
    return ResNet_modify(BasicBlock_s, [5, 5, 5], num_classes=num_class)

def resnet34(num_class=100):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class)


def resnet50(num_class=100):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_class=num_class)


def resnet101(num_class=100):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_class=num_class)


def resnet152(num_class=100):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_class=num_class)
