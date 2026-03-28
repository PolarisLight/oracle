import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models.zero_resnet import Backbone  # 假设你使用的是 ResNet18 模型，修改成你自己的模型
import tqdm

# 定义损失函数：交叉熵损失 + Supervised Contrastive Loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1)
        
    def forward(self, features, labels):
        labels = labels.view(-1, 1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(features.device)
        contrastive_loss = self.compute_loss(features, mask)
        return contrastive_loss

    def compute_loss(self, features, mask):
        similarity_matrix = self.similarity(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        logits_max = torch.max(similarity_matrix, dim=-1, keepdim=True)[0]
        logits = similarity_matrix - logits_max.detach()
        logits = logits * mask
        loss = torch.sum(torch.exp(logits)) / torch.sum(mask)
        return loss


# EWC 正则化类
class EWC:
    def __init__(self, model, dataloader, lamda=1000):
        self.model = model
        self.lamda = lamda
        self.fisher_information = self.compute_fisher_information(dataloader)
        self.old_params = self.get_params()

    # 计算Fisher信息矩阵，只考虑fcrt层
    def compute_fisher_information(self, dataloader):
        fisher_information = {}
        self.model.train()  # 确保模型在训练模式
        for name, param in self.model.named_parameters():
            # 只初始化fcrt层的Fisher信息矩阵
            if 'fcrt' in name:
                fisher_information[name] = torch.zeros_like(param)

        # 遍历数据集进行反向传播，计算Fisher信息
        for datas in dataloader:
            data, labels = datas['image'], datas['label']
            self.model.zero_grad()  # 清除之前的梯度
            data, labels = data.cuda(), labels.cuda()
            outputs, fcrt_out = self.model(data)
            loss = nn.CrossEntropyLoss()(fcrt_out, labels)
            loss.backward()

            # 计算每个fcrt参数的Fisher信息
            for name, param in self.model.named_parameters():
                if 'fcrt' in name and param.grad is not None:  # 只计算fcrt层
                    fisher_information[name] += param.grad ** 2 / len(dataloader)

        return fisher_information

    # 获取模型参数
    def get_params(self):
        return {name: param.clone() for name, param in self.model.named_parameters() if 'fc' in name}

    # 计算EWC损失
    def ewc_loss(self):
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:  # 仅对fcrt层应用EWC
                loss += (self.fisher_information[name] * (param - self.old_params[name]) ** 2).sum()
        return self.lamda * loss




# 自定义模型，包含两个fc层：一个用于阶段1，另一个用于阶段2
class MyModel(nn.Module):
    def __init__(self, num_classes=100):
        super(MyModel, self).__init__()
        self.encoder = Backbone()
        
        # 阶段1的fc层
        self.fc = nn.Linear(256, num_classes)  # CIFAR-100有100个类
        
        # 阶段2的fcrt层
        self.fcrt = nn.Linear(256, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        fc_out = self.fc(features)
        fcrt_out = self.fcrt(features)
        return fc_out, fcrt_out


# 获取数据加载器
def get_data_loader(dataset, batch_size=128, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# 测试模型
def test(model, test_loader,post_traing=False):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在测试时不计算梯度
        for data, labels in test_loader:
            data,labels = data.to('cuda'),labels.to('cuda')
            if post_traing:
                _, outputs= model(data)
            else:
                outputs, _ = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# 训练模型
def train(model, dataloader, criterion, optimizer, scheduler=None, num_epochs=10, test_loader=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in tqdm.tqdm(dataloader):
            x,y = data['image'].to('cuda'),data['label'].to('cuda')
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # 更新学习率
           
            total_loss += loss.item()
        

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
        
        # 计算测试集准确率
        if test_loader is not None:
            accuracy = test(model, test_loader)
            print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%")
    if scheduler:
        scheduler.step()



# 训练第二阶段，使用不同的优化器和学习率调度器
def train_stage_two(model, dataset, indices, criterion, optimizer_fcrt, scheduler_fcrt, ewc, num_epochs=10, test_loader=None):
    # 冻结所有 encoder 的层，只重新训练 fc 和 fcrt
    for param in model.parameters():
        param.requires_grad = False  # 冻结encoder部分
    
    # 选择部分数据集
    subset = Subset(dataset, indices)
    dataloader = get_data_loader(subset)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss_fcrt = 0

        for data in tqdm.tqdm(dataloader):
            optimizer_fcrt.zero_grad()
            x,y = data['image'].to('cuda'),data['label'].to('cuda')
            # 前向传递
            _, fcrt_out = model(x)

            # 计算fcrt损失
            loss_fcrt = criterion(fcrt_out, y)

            # 加入EWC正则化损失
            ewc_loss = ewc.ewc_loss()  # 计算EWC正则化损失
            total_loss = loss_fcrt + ewc_loss  # 总损失

            # 反向传播：计算总损失的梯度
            total_loss.backward()

            # 更新fc和fcrt层的参数
            optimizer_fcrt.step()



            total_loss_fcrt += loss_fcrt.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss fcrt: {total_loss_fcrt / len(dataloader)}")

        # 计算测试集准确率
        if test_loader is not None:
            accuracy = test(model, test_loader, post_traing=True)
            print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%")
                    # 更新学习率
    scheduler_fcrt.step()

def split_data_into_head_mid_tail(dataset, num_head=40, num_mid=30, num_tail=30):
    # 固定划分：头类、中类、尾类
    head_classes = list(range(num_head))  # 0-39类为头类
    mid_classes = list(range(num_head, num_head + num_mid))  # 40-69类为中类
    tail_classes = list(range(num_head + num_mid, 100))  # 70-99类为尾类
    
    # 按类别划分数据集
    head_indices = [i for i, data in enumerate(dataset) if torch.argmax(torch.Tensor(data['label'])).item() in head_classes]
    mid_indices = [i for i, data in enumerate(dataset) if torch.argmax(torch.Tensor(data['label'])).item() in mid_classes]
    tail_indices = [i for i, data in enumerate(dataset) if torch.argmax(torch.Tensor(data['label'])).item() in tail_classes]
    
    return head_indices, mid_indices, tail_indices


# 定义训练和测试过程的主函数
def main():
    # 1. 加载数据集
    from datasets.CIFAR_LT import IMBALANCECIFAR100
    from config.Arguments import Arguments
    args = Arguments()
    args.dataset = Arguments()
    args.dataset.name = 'CIFAR100'
    args.dataset.label_dir = 'H:\DatasetD\cifar'
    args.dataset.data_dir = 'H:\DatasetD\cifar'
    args.dataset.loader = None
    args.dataset.imgsz = 32
    args.dataset.imb_factor = 0.01
    dataset = IMBALANCECIFAR100(args=args,train=True)
    
    test_dataset = torchvision.datasets.CIFAR100(root=args.dataset.data_dir, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    test_loader = get_data_loader(test_dataset)

    # 2. 模型和损失函数
    model = MyModel(num_classes=100).to('cuda')  # CIFAR-100有100个类
    criterion_ce = nn.CrossEntropyLoss()

    # 3. 优化器和Cosine学习率调度器
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 4. 第一阶段训练
    print("Training on CIFAR-100-LT...")
    train_loader = get_data_loader(dataset)
    train(model, train_loader, criterion_ce, optimizer, scheduler, num_epochs=10, test_loader=test_loader)
    model.fcrt.weight.data.copy_(model.fc.weight.data)  # 将fc层的权重复制到fcrt层
    model.fcrt.bias.data.copy_(model.fc.bias.data)  # 将fc层的偏置复制到fcrt层

    head_indices, mid_indices, tail_indices = split_data_into_head_mid_tail(dataset,34,33,33)

    # 6. 创建EWC正则化实例
    ewc = EWC(model, get_data_loader(dataset),lamda=100)
    from loss.moe_loss import LALoss
    la = LALoss(dataset.get_cls_num_list())
    # 7. 第二阶段训练：冻结encoder，只训练fc和fcrt层
    print("Training head classes...")
    optimizer_fcrt = optim.SGD(model.fcrt.parameters(), lr=0.1, momentum=0.9)
    scheduler_fcrt = optim.lr_scheduler.CosineAnnealingLR(optimizer_fcrt, T_max=30)
    train_stage_two(model, dataset, head_indices, criterion=la.call, optimizer_fcrt=optimizer_fcrt, scheduler_fcrt=scheduler_fcrt, ewc=ewc, num_epochs=10, test_loader=test_loader)

    print("Training mid classes...")
    train_stage_two(model, dataset, mid_indices, criterion=la.call, optimizer_fcrt=optimizer_fcrt, scheduler_fcrt=scheduler_fcrt, ewc=ewc, num_epochs=10, test_loader=test_loader)

    print("Training tail classes with regularization to prevent forgetting head classes...")
    train_stage_two(model, dataset, tail_indices, criterion=la.call, optimizer_fcrt=optimizer_fcrt, scheduler_fcrt=scheduler_fcrt, ewc=ewc, num_epochs=10, test_loader=test_loader)


if __name__ == "__main__":
    main()
