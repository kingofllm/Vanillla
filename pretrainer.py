import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset_process import make_supervised_data_module
from mobilenetv2_tiny_imagenet import mobilenetv2_T_w
from resnet_stanford_cars_nabirds import resnet18, resnet50


def print_args(args):
    print('\n' + '='*30)
    print(f"{'Argument':<20} | {'Value'}")
    print('-'*30)
    for arg, value in vars(args).items():
        print(f"{arg:<20} | {value}")
    print('='*30 + '\n')

def __train_epoch(net, trainloader, device, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(trainloader):
        if batch is None: continue
        
        # 从字典中提取数据
        inputs = batch["images"].to(device)

        targets = batch["labels"].to(device)
        # targets = batch["fine_idx"].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0: # 减少打印频率
            print(f'Batch: {batch_idx} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')


def __test_epoch(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            if batch is None: continue
            
            inputs = batch["images"].to(device)
            # targets = batch["fine_idx"].to(device)
            targets = batch["labels"].to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'Test Acc: {acc:.3f}%')
    return acc
    
def _pretrain(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0.0

    # 1. 准备数据 (使用 dataset.py 中的工厂函数)
    print(f'==> Preparing {args.dataset_type} data from {args.data_dir}..')
    
    # 构造 dataset.py 需要的路径
    args.train_dir = os.path.join(args.data_dir, "train")
    args.test_dir = os.path.join(args.data_dir, "test")
    args.use_coarse = False # Stanford Cars 不需要此参数，但为了兼容性保留

    data_module = make_supervised_data_module(args)
    trainloader = data_module["train_dataloader"]
    testloader = data_module["test_dataloader"]

    # 2. 构建模型
    if args.dataset_type == "stanford-cars":
        num_classes =  196
    elif args.dataset_type  == "nabirds":
        num_classes =  555
    elif args.dataset_type == "cifar":
        num_classes = 100
    elif args.dataset_type == "tiny-imagenet":
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    
    print(f'==> Building mobilenetv2_T_w for {args.dataset_type} ({num_classes} classes)..')
    
    # net = resnet18(num_classes=num_classes)

    net = mobilenetv2_T_w(T=6, W=1.4, feature_dim=num_classes)

    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, num_classes)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 3. 优化器与调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    milestones = [int(x) for x in args.scheduler_milestones.split()]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # 4. 训练循环
    for epoch_idx in range(1, args.n_epoch + 1):
        print(f'\nEpoch: {epoch_idx} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        __train_epoch(net, trainloader, device, criterion, optimizer)
        acc = __test_epoch(net, testloader, device, criterion)
        
        scheduler.step()

        if acc > best_acc:
            print('Saving Best Model..')
            state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch_idx}
            if not os.path.isdir('checkpoint'): os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/{args.model_name}_best.pth')
            best_acc = acc

def main():
    parser = argparse.ArgumentParser(description='PyTorch nabirds Training')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--n_epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_dir', default='/public/home/hechanghong2024/yjn/TAOKD/data/nabirds', type=str)
    parser.add_argument('--dataset_type', default='nabirds', type=str,
                        choices=['stanford-cars', 'tiny-imagenet', 'nabirds', 'cifar'])
    parser.add_argument('--scheduler_milestones', default="140 200 250", type=str)
    parser.add_argument('--model_name', default='mobilenetv2_T_w_nabirds', type=str)
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 5e-4)')
    args = parser.parse_args()
    print_args(args)

    # 设置随机种子
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    _pretrain(args)

if __name__ == '__main__':
    main()