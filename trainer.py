import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import progress_bar
from dataset import make_supervised_data_module
from resnet_stanford_cars_nabirds import resnet18, resnet50
from mobilenetv2_tiny_imagenet import mobilenetv2_T_w

def print_args(args):
    print('\n' + '='*30)
    print(f"{'Argument':<20} | {'Value'}")
    print('-'*30)
    for arg, value in vars(args).items():
        print(f"{arg:<20} | {value}")
    print('='*30 + '\n')

def __train_epoch(student_net, teacher_net, trainloader, device, criterion,
                  optimizer):
    student_net.train()
    teacher_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(trainloader):

        inputs = batch["images"].to(device)
        targets = batch["labels"].to(device)
        pseudo_targets = teacher_net(inputs)

        optimizer.zero_grad()
        outputs = student_net(inputs)
        loss = criterion(outputs, pseudo_targets, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx,
                     len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (train_loss / (batch_idx + 1), 100. * correct / total,
                      correct, total))


def __test_epoch(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 修改点：适配字典格式的数据读取 <---
        for batch_idx, batch in enumerate(testloader):
            inputs = batch["images"].to(device)
            targets = batch["labels"].to(device)

            outputs = net(inputs)  # 修改点：使用传入的参数 net <---
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(testloader), 'Acc: %.3f%% (%d/%d)' %
                         (100. * correct / total, correct, total))

    acc = 100. * correct / total
    return acc

def _make_criterion(alpha=0.9, T=4.0, mode='mse'):
    def criterion(outputs, targets, labels):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion


def _train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 准备数据 (调用 dataset.py 中的函数)
    print('==> Preparing data from Parquet files..')
    # 构造适配 dataset.make_supervised_data_module 的参数对象
    data_module = make_supervised_data_module(args)
    trainloader = data_module["train_dataloader"]
    testloader = data_module["test_dataloader"]

    save_dir = os.path.join('student_checkpoint', args.dataset_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 2. 教师模型 (ResNet50)
    print('==> Building Teacher: ResNet50..')
    teacher_net = resnet50(num_classes=args.num_classes)
    teacher_net = teacher_net.to(device)

    # 加载权重
    checkpoint_path = os.path.join('./checkpoint', f'{args.model_name}.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['net']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    teacher_net.load_state_dict(state_dict)

    teacher_acc = __test_epoch(teacher_net, testloader, device)
    print(f'>> Teacher Accuracy: {teacher_acc:.2f}%')

    # 3. 学生模型
    print('==> Building Student: mobilenetv2_T_w..')
    # student_net = resnet18(num_classes=args.num_classes)
    student_net = mobilenetv2_T_w(T=6, W=1.4, feature_dim=args.num_classes)
    student_net = student_net.to(device)

    criterion = _make_criterion(alpha=args.alpha, T=args.T, mode=args.kd_mode)
    optimizer = optim.SGD(student_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    milestones = [int(x) for x in args.scheduler_milestones.split()]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_acc = 0  # 初始化最高准确率  
    for epoch in range(1, args.n_epoch + 1):
        print(f'\nEpoch: {epoch} | LR: {optimizer.param_groups[0]["lr"]:.4f}')
        __train_epoch(student_net, teacher_net, trainloader, device, criterion, optimizer)
        acc = __test_epoch(student_net, testloader, device)

        if epoch % 20 == 0:
            print('\n' + '!' * 30)
            print(f'==> Periodic Checkpoint at Epoch {epoch}')
            print(f'>> Student Accuracy: {acc:.2f}%')
            teacher_acc = __test_epoch(teacher_net, testloader, device)
            print(f'>> Teacher Accuracy: {teacher_acc:.2f}%')
            print('!' * 30 + '\n')

        scheduler.step()

        if acc > best_acc:
            print(f'Saving best model... (Acc: {acc:.2f}%)')
            state = {
                'net': student_net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, f'./{save_dir}/mobilenetv2_T_w_student_batch_size_16_best.pth')
            best_acc = acc
        

def main():
    parser = argparse.ArgumentParser(description='ResNet50-to-mobilenetv2_T_w KD on stanford-cars')
    # 数据路径相关 (对接 dataset.py)
    parser.add_argument('--train_dir', default='/public/home/hechanghong2024/yjn/TAOKD/data/stanford-cars/train', type=str)
    parser.add_argument('--test_dir', default='/public/home/hechanghong2024/yjn/TAOKD/data/stanford-cars/test', type=str)
    parser.add_argument('--num_classes', default=196, type=int, help='Number of classes in dataset')
    parser.add_argument('--dataset_type', default='stanford-cars', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_coarse', default=False) 

    # 训练超参数
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--n_epoch', default=300, type=int)
    parser.add_argument('--scheduler_milestones', default="140 200 250", type=str)
    parser.add_argument('--model_name', default='resnet50_stanford_cars_batch-size_32_best', type=str)
    
    # 蒸馏超参数
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--T', default=1.0, type=float)
    parser.add_argument('--kd_mode', default='mse', choices=['cse', 'mse'])
    args = parser.parse_args()

    print_args(args)
    _train(args)

if __name__ == '__main__':
    main()
