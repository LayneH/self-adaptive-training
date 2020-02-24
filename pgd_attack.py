from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models import get_model
from utils import accuracy, AverageMeter


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn34',
                    help='model architecture')
parser.add_argument('--test-batch-size', type=int, default=200,
                    help='input batch size for testing (default: 200)')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, type=float,
                    help='perturb step size')
parser.add_argument('--random', default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-dir', default='./checkpoints', type=str,
                    help='directory of model checkpoints for white-box attack evaluation')

args = parser.parse_args()

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='~/datasets/CIFAR10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)


def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    out = model(X)
    acc = accuracy(out.data, y)[0].item()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda(non_blocking=True)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    acc_pgd = accuracy(model(X_pgd).data, y)[0].item()
    return acc, acc_pgd


def eval_adv_test_whitebox(model, test_loader):
    robust_accs = AverageMeter()
    natural_accs = AverageMeter()
    model.eval()

    for data, target in test_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        X, y = Variable(data, requires_grad=True), Variable(target)
        acc, acc_pgd = _pgd_whitebox(model, X, y)
        natural_accs.update(acc, data.shape[0])
        robust_accs.update(acc_pgd, data.shape[0])
        print('natural acc: {:.2f}, robust acc: {:.2f}'.format(natural_accs.avg, robust_accs.avg))
    
    return natural_accs.avg, robust_accs.avg


def main():
    eval_epochs = range(71, 101)
    model = get_model(args, 10)
    model = nn.DataParallel(model).cuda()
    
    natural_accs, robust_accs = [], []
    for epoch in eval_epochs:
        model_path = os.path.join(args.model_dir, "checkpoint_{}.tar".format(epoch))
        if not os.path.isfile(model_path):
            break
        print("evaluating {}...".format(model_path))
        model.load_state_dict(torch.load(model_path)['state_dict'])
        natural_acc, robust_acc = eval_adv_test_whitebox(model, test_loader)
        natural_accs.append(natural_acc)
        robust_accs.append(robust_acc)
    print("natural accs: ", natural_accs)
    print("robust accs: ", robust_accs)
    np.save(os.path.join(args.model_dir, "natural_accs.npy"), natural_accs)
    np.save(os.path.join(args.model_dir, "robust_accs.npy"), robust_accs)


if __name__ == '__main__':
    main()
