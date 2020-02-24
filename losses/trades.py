import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TRADES():
    def __init__(self, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta

    def __call__(self, x_natural, y, index, epoch, model, optimizer):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()
        batch_size = len(x_natural)

        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        # calculate robust loss
        logits = model(x_natural)

        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return logits, loss


class TRADES_SAT():
    def __init__(self, labels, num_classes=10, momentum=0.9, es=70, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta

    def __call__(self, x_natural, y, index, epoch, model, optimizer):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()
        batch_size = len(x_natural)

        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        # calculate robust loss
        logits = model(x_natural)
        if epoch < self.es:
            loss_natural = F.cross_entropy(logits, y)
        else:
            prob = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob
            weights, _ = self.soft_labels[index].max(dim=1)
            weights *= logits.shape[0] / weights.sum()
            loss_natural = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)
            loss_natural = (loss_natural * weights).mean()
        
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return logits, loss
