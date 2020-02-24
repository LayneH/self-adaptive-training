import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropy():
    def __init__(self):
        self.crit = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, index, epoch):
        loss =  self.crit(logits, targets)
        return loss


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=10, momentum=0.9, es=40):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es

    def __call__(self, logits, targets, index, epoch):
        if epoch < self.es:
            return F.cross_entropy(logits, targets)
        
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        # obtain weights
        weights, _ = self.soft_labels[index].max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # compute cross entropy loss, without reduction
        loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)

        # sample weighted mean
        loss = (loss * weights).mean()
        return loss


class SelfAdaptiveTrainingSCE():
    def __init__(self, labels, num_classes=10, momentum=0.9, es=40, alpha=1, beta=0.3):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        self.alpha = alpha
        self.beta = beta
        print("alpha = {}, beta = {}".format(alpha, beta))


    def __call__(self, logits, targets, index, epoch):
        if epoch < self.es:
            return F.cross_entropy(logits, targets)

        # obtain prob, then update running avg
        prob = F.softmax(logits, dim=1)
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob.detach()

        # obtain weights based largest and second largest prob
        weights, _ = self.soft_labels[index].max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # use symmetric cross entropy loss, without reduction
        loss = - self.alpha * torch.sum(self.soft_labels[index] * torch.log(prob), dim=-1) \
                - self.beta * torch.sum(prob * torch.log(self.soft_labels[index]), dim=-1)

        # sample weighted mean
        loss = (loss * weights).mean()
        return loss
