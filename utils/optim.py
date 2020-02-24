import torch.optim as optim
from torch.optim import lr_scheduler


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        print("Using `SGD` optimizer")
        return optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    elif args.optimizer == 'adam':
        print("Using `Adam` optimizer")
        return optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    else:
        raise KeyError("Optimizer `{}` is not supported.".format(args.optimizer))


def get_scheduler(optimizer, args):
    if args.lr_schedule == 'step':
        print("Using `step` schedule")
        return lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.lr_gamma)
    
    elif args.lr_schedule == 'cosine':
        print("Using `cosine` schedule")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    else:
        raise KeyError("LR schedule `{}` is not supported.".format(args.lr_schedule))