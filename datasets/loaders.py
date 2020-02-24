from __future__ import absolute_import

from copy import deepcopy
import torch
import numpy as np

from .utils import get_transform
from .random_noise import label_noise, image_noise
from .datasets import CIFAR10, CIFAR100


def get_loader(args, data_aug=True):
    tform_train = get_transform(args, train=True, data_aug=data_aug)
    tform_test = get_transform(args, train=False, data_aug=data_aug)

    if args.dataset == 'cifar10':
        clean_train_set = CIFAR10(root=args.data_root, train=True, download=True, transform=tform_train)
        test_set = CIFAR10(root=args.data_root, train=False, download=True, transform=tform_test)

    
    elif args.dataset == 'cifar100':
        clean_train_set = CIFAR100(root=args.data_root, train=True, download=True, transform=tform_train)
        test_set = CIFAR100(root=args.data_root, train=False, download=True, transform=tform_test)
    
    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(args.dataset))
    
    if args.noise_rate > 0:
        noisy_train_set = deepcopy(clean_train_set)
        '''corrupt the dataset'''
        if args.noise_type == 'corrupted_label':
            label_noise(noisy_train_set, args)
        elif args.noise_type in ['Gaussian', 'random_pixels', 'shuffled_pixels']:
            image_noise(noisy_train_set, args)
        else:
            raise ValueError("Noise type {} is not supported yet.".format(args.noise_type))
        train_set = noisy_train_set
    else:
        print("Using clean dataset.")
        train_set = clean_train_set
    
    num_train = int(len(train_set) * 0.9)
    train_idx = list(range(num_train))
    val_idx = list(range(num_train, len(train_set)))
    
    if args.train_sets == 'trainval':
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.train_sets == 'train':
        train_subset = torch.utils.data.Subset(train_set, train_idx)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        raise KeyError("Train sets {} if not supported.".format(args.train_sets))
    
    # for validation, we need to disable the data augmentation
    clean_train_set_for_val = deepcopy(clean_train_set)
    clean_train_set_for_val.transform = tform_test
    if args.noise_rate > 0:
        noisy_train_set_for_val = deepcopy(noisy_train_set)
        noisy_train_set_for_val.transform = tform_test

    val_sets = []
    if 'clean_set' in args.val_sets:
        val_sets.append(clean_train_set_for_val)
    if 'noisy_set' in args.val_sets:
        val_sets.append(noisy_train_set_for_val)
    if 'test_set' in args.val_sets:
        val_sets.append(test_set)
    if 'clean_train' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(clean_train_set_for_val, train_idx))
    if 'noisy_train' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(noisy_train_set_for_val, train_idx))
    if 'clean_val' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(clean_train_set_for_val, val_idx))
    if 'noisy_val' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(noisy_train_set_for_val, val_idx))
    
    

    val_loaders = [
        torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        for val_set in val_sets
    ]

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    
    return train_loader, val_loaders, test_loader, train_set.num_classes, np.asarray(train_set.targets)
