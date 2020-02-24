import torchvision.transforms as transforms

def get_mean_std(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    
    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(args.dataset))
    return mean, std


def get_transform(args, train=True, data_aug=True):
    mean, std = get_mean_std(args)

    if args.turn_off_aug:
        print("Data augmentation is turned off!")
        train = False
    
    train = (train and data_aug)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if train:
            tform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            tform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    
    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(args.dataset))

    return tform
