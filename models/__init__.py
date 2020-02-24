from __future__ import absolute_import

from .resnet import resnet18, resnet34
from .wideresnet import wrn34

model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'wrn34': wrn34,
}

def get_model(args, num_classes, **kwargs):
    return model_dict[args.arch](num_classes, **kwargs)
