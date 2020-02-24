from __future__ import absolute_import

from .loss import CrossEntropy, SelfAdaptiveTrainingCE, SelfAdaptiveTrainingSCE
from .trades import TRADES, TRADES_SAT


def get_loss(args, labels=None, num_classes=10):
    if args.loss == 'ce':
        criterion = CrossEntropy()
    
    elif args.loss == 'sat':
        criterion = SelfAdaptiveTrainingCE(labels, num_classes=num_classes, momentum=args.sat_alpha, es=args.sat_es)
    
    elif args.loss == 'sat_sce':
        alpha, beta = 1, 0.3
        criterion = SelfAdaptiveTrainingSCE(labels, num_classes=num_classes, momentum=args.sat_alpha, es=args.sat_es, alpha=alpha, beta=beta)
    
    elif args.loss == 'trades':
        criterion = TRADES(step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps, beta=args.beta)
    
    elif args.loss == 'trades_sat':
        criterion = TRADES_SAT(labels, num_classes=num_classes, momentum=args.sat_alpha , es=args.sat_es,
                        step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps, beta=args.beta)
    
    else:
        raise KeyError("Loss `{}` is not supported.".format(args.loss))

    return criterion
