from torch import optim

import platalea.schedulers


def create_scheduler(config, optimizer, data):
    if 'lr' in config.keys():
        raise KeyError('Illegal keyword "lr" used in config. Use keyword "constant_lr" instead.')

    configured_scheduler = config.get('lr_scheduler')

    if configured_scheduler is None or configured_scheduler == 'cyclic':
        scheduler = platalea.schedulers.cyclic(optimizer, len(data['train']), max_lr=config['max_lr'],
                                               min_lr=config['min_lr'])
    elif configured_scheduler == 'noam':
        scheduler = platalea.schedulers.noam(optimizer, config['d_model'])
    elif configured_scheduler == 'constant':
        scheduler = platalea.schedulers.constant(optimizer, config['constant_lr'])
    else:
        raise Exception(
            "lr_scheduler config value " + configured_scheduler + " is invalid, use cyclic or noam or constant")
    return scheduler


def create_optimizer(net_parameters, regularization, config={}):
    if 'opt' in config.keys() and config['opt'] == 'adadelta':
        optimizer = optim.Adadelta(net_parameters, lr=1, rho=0.95, eps=1e-8)
    else:
        optimizer = optim.Adam(net_parameters, lr=1, weight_decay=regularization)
    optimizer.zero_grad()
    return optimizer