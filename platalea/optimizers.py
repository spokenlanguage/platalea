from torch import optim


def create_optimizer(config, net_parameters):
    """
    Create learning rate scheduler given the settings in the config dict, optimizer and data.
    :param config: configuration dict
    :param net_parameters: parameters of the network object
    :return: parameter optimizer
    """
    if 'opt' in config.keys() and config['opt'] == 'adadelta':
        optimizer = optim.Adadelta(net_parameters, lr=1, rho=0.95, eps=1e-8, weight_decay=config['l2_regularization'])
    else:
        optimizer = optim.Adam(net_parameters, lr=1, weight_decay=config['l2_regularization'])
    optimizer.zero_grad()
    return optimizer
