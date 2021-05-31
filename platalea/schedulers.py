import logging
import numpy as np
from torch.optim import lr_scheduler


def cyclic(optimizer, n_batches, max_lr, min_lr):
    stepsize = n_batches * 4
    logging.info("Setting stepsize of {}".format(stepsize))

    def learning_rate(iteration):
        lr = (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1)) + min_lr
        return lr

    scheduler = lr_scheduler.LambdaLR(optimizer, learning_rate, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press
    # between min and max lr
    return scheduler


def noam(optimizer, d_model, warmup_steps=4000):
    """
    Learning rate schedule as given in Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf),
    colloquially named "Noam" after its second author Noam Shazeer.

    Amplitude of the schedule's rate is modulated by the model size `d_model`. The paper used 4000
    warmup steps in 100000 total step training runs, so this is the default value for `warmup_steps`.
    """

    def learning_rate(iteration):
        step = iteration + 1  # skip zero to avoid divide by zero
        lr = d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)
        return lr

    scheduler = lr_scheduler.LambdaLR(optimizer, learning_rate)
    return scheduler


def constant(optimizer, lr):
    """
    Constant learning rate scheduler. The most trivial kind of scheduler, that keeps the learning rate constant.
    :param optimizer:
    :param lr:
    :return:
    """
    logging.info("Using constant learning rate of {}".format(lr))

    def learning_rate(_iteration):
        return lr

    return lr_scheduler.LambdaLR(optimizer, learning_rate, last_epoch=-1)


def create_scheduler(config, optimizer, data):
    """
    Create learning rate scheduler given the settings in the config dict, optimizer and data.
    :param config: configuration dict
    :param optimizer: pytorch optimizer object
    :param data: dict containing the dataset
    :return: learning rate scheduler
    """
    if 'lr' in config.keys():
        raise KeyError('Illegal keyword "lr" used in config. Use keyword "constant_lr" instead.')

    configured_scheduler = config.get('lr_scheduler')

    if configured_scheduler is None or configured_scheduler == 'cyclic':
        scheduler = cyclic(optimizer, len(data['train']), max_lr=config['max_lr'],
                           min_lr=config['min_lr'])
    elif configured_scheduler == 'noam':
        scheduler = noam(optimizer, config['d_model'])
    elif configured_scheduler == 'constant':
        scheduler = constant(optimizer, config['constant_lr'])
    else:
        raise Exception(
            "lr_scheduler config value " + configured_scheduler + " is invalid, use cyclic or noam or constant")
    return scheduler
