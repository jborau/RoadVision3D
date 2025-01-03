import torch
import torch.nn as nn


def focal_loss(prediction, target, alpha=0.25, beta=2.):
    positive_index = target.eq(1).float()
    negative_index = target.lt(1).float()

    negative_weights = torch.pow(1 - target, beta)
    loss = 0.

    positive_loss = torch.log(prediction) \
                    * torch.pow(1 - prediction, alpha) * positive_index
    negative_loss = torch.log(1 - prediction) \
                    * torch.pow(prediction, alpha) * negative_weights * negative_index

    num_positive = positive_index.float().sum()
    positive_loss = positive_loss.sum()
    negative_loss = negative_loss.sum()

    if num_positive == 0:
        loss -= negative_loss
    else:
        loss -= (positive_loss + negative_loss) / num_positive

    return loss


def focal_loss_cornernet(input, target, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    '''

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    loss = 0

    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * neg_weights

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()

