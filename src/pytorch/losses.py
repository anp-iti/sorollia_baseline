import torch
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])

def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss

def get_loss_func(loss_type):
    if loss_type == 'clip_bce': #sigmoid
        return clip_bce
    elif loss_type == 'clip_classification': #softmax
        return clip_nll
    else:
        return 0 