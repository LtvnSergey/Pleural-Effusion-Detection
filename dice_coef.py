import numpy as np
import torch
from utils import show_dice_example
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def dice_coefficient(mask1, mask2):
    """
    Calculate DICE coefficient between masks in batch
    :param mask1: predicted mask
    :param mask2: ground truth mask
    :return: DICE coefficient
    """

    # Calculate  summary area for each masks
    sum1 = mask1.sum(axis=(1, 2, 3))
    sum2 = mask2.sum(axis=(1, 2, 3))

    # Calculate intersection
    intersection = (mask1 * mask2).sum(axis=(1, 2, 3))

    #  Calculate mean dice coefficient (set value to 1 if both masks are empty)
    coefficient = (2 * intersection / (sum1 + sum2)).nan_to_num(1.).mean().item()

    return  coefficient


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


if __name__ == '__main__':

    # Create two random  masks  as single-item batches
    # One mask will be the predicted one
    # Another mask will be ground truth
    np.random.seed(2222)
    mask_pred = torch.Tensor(np.random.randint(low=-100, high=100, size=(2, 1, 3, 3)))
    mask_true = torch.Tensor(np.random.randint(low=0, high=2, size=(2, 1, 3, 3)))

    #
    # # Calculate dice coefficient
    # dice_pred_true = dice_coefficient(mask_pred, mask_true)
    # dice_true_true = dice_coefficient(mask_true, mask_true)
    #
    # # print(mask_pred*0)
    loss = DiceLoss()
    with torch.no_grad():
        # dice_loss_pred_true = loss(mask_pred, mask_true)
        dice_loss_pred_true = loss(mask_pred, mask_true)

    # print(dice_pred_true, dice_true_true)
    print(dice_loss_pred_true)

    loss = smp.losses.DiceLoss(mode='binary')
    dice_loss_pred_true = loss(mask_pred, mask_true)
    print(dice_loss_pred_true)

    # # Example with different masks
    # show_dice_example(mask_pred, mask_true, dice_pred_true)
    #
    # # Example of identical masks
    # show_dice_example(mask_true, mask_true, dice_true_true)
