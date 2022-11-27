import numpy as np
import torch
from utils import show_dice_example

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


if __name__ == '__main__':

    # Create two random  masks  as single-item batches
    # One mask will be the predicted one
    # Another mask will be ground truth
    np.random.seed(2222)
    mask_pred = torch.Tensor(np.random.randint(low=0, high=2, size=(1, 1, 10, 10)))
    mask_true = torch.Tensor(np.random.randint(low=0, high=2, size=(1, 1, 10, 10)))

    # Calculate dice coefficient
    dice_pred_true = dice_coefficient(mask_pred, mask_true)
    dice_true_true = dice_coefficient(mask_true, mask_true)

    # Example with different masks
    show_dice_example(mask_pred, mask_true, dice_pred_true)

    # Example of identical masks
    show_dice_example(mask_true, mask_true, dice_true_true)
