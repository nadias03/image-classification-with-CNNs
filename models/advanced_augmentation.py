import numpy as np
import torch

def mixup_data(inputs, targets, alpha=0.4):
    """
    Apply MixUp augmentation to a batch of inputs and labels.

    MixUp creates new training samples by linearly combining pairs of images
    and their corresponding labels using a mixing coefficient lambda.

    Args:
        inputs (torch.Tensor): Batch of input images of shape (batch_size, C, H, W).
        targets (torch.Tensor): Batch of integer class labels of shape (batch_size, ).
        alpha (float): Hyperparameter controlling the Beta distribution used to sample lambda.
                       Higher values lead to stronger mixing.
    
    Returns:
        tuple:
            mixed_inputs (torch.Tensor): Augmented batch of images of shape (batch_size, C, H, W).
            targets_a (torch.Tensor): Original labels of shape (batch_size,).
            targets_b (torch.Tensor): Shuffled labels of shape (batch_size,).
            lam (float): Mixing coefficient sampled from Beta(alpha, alpha).
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = inputs.size(0)

    index = torch.randperm(batch_size, device=inputs.device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

    targets_a = targets
    targets_b = targets[index]

    return mixed_inputs, targets_a, targets_b, lam

def random_box(size, lam):
    """
    Generate a random box for CutMix.

    Args:
        size (torch.Size or tuple): Shape of input batch (batch_size, C, H, W)
        lam (float): Mixing coefficient.

    Returns:
        tuple:
            x1 (int): Left coordinate of the box.
            y1 (int): Top coordinate of the box.
            x2 (int): Right coordinate of the box.
            y2 (int): Bottom coordinate of the box.  
    """

    _, _, H, W = size

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

def cutmix_data(inputs, targets, alpha=0.4):
    """
    Apply CutMix augmentation to a batch of inputs and labels.

    Args:
        inputs (torch.Tensor): Batch of input images of shape (batch_size, C, H, W).
        targets (torch.Tensor): Batch of integer class labels of shape (batch_size,).
        alpha (float): Hyperparameter controlling the Beta distribution used to sample λ.

    Returns:
        tuple:
            mixed_inputs (torch.Tensor): Batch of images after CutMix augmentation.
            targets_a (torch.Tensor): Original labels of shape (batch_size,).
            targets_b (torch.Tensor): Shuffled labels of shape (batch_size,).
            lam (float): Adjusted mixing coefficient based on the actual box area.
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)

    targets_a = targets
    targets_b = targets[index]

    x1, y1, x2, y2 = random_box(inputs.size(), lam)

    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    box_area = (x2 - x1) * (y2 - y1)
    total_area = inputs.size(-1) * inputs.size(-2)
    lam = 1.0 - box_area / total_area

    return mixed_inputs, targets_a, targets_b, lam