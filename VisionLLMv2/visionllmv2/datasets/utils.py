# Utilities for boxes/masks/keypoints operations.

import torch

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x) + 1
        bounding_boxes[index, 3] = torch.max(y) + 1

    return bounding_boxes

def boxes_to_masks(boxes: torch.Tensor, img_shape: torch.Tensor) -> torch.Tensor:
    """
    Compute the 0-1 mask from boxes.

    Args:
        boxes: [n, 4], xyxy in image shape.
        img_shape: (h, w)
    Returns:
        masks: [n, h, w], 0-1 mask.
    """

    n = boxes.shape[0]
    h, w = img_shape[:2]
    masks = torch.zeros((n, h, w), dtype=torch.float32)

    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        # box range
        x1 = max(min(x1, w - 1), 0)
        y1 = max(min(y1, h - 1), 0)
        x2 = max(min(x2, w - 1), 0)
        y2 = max(min(y2, h - 1), 0)
        masks[i, int(y1):int(y2)+1, int(x1):int(x2)+1] = 1
    
    return masks

