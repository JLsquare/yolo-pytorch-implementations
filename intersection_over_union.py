import torch

def intersection_over_union(box1: torch.Tensor, box2: torch.Tensor, box_format: str = "midpoint") -> torch.Tensor:
    """
    Calculates the intersection over union of two bounding boxes

    param: box1 (torch.tensor) - bounding box 1
    param: box2 (torch.tensor) - bounding box 2
    param: box_format (str) - midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)
    return: torch.tensor - intersection over union
    """
    if box_format == "midpoint":
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2   
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = box1[..., 0:1]
        box1_y1 = box1[..., 1:2]
        box1_x2 = box1[..., 2:3]
        box1_y2 = box1[..., 3:4]
        box2_x1 = box2[..., 0:1]
        box2_y1 = box2[..., 1:2]
        box2_x2 = box2[..., 2:3]
        box2_y2 = box2[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)