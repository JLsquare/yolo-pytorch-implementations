import torch
import math

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

def iou_width_height(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate intersection over union for width and height

    param: boxes1 (torch.Tensor) - bounding boxes 1
    param: boxes2 (torch.Tensor) - bounding boxes 2
    return: torch.Tensor - intersection over union for width and height
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = (boxes1[..., 0] * boxes1[..., 1]) + (boxes2[..., 0] * boxes2[..., 1]) - intersection
    return intersection / union

def bbox_ciou(box1, box2):
    """
    Calculate the Complete IoU, the IoU that accounts for the distance between the boxes

    param: box1 (torch.Tensor) - bounding box 1
    param: box2 (torch.Tensor) - bounding box 2
    return: torch.Tensor - complete IoU
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7
    union = w1 * h1 + w2 * h2 - inter + 1e-7

    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c2 = cw ** 2 + ch ** 2 + 1e-7  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared

    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou