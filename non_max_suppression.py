import torch
from intersection_over_union import intersection_over_union

def non_max_suppression(bboxes: list, iou_threshold: float, prob_threshold: float, box_format: str = "corners"):
    """
    Does Non Max Suppression given bboxes

    param: bboxes (list) - list of lists containing all bboxes with each bboxes
    [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    param: iou_threshold (float) - threshold where predicted bboxes is correct
    param: prob_threshold (float) - threshold where predicted bboxes is correct
    param: box_format (str) - midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)
    return: bboxes (list) - bboxes after performing NMS given a specific IoU threshold
    """
    assert isinstance(bboxes, list)

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] 
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms