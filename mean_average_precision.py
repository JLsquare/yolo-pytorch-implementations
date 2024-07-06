import torch
from collections import Counter
from intersection_over_union import intersection_over_union

def mean_average_precision(pred_boxes: list, true_boxes: list, iou_threshold: float = 0.5, 
                           num_classes: int = 20, box_format: str = "corners") -> float:
    """
    Calculates mean average precision

    param: pred_boxes (list) - list of lists containing all bboxes with each bboxes
    [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    param: true_boxes (list) - list of lists containing all bboxes with each bboxes
    [train_idx, class_prediction, x1, y1, x2, y2]
    param: iou_threshold (float) - threshold where predicted bboxes is correct
    param: num_classes (int) - number of classes
    param: box_format (str) - midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)
    return: float - mAP value across all classes given a specific IoU threshold
    """
    average_precisions = []
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
                
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
                
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        detections.sort(key=lambda x: x[2], reverse=True)
        true_positive = torch.zeros((len(detections)))
        false_positive = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            
            best_iou = 0
            
            for idx, ground_truth in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(ground_truth[3:]),
                    box_format=box_format
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_idx = idx
                    
            if best_iou > iou_threshold and amount_bboxes[detection[0]][best_ground_truth_idx] == 0:
                true_positive[detection_idx] = 1
                amount_bboxes[detection[0]][best_ground_truth_idx] = 1
            else:
                false_positive[detection_idx] = 1
                
        true_positive_cumsum = torch.cumsum(true_positive, dim=0)
        false_positive_cumsum = torch.cumsum(false_positive, dim=0)
        recalls = true_positive_cumsum / (total_true_bboxes + 1e-6)
        precisions = torch.divide(true_positive_cumsum, (true_positive_cumsum + false_positive_cumsum + 1e-6))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)