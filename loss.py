import torch
import torch.nn as nn
import torch.nn.functional as F
from intersection_over_union import intersection_over_union, bbox_ciou

class YOLOv1Loss(nn.Module):
    """
    YOLOv1 loss function
    """
    def __init__(self, split_size: int = 7, num_boxes: int = 2, num_classes: int = 80):
        """
        Initialize the YOLOv1 loss function

        param: split_size (int) - size of the split grid
        param: num_boxes (int) - number of bounding boxes
        param: num_classes (int) - number of classes
        """
        super(YOLOv1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCELoss()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the YOLOv1 model

        param: predictions (torch.Tensor) - predictions from the model
        param: target (torch.Tensor) - target values
        return: torch.Tensor - loss
        """
        predictions = predictions.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5)

        ious = []
        for i in range(self.num_boxes):
            box_start = self.num_classes + 1 + i * 5
            box_end = box_start + 4
            iou = intersection_over_union(
                predictions[..., box_start:box_end],
                target[..., self.num_classes+1:self.num_classes+5]
            )
            ious.append(iou.unsqueeze(0))

        ious = torch.cat(ious, dim=0)
        bestbox = ious.argmax(0)
        exists_box = target[..., self.num_classes].unsqueeze(3)

        # Box coordinates loss
        box_predictions = torch.zeros_like(target[..., self.num_classes+1:self.num_classes+5])
        for i in range(self.num_boxes):
            box_start = self.num_classes + 1 + i * 5
            box_end = box_start + 4
            mask = (bestbox == i).squeeze(-1).unsqueeze(-1).expand_as(box_predictions)
            box_predictions += predictions[..., box_start:box_end] * mask.float()
        box_predictions = exists_box * box_predictions

        box_targets = exists_box * target[..., self.num_classes+1:self.num_classes+5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        box_loss *= self.lambda_coord

        # Object loss
        pred_box = torch.zeros_like(target[..., self.num_classes:self.num_classes+1])
        for i in range(self.num_boxes):
            conf_start = self.num_classes + i * 5
            mask = (bestbox == i).squeeze(-1).unsqueeze(-1).expand_as(pred_box)
            pred_box += predictions[..., conf_start:conf_start+1] * mask.float()

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.num_classes:self.num_classes+1])
        )

        # No object loss
        no_object_loss = sum(
            self.mse(
                torch.flatten((1 - exists_box) * predictions[..., self.num_classes+i*5:self.num_classes+1+i*5], start_dim=1),
                torch.flatten((1 - exists_box) * target[..., self.num_classes:self.num_classes+1], start_dim=1)
            )
            for i in range(self.num_boxes)
        )

        no_object_loss *= self.lambda_noobj

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.num_classes], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.num_classes], end_dim=-2)
        )

        return box_loss + object_loss + no_object_loss + class_loss
    
class YOLOv3Loss(nn.Module):
    """
    YOLOv3 loss function
    """
    def __init__(self, num_classes: int = 20, num_anchors: int = 3, lambda_class: float = 1, 
                 lambda_noobj: float = 10, lambda_obj: float = 1, lambda_box: float = 10):
        """
        Initialize the YOLOv3 loss function

        param: num_classes (int) - number of classes
        param: num_anchors (int) - number of anchors per scale
        param: lambda_class (float) - class loss weight
        param: lambda_noobj (float) - no object loss weight
        param: lambda_obj (float) - object loss weight
        param: lambda_box (float) - box loss weight
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box

    def forward(self, predictions: list[torch.Tensor], targets: list[torch.Tensor], anchors: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculate the loss for the YOLOv3 model

        param: predictions (list[torch.Tensor]) - predictions from the model for each scale
        param: targets (list[torch.Tensor]) - target values for each scale
        param: anchors (list[torch.Tensor]) - anchor boxes for each scale
        return: torch.Tensor - total loss
        """
        device = predictions[0].device
        total_loss = torch.zeros(1, device=device)

        for pred, target, anchor in zip(predictions, targets, anchors):
            assert pred.shape == target.shape, f"Prediction shape {pred.shape} doesn't match target shape {target.shape}"
            obj = target[..., 0] == 1
            noobj = target[..., 0] == 0

            # No object loss
            no_object_loss = self.bce(
                pred[..., 0:1][noobj],
                target[..., 0:1][noobj]
            )

            # Object loss
            anchor = anchor.reshape(1, self.num_anchors, 1, 1, 2)
            box_preds = torch.cat([
                self.sigmoid(pred[..., 1:3]), 
                torch.exp(pred[..., 3:5]) * anchor
            ], dim=-1)
            ious = intersection_over_union(
                box_preds[obj], 
                target[..., 1:5][obj]
            ).detach()
            object_loss = self.mse(
                self.sigmoid(pred[..., 0:1][obj]), 
                ious * target[..., 0:1][obj]
            )

            # Box coordinate loss
            pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) # x, y
            target[..., 3:5] = torch.log( # w, h
                (1e-16 + target[..., 3:5] / anchor)
            )
            box_loss = self.mse(
                pred[..., 1:5][obj], 
                target[..., 1:5][obj]
            )

            # Class loss (using BCE for multi-label classification)
            class_loss = self.mse(
                pred[..., 5:][obj],
                target[..., 5:][obj]
            )

            # Compute total loss
            total_loss += (
                self.lambda_box * box_loss +
                self.lambda_obj * object_loss +
                self.lambda_noobj * no_object_loss +
                self.lambda_class * class_loss
            )

        return total_loss
    
class FocalLoss(nn.Module):
    """
    Focal Loss function
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Initialize the Focal Loss function

        param: gamma (float) - gamma value
        param: alpha (float) - alpha value
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Focal Loss

        param: input (torch.Tensor) - input tensor
        param: target (torch.Tensor) - target tensor
        return: torch.Tensor - focal
        """
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()
    
class YOLOv4Loss(nn.Module):
    def __init__(self, num_classes: int = 20, num_anchors: int = 3):
        super(YOLOv4Loss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        
        self.lambda_ciou = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 0.5
        
    def forward(self, predictions: list[torch.Tensor], targets: list[torch.Tensor], anchors: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {
            'ciou': 0.0,
            'conf': 0.0,
            'cls': 0.0
        }

        for pred, target, anchor in zip(predictions, targets, anchors):
            assert pred.shape == target.shape, f"Prediction shape {pred.shape} doesn't match target shape {target.shape}"
            
            obj_mask = target[..., 0] == 1
            noobj_mask = target[..., 0] == 0
            
            # CIoU Loss
            pred_boxes = torch.cat([
                pred[..., 1:3].sigmoid(),  # x, y
                torch.exp(pred[..., 3:5]) * anchor.unsqueeze(1).unsqueeze(1)  # w, h
            ], dim=-1)
            target_boxes = target[..., 1:5]
            ciou = bbox_ciou(pred_boxes[obj_mask], target_boxes[obj_mask])
            ciou_loss = self.lambda_ciou * (1 - ciou + 1e-8).mean()
            
            # Confidence Loss
            conf_loss_obj = self.focal(pred[..., 0][obj_mask], target[..., 0][obj_mask])
            conf_loss_noobj = self.focal(pred[..., 0][noobj_mask], target[..., 0][noobj_mask])
            conf_loss = self.lambda_conf * (conf_loss_obj + conf_loss_noobj)
            
            # Class Loss
            class_loss = self.lambda_cls * self.bce(
                pred[..., 5:][obj_mask],
                target[..., 5:][obj_mask]
            )
            
            scale_loss = ciou_loss + conf_loss + class_loss
            total_loss += scale_loss
            loss_components['ciou'] += ciou_loss.item()
            loss_components['conf'] += conf_loss.item()
            loss_components['cls'] += class_loss.item()

        return total_loss, loss_components