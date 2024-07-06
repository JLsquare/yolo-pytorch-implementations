import torch
import torch.nn as nn
from intersection_over_union import intersection_over_union

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