import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from non_max_suppression import non_max_suppression

def plot_image(image: torch.Tensor, boxes: list):
    """
    Plots the image with bounding boxes

    param: image (tensor) - image to plot
    param: boxes (list) - list of boxes in format (x, y, w, h)
    """
    img = np.array(image)
    height, width, _ = img.shape

    _, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()


def get_bboxes_yolov1(loader: torch.utils.data.DataLoader, model: nn.Module, iou_threshold: float, 
               prob_threshold: float, box_format: str = "midpoint", device: str = "cuda", 
               split_size: int = 7, num_boxes: int = 2, num_classes: int = 20) -> list:
    """
    Get bboxes from loader for YOLOv1

    param: loader (torch.utils.data.DataLoader) - data loader to get data from
    param: model (nn.Module) - model to get predictions from
    param: iou_threshold (float) - threshold where predicted bboxes is correct
    param: prob_threshold (float) - threshold where predicted bboxes is correct
    param: box_format (str) - midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)
    param: device (str) - cuda/cpu
    param: split_size (int) - split size of image
    param: num_boxes (int) - number of boxes per cell
    param: num_classes (int) - number of classes
    return: all_pred_boxes (list) - list of all predicted boxes
    """
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for (x, labels) in loader:
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cells_to_bboxes_yolov1(labels, split_size, num_boxes, num_classes)
        bboxes = cells_to_bboxes_yolov1(predictions, split_size, num_boxes, num_classes)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def get_bboxes_yolov3(loader: torch.utils.data.DataLoader, model: nn.Module, iou_threshold: float,
                      anchors: list, prob_threshold: float, box_format: str = "midpoint", device: str = "cuda") -> list:
    """
    Get bboxes from loader for YOLOv3 and YOLOv4 (multiple scales and anchors)

    param: loader (torch.utils.data.DataLoader) - data loader to get data from
    param: model (nn.Module) - model to get predictions from
    param: iou_threshold (float) - threshold where predicted bboxes is correct
    param: anchors (list) - anchors used in model
    param: threshold (float) - threshold where predicted bboxes is correct
    param: box_format (str) - midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)
    param: device (str) - cuda/cpu
    return: all_pred_boxes (list) - list of all predicted boxes
    """
    """
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)
        labels = labels[2].to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            split_size = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * split_size

            boxes_scale_i = cells_to_bboxes_yolov3(
                predictions[i], anchor, splite_size=split_size
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        true_bboxes = cells_to_bboxes_yolov3(
            labels, anchor, splite_size=split_size
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes=bboxes[idx], iou_threshold=iou_threshold, prob_threshold=prob_threshold, box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    print(f"Processed all batches, found {len(all_pred_boxes)} pred boxes and {len(all_true_boxes)} true boxes")  # Add this line
    model.train()
    return all_pred_boxes, all_true_boxes
    """
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)
        labels = [label.to(device) for label in labels]

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes_yolov3(predictions[i], anchor, split_size=S)
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        true_bboxes = cells_to_bboxes_yolov3(labels[2], anchor, split_size=labels[2].shape[2])

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes=bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions: torch.Tensor, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20) -> torch.Tensor:
    """
    Converts bounding boxes output from model into proper format

    param: predictions (torch.Tensor) - output from model
    param: split_size (int) - split size of image
    param: num_boxes (int) - number of boxes per cell
    param: num_classes (int) - number of classes
    return: converted_preds (torch.Tensor) - converted predictions
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, split_size, split_size, num_classes + num_boxes * 5)
    
    all_bboxes = [predictions[..., num_classes+1+i*5:num_classes+5+i*5] for i in range(num_boxes)]
    all_scores = [predictions[..., num_classes+i*5].unsqueeze(0) for i in range(num_boxes)]
    scores = torch.cat(all_scores, dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = torch.zeros_like(all_bboxes[0])
    for i, bbox in enumerate(all_bboxes):
        best_boxes += (best_box == i) * bbox
    
    cell_indices = torch.arange(split_size).repeat(batch_size, split_size, 1).unsqueeze(-1)
    x = 1 / split_size * (best_boxes[..., :1] + cell_indices)
    y = 1 / split_size * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / split_size * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :num_classes].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(scores, dim=0)[0].unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cells_to_bboxes_yolov1(out: torch.Tensor, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20) -> list:
    """
    Converts output from model into bounding boxes

    param: out (torch.Tensor) - output from model
    param: split_size (int) - split size of image
    param: num_boxes (int) - number of boxes per cell
    param: num_classes (int) - number of classes
    return: list of (class, x, y, w, h, prob) boxes
    """
    converted_pred = convert_cellboxes(out, split_size, num_boxes, num_classes).reshape(out.shape[0], split_size * split_size, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(split_size * split_size):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def cells_to_bboxes_yolov3(predictions: torch.Tensor, anchors, split_size):
    """
    Scales the predictions coming from the model to actual values

    param: predictions (torch.Tensor) - tensor of size (N, 3, S, S, num_classes + 5)
    param: anchors (torch.Tensor) - tensor of size (3, 2)
    param: splite_size (int) - number of cells the image is divided in on the width/height
    return: converted_bboxes (list) - size (N, num_anchors * S * S, 6) [train_idx, class_pred, prob_score
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    scores = torch.sigmoid(predictions[..., 0:1])
    best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
    box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
    box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * anchors

    cell_indices = (
        torch.arange(split_size)
        .repeat(predictions.shape[0], 3, split_size, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    #x = 1 / splite_size * (box_predictions[..., 0:1] + cell_indices)
    #y = 1 / splite_size * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    #w_h = 1 / splite_size * box_predictions[..., 2:4]
    x = (box_predictions[..., 0:1] + cell_indices) / split_size
    y = (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / split_size
    w_h = box_predictions[..., 2:4] / split_size

    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * split_size * split_size, 6)
    return converted_bboxes.tolist()

def save_checkpoint(state: dict, filename: str = "my_checkpoint.pth.tar"):
    """
    Saves model to checkpoint

    param state: dict, state of model to save
    param filename: str, filename for saving checkpoint
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint: dict, model: nn.Module, optimizer: torch.optim.Optimizer | None = None):
    """
    Loads model parameters from a checkpoint

    param: checkpoint (dict) - loaded checkpoint
    param: model (nn.Module) - model to load parameters into
    param: optimizer (torch.optim) - optimizer for model
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
def trainable_parameters(model: nn.Module) -> int:
    """
    Get number of trainable parameters in model

    param: model (torch.nn.Module) - model to get parameters from
    return: int - number of trainable parameters
    """
    return sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))