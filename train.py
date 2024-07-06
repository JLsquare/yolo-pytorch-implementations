import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from loss import YOLOv1Loss
from mean_average_precision import mean_average_precision
from utils import (
    get_bboxes,
    save_checkpoint,
)

def train_epoch(train_loader: DataLoader, model: YOLOv1, optimizer: optim.Adam, loss_fn: YOLOv1Loss, device: str) -> float:
    """
    Train the model for one epoch

    param: train_loader (torch.utils.data.DataLoader) - data loader for training data
    param: model (nn.Module) - model to train
    param: optimizer (torch.optim) - optimizer for model
    param: loss_fn (torch.nn) - loss function for model
    param: device (str) - device to use
    return: float - mean loss for the epoch
    """
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for (x, y) in loop:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return sum(mean_loss) / len(mean_loss)

def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Adam, loss_fn: nn.Module, device: str, 
          epochs: int, start_epoch: int = 1, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20):
    """
    Train the model

    param: model (nn.Module) - model to train
    param: train_loader (torch.utils.data.DataLoader) - data loader for training data
    param: optimizer (torch.optim) - optimizer for model
    param: loss_fn (torch.nn) - loss function for model
    param: device (str) - device to use
    param: epochs (int) - number of epochs to train
    param: start_epoch (int) - epoch to start training from
    param: split_size (int) - split size of image
    param: num_boxes (int) - number of boxes per cell
    param: num_classes (int) - number of classes
    """
    for epoch in range(start_epoch, epochs + 1):
        mean_loss = train_epoch(train_loader, model, optimizer, loss_fn, device)
        print(f"Epoch [{epoch}/{epochs}], Mean loss: {mean_loss:.4f}")

        model.eval()
        with torch.no_grad():
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, prob_threshold=0.4, box_format="midpoint",
                device=device, split_size=split_size, num_boxes=num_boxes, num_classes=num_classes,
            )
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint",
            )
        print(f"Epoch [{epoch}/{epochs}], Train mAP: {mean_avg_prec:.4f}")

        if epoch % 10 == 0 and epoch != start_epoch:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"yolov1-{epoch}.pt")
