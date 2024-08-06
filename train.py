import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLO
from loss import YOLOv1Loss, YOLOv3Loss, YOLOv4Loss
from mean_average_precision import mean_average_precision
from utils import (
    get_bboxes_yolov1,
    get_bboxes_yolov3,
    save_checkpoint
)
from scheduler import CosineAnnealingWarmupRestarts

def train_yolov1_epoch(train_loader: DataLoader, model: YOLO, optimizer: optim.Adam, loss_fn: YOLOv1Loss, device: str) -> float:
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

def train_yolov1(model: nn.Module, train_loader: DataLoader, optimizer: optim.Adam, loss_fn: nn.Module, device: str, 
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
        mean_loss = train_yolov1_epoch(train_loader, model, optimizer, loss_fn, device)
        print(f"Epoch [{epoch}/{epochs}], Mean loss: {mean_loss:.4f}")

        model.eval()
        with torch.no_grad():
            pred_boxes, target_boxes = get_bboxes_yolov1(
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

def train_yolov3_epoch(train_loader: DataLoader, model: YOLO, optimizer: optim.Adam, loss_fn: YOLOv3Loss, device: str, scaled_anchors: list[torch.Tensor], scaler: torch.cuda.amp.GradScaler) -> float:
    """
    Train the model for one epoch

    param: train_loader (torch.utils.data.DataLoader) - data loader for training data
    param: model (nn.Module) - model to train
    param: optimizer (torch.optim) - optimizer for model
    param: loss_fn (torch.nn) - loss function for model
    param: device (str) - device to use
    param: scaled_anchors (list[torch.Tensor]) - scaled anchors for the model
    param: scaler (torch.cuda.amp.GradScaler) - gradient scaler for mixed precision training
    return: float - mean loss for the epoch
    """
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = [yi.to(device) for yi in y]

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y, scaled_anchors)

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=loss.item(), mean_loss=mean_loss)

    return mean_loss

def train_yolov3(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, optimizer: optim.Adam, loss_fn: nn.Module, device: str,
            epochs: int, split_sizes: list, anchors: list, image_size: int = 416, start_epoch: int = 1):
    """
    Train the model

    param: model (nn.Module) - model to train
    param: train_loader (torch.utils.data.DataLoader) - data loader for training data
    param: test_loader (torch.utils.data.DataLoader) - data loader for test data
    param: optimizer (torch.optim) - optimizer for model
    param: loss_fn (torch.nn) - loss function for model
    param: device (str) - device to use
    param: epochs (int) - number of epochs to train
    param: split_sizes (list) - list of split sizes
    param: anchors (list) - list of anchors
    param: image_size (int) - size of input images
    param: start_epoch (int) - epoch to start training from
    """
    scaler = torch.cuda.amp.GradScaler()

    """
    scaled_anchors = [
        torch.tensor([(a[0] / (image_size / s), a[1] / (image_size / s)) for a in anchor_group], device=device)
        for s, anchor_group in zip(split_sizes, anchors)
    ]

    scaled_anchors = sorted(zip(split_sizes, scaled_anchors), key=lambda x: x[0], reverse=True)
    scaled_anchors = [anchor for _, anchor in scaled_anchors]
    """
    scaled_anchors = (
        torch.tensor(anchors) * torch.tensor(split_sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    for epoch in range(start_epoch, epochs + 1):
        mean_loss = train_yolov3_epoch(train_loader, model, optimizer, loss_fn, device, scaled_anchors, scaler)
        print(f"Epoch [{epoch}/{epochs}], Mean loss: {mean_loss:.4f}")

        # Evaluate on test set
        if epoch % 10 == 0 and epoch != start_epoch:
            model.eval()
            with torch.no_grad():
                pred_boxes, target_boxes = get_bboxes_yolov3(
                    test_loader, model, iou_threshold=0.45, anchors=anchors, prob_threshold=0.4,
                )
                mean_avg_prec = mean_average_precision(
                    pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint",
                )
                print(f"Test mAP: {mean_avg_prec:.4f}")

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"yolov3-{epoch}.pt")

def train_yolov4_epoch(train_loader: DataLoader, model: YOLO, optimizer: optim.Optimizer, 
                       loss_fn: YOLOv4Loss, device: str, scaled_anchors: list[torch.Tensor], 
                       scaler: amp.GradScaler) -> dict:
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = {"total": [], "ciou": [], "conf": [], "cls": []}

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = [yi.to(device) for yi in y]

        with amp.autocast():
            out = model(x)

            y = y[::-1]
            
            loss = loss_fn(out, y, scaled_anchors)
            total_loss, loss_components = loss

        losses["total"].append(total_loss.item())
        losses["ciou"].append(loss_components['ciou'])
        losses["conf"].append(loss_components['conf'])
        losses["cls"].append(loss_components['cls'])

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_losses = {k: sum(v) / len(v) for k, v in losses.items()}
        loop.set_postfix(**mean_losses)

    return mean_losses

def train_yolov4(model: YOLO, train_loader: DataLoader, test_loader: DataLoader, 
                 optimizer: optim.Optimizer, loss_fn: YOLOv4Loss, device: str,
                 epochs: int, split_sizes: list, anchors: list, image_size: int = 416, start_epoch: int = 1):
    """
    Train the YOLOv4 model

    param: model (YOLO) - YOLOv4 model to train
    param: train_loader (DataLoader) - data loader for training data
    param: test_loader (DataLoader) - data loader for test data
    param: optimizer (optim.Optimizer) - optimizer for model
    param: loss_fn (YOLOv4Loss) - loss function for model
    param: device (str) - device to use
    param: epochs (int) - number of epochs to train
    param: split_sizes (list) - list of split sizes
    param: anchors (list) - list of anchors
    param: start_epoch (int) - epoch to start training from
    """
    scaler = amp.GradScaler()

    """
    scaled_anchors = [
        torch.tensor([(a[0] / (image_size / s), a[1] / (image_size / s)) for a in anchor_group], device=device)
        for s, anchor_group in zip(split_sizes, anchors)
    ]

    scaled_anchors = sorted(zip(split_sizes, scaled_anchors), key=lambda x: x[0], reverse=True)
    scaled_anchors = [anchor for _, anchor in scaled_anchors]
    """
    scaled_anchors = (
        torch.tensor(anchors) * torch.tensor(split_sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=epochs,
        cycle_mult=1.0,
        max_lr=0.01,
        min_lr=0.0001,
        warmup_steps=3,
        gamma=0.5
    )

    for epoch in range(start_epoch, epochs + 1):
        mean_losses = train_yolov4_epoch(train_loader, model, optimizer, loss_fn, device, scaled_anchors, scaler)
        
        print(f"Epoch [{epoch}/{epochs}]")
        #for k, v in mean_losses.items():
        #    print(f"{k.capitalize()} Loss: {v:.4f}")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            pred_boxes, target_boxes = get_bboxes_yolov3(
                test_loader, model, iou_threshold=0.5, anchors=anchors, prob_threshold=0.95,
            )
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint",
            )
        print(f"Test mAP: {mean_avg_prec:.4f}")

        # Learning rate scheduling
        scheduler.step()

        if epoch % 10 == 0 and epoch != start_epoch:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"yolov4-{epoch}.pt")