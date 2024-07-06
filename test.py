import torch
from torch.utils.data import DataLoader
from model import YOLOv1
from utils import cellboxes_to_boxes, non_max_suppression, plot_image

def test_model(count: int, model: YOLOv1, test_loader: DataLoader, device: str, split_size: int, num_boxes: int, num_classes: int):
    """
    Test the model

    param: count (int) - number of images to test
    param: model (YOLOv1) - model to test
    param: test_loader (DataLoader) - data loader for testing data
    param: device (str) - device to use
    param: split_size (int) - split size of image
    param: num_boxes (int) - number of boxes per cell
    param: num_classes (int) - number of classes
    """
    model.eval()

    for x, _ in test_loader:
        x = x.to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        for idx in range(batch_size):
            bboxes = cellboxes_to_boxes(predictions, split_size, num_boxes, num_classes)
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, prob_threshold=0.4, box_format="midpoint")
            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

            count -= 1
            if count == 0:
                return

