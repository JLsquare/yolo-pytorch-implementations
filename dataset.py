import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from intersection_over_union import iou_width_height

class BaseYOLODataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, img_dir: str, label_dir: str, transform=None):
        """
        Initialize the VOC / COCO Dataset for YOLO models

        param: csv_file (str) - path to the csv file with annotations
        param: img_dir (str) - path to the directory with images
        param: label_dir (str) - path to the directory with labels
        param: transform (torchvision.transforms) - transformations to apply to images
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the number of items in the dataset

        return: int - number of items in the dataset
        """
        return len(self.annotations)

    def _get_image_and_boxes(self, index: int) -> tuple[np.ndarray, list]:
        """
        Get the image and bounding boxes for an item in the dataset

        param: index (int) - index of the item to get
        return: tuple - image and bounding boxes
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        #TODO: Change from albumentations to torchvision.transforms v2
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        return image, bboxes

class YOLOv1Dataset(BaseYOLODataset):
    """
    Dataset class for YOLOv1
    """
    def __init__(self, csv_file: str, img_dir: str, label_dir: str, split_size: int = 7, 
                 num_boxes: int = 2, num_classes: int = 20, transform=None):
        """
        Initialize the Dataset

        param: csv_file (str) - path to the csv file with annotations
        param: img_dir (str) - path to the directory with images
        param: label_dir (str) - path to the directory with labels
        param: split_size (int) - size of the split grid
        param: num_boxes (int) - number of bounding boxes
        param: num_classes (int) - number of classes
        param: transform (torchvision.transforms) - transformations to apply to images
        """
        super().__init__(csv_file, img_dir, label_dir, transform)
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset

        param: index (int) - index of the item to get
        return: tuple - image and target
        """
        image, boxes = self._get_image_and_boxes(index)

        target = torch.zeros((self.split_size, self.split_size, self.num_classes + 5 * self.num_boxes)) # [7, 7, 50]
        for box in boxes: 
            x, y, width, height, class_label = box
            class_label = int(class_label)

            i, j = int(self.split_size * y), int(self.split_size * x) # [0, 7], [0, 7], which cell
            x_cell, y_cell = self.split_size * x - j, self.split_size * y - i # [0, 1], [0, 1], relative position of box center inside cell

            width_cell, height_cell = (
                width * self.split_size,
                height * self.split_size,
            ) # relative size of box inside cell

            if target[i, j, self.num_classes] == 0: # if no object in cell [i, j] already
                target[i, j, self.num_classes] = 1 # set objectness to 1
                box_coordinates = torch.tensor( 
                    [x_cell, y_cell, width_cell, height_cell]
                ) 
                target[i, j, self.num_classes+1:self.num_classes+5] = box_coordinates # set box coordinates
                target[i, j, class_label] = 1 # set class label (one-hot encoding)

        return image, target
    
class YOLOv3Dataset(BaseYOLODataset):
    """
    Dataset class for YOLOv3
    """
    def __init__(self, csv_file: str, img_dir: str, label_dir: str, anchors: list, 
                 split_sizes: list = [13,26,52], num_classes: int = 20, transform=None):
        """
        Initialize the Dataset

        param: csv (str) - path to the csv file with annotations
        param: img_dir (str) - path to the directory with images
        param: label_dir (str) - path to the directory with labels
        param: anchors (list) - list of anchors
        param: split_sizes (list) - list of grid sizes
        param: num_classes (int) - number of classes
        param: transform (torchvision.transforms) - transformations to apply to images
        """
        super().__init__(csv_file, img_dir, label_dir, transform)
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.split_sizes = split_sizes
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        """
        Get an item from the dataset

        param: index (int) - index of the item to get
        return: tuple - image and targets
        """
        image, bboxes = self._get_image_and_boxes(index)

        targets = [torch.zeros((self.num_anchors_per_scale, split_size, split_size, 5 + self.num_classes)) for split_size in self.split_sizes]

        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # which scale [0,1,2]
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # which anchor on scale [0,1,2]
                split_size = self.split_sizes[scale_idx]
                i, j = int(split_size * y), int(split_size * x) # which cell, x = 0.5, split_size = 13, i = int(6.5) = 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # check if anchor already assigned

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = split_size * x - j, split_size * y - i # relative position of box center inside cell ([0,1], [0,1])
                    width_cell, height_cell = ( # relative size of box inside cell (can be bigger than 1)
                        width * split_size,
                        height * split_size,
                    )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = torch.tensor([x_cell, y_cell, width_cell, height_cell]) # box coordinates and size
                    targets[scale_idx][anchor_on_scale, i, j, 5 + int(class_label)] = 1 # class label (one-hot encoding)
                    has_anchor[scale_idx] = True # assign anchor

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction

        return image, tuple(targets)
    
class YOLOv4Dataset(BaseYOLODataset):
    def __init__(self, csv_file: str, img_dir: str, label_dir: str, anchors: list, 
                 split_sizes: list = [13,26,52], num_classes: int = 20, transform=None):
        """
        Initialize the Dataset

        param: csv_file (str) - path to the csv file with annotations
        param: img_dir (str) - path to the directory with images
        param: label_dir (str) - path to the directory with labels
        param: anchors (list) - list of anchors
        param: split_sizes (list) - list of grid sizes
        param: num_classes (int) - number of classes
        param: transform (torchvision.transforms) - transformations to apply to images
        """
        super().__init__(csv_file, img_dir, label_dir, transform)
        self.split_sizes = split_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5

    def anchor_free_guidance(self, box: list, anchors: torch.Tensor) -> torch.Tensor:
        """
        Implement a basic version of Anchor Free Guidance

        param: box (list) - bounding box coordinates
        param: anchors (torch.Tensor) - anchor boxes
        return: torch.Tensor - guidance score
        """
        box_w, box_h = torch.tensor(box[2:4])
        anchor_w, anchor_h = anchors[:, 0], anchors[:, 1]
        
        # calculate aspect ratio similarity
        box_ratio = box_w / box_h
        anchor_ratio = anchor_w / anchor_h
        aspect_ratio_similarity = torch.min(box_ratio, anchor_ratio) / torch.max(box_ratio, anchor_ratio)
        
        # calculate size similarity
        box_area = box_w * box_h
        anchor_area = anchor_w * anchor_h
        size_similarity = torch.min(box_area, anchor_area) / torch.max(box_area, anchor_area)
        
        # combine similarities
        guidance_score = aspect_ratio_similarity * size_similarity
        return guidance_score

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        """
        Get an item from the dataset

        param: index (int) - index of the item to get
        return: tuple - image and targets
        """
        image, bboxes = self._get_image_and_boxes(index)

        targets = [torch.zeros((self.num_anchors_per_scale, split_size, split_size, 5 + self.num_classes)) for split_size in self.split_sizes]

        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) # [3], calculate iou with anchors
            guidance_scores = self.anchor_free_guidance(box, self.anchors) # [3], calculate guidance scores
            combined_scores = iou_anchors * guidance_scores # [3], combine iou and guidance scores
            anchor_indices = combined_scores.argsort(descending=True, dim=0) # sort anchors by combined scores
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # which scale [0,1,2]
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # which anchor on scale [0,1,2]
                split_size = self.split_sizes[scale_idx]
                i, j = int(split_size * y), int(split_size * x) # which cell, x = 0.5, split_size = 13, i = int(6.5) = 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # check if anchor already assigned

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # set objectness to 1
                    x_cell, y_cell = split_size * x - j, split_size * y - i # relative position of box center inside cell ([0,1], [0,1])
                    width_cell, height_cell = width * split_size, height * split_size # relative size of box inside cell (can be bigger than 1)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) # box coordinates and size
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5 + int(class_label)] = 1 # class label (one-hot encoding)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and combined_scores[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction

        return image, tuple(targets)