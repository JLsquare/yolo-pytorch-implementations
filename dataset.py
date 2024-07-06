import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    """
    Dataset class for the PASCAL VOC dataset for YOLOv1
    """
    def __init__(self, csv_file: str, img_dir: str, label_dir: str, split_size: int = 7, 
                 num_boxes: int = 2, num_classes: int = 20, transform=None):
        """
        Initialize the VOC Dataset

        param: csv_file (str) - path to the csv file with annotations
        param: img_dir (str) - path to the directory with images
        param: label_dir (str) - path to the directory with labels
        param: split_size (int) - size of the split grid
        param: num_boxes (int) - number of bounding boxes
        param: num_classes (int) - number of classes
        param: transform (torchvision.transforms) - transformations to apply to images
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self) -> int:
        """
        Get the length of the dataset

        return: int - length of the dataset
        """
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset

        param: index (int) - index of the item to get
        return: tuple - image and label matrix
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.split_size, self.split_size, self.num_classes + 5 * self.num_boxes))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.split_size * y), int(self.split_size * x)
            x_cell, y_cell = self.split_size * x - j, self.split_size * y - i

            width_cell, height_cell = (
                width * self.split_size,
                height * self.split_size,
            )

            if label_matrix[i, j, self.num_classes] == 0:
                label_matrix[i, j, self.num_classes] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.num_classes+1:self.num_classes+5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix