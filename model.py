import torch
import torch.nn as nn

yolov1_config = [
    ("Conv2d", 7, 64, 2, 3),
    ("MaxPool2d", 2, 2),
    ("Conv2d", 3, 192, 1, 1),
    ("MaxPool2d", 2, 2),
    ("Conv2d", 1, 128, 1, 0),
    ("Conv2d", 3, 256, 1, 1),
    ("Conv2d", 1, 256, 1, 0),
    ("Conv2d", 3, 512, 1, 1),
    ("MaxPool2d", 2, 2),
    ("Conv2d", 1, 256, 1, 0),
    ("Conv2d", 3, 512, 1, 1),
    ("Conv2d", 1, 256, 1, 0),
    ("Conv2d", 3, 512, 1, 1),
    ("Conv2d", 1, 256, 1, 0),
    ("Conv2d", 3, 512, 1, 1),
    ("Conv2d", 1, 256, 1, 0),
    ("Conv2d", 3, 512, 1, 1),
    ("Conv2d", 1, 512, 1, 0),
    ("Conv2d", 3, 1024, 1, 1),
    ("MaxPool2d", 2, 2),
    ("Conv2d", 1, 512, 1, 0),
    ("Conv2d", 3, 1024, 1, 1),
    ("Conv2d", 1, 512, 1, 0),
    ("Conv2d", 3, 1024, 1, 1),
    ("Conv2d", 3, 1024, 1, 1),
    ("Conv2d", 3, 1024, 2, 1),
    ("Conv2d", 3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    """
    CNN block for YOLOv1
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        """
        Constructor for CNNBlock

        param: in_channels (int) - number of input channels
        param: out_channels (int) - number of output channels
        param: kernel_size (int) - size of kernel
        param: stride (int) - stride of kernel
        param: padding (int) - padding of kernel
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNNBlock

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    
class YOLOv1(nn.Module):
    """
    YOLOv1 model
    """
    def __init__(self, config: list[tuple] = yolov1_config, in_channels: int = 3, split_size: int = 7, 
                 num_boxes: int = 2, num_classes: int = 20, linear_size: int = 4096):
        """
        Constructor for YOLOv1

        param: config (list[tuple]) - configuration for YOLOv1
        param: in_channels (int) - number of input channels
        param: split_size (int) - split size of image
        param: num_boxes (int) - number of boxes per cell
        param: num_classes (int) - number of classes
        param: linear_size (int) - size of linear layer
        """
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(config)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes, linear_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for YOLOv1

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.conv_layers(x)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self, config: list[tuple]) -> nn.Sequential:
        """
        Create convolutional layers for YOLOv1 from config

        param: config (list[tuple]) - configuration for YOLOv1
        return: nn.Sequential - sequential convolutional layers
        """
        layers = []
        in_channels = self.in_channels

        for x in config:
            if x[0] == "Conv2d":
                layers.append(
                    CNNBlock(in_channels, x[2], kernel_size=x[1], stride=x[3], padding=x[4])
                )
                in_channels = x[2]
            elif x[0] == "MaxPool2d":
                layers.append(
                    nn.MaxPool2d(kernel_size=x[1], stride=x[2])
                )
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20, linear_size: int = 1024):
        """
        Create fully connected layers for YOLOv1 to convert to bounding boxes

        param: split_size (int) - split size of image
        param: num_boxes (int) - number of boxes per cell
        param: num_classes (int) - number of classes
        param: linear_size (int) - size of linear layer
        return: nn.Sequential - sequential fully connected layers
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, linear_size),
            nn.LeakyReLU(0.1),
            nn.Linear(linear_size, split_size * split_size * (num_classes + num_boxes * 5)),
        )
    
if __name__ == "__main__":
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20, linear_size=1024)
    print(model)
    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)