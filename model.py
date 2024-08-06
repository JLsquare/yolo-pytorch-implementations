import torch
import torch.nn as nn

yolov1_config = [
    # (from, module, *args)
    # Backbone
    (-1, "Conv", 64, 7, 2, 3),
    (-1, "MaxPool2d", 2, 2),
    (-1, "Conv", 192, 3, 1, 1),
    (-1, "MaxPool2d", 2, 2),
    (-1, "Conv", 128, 1, 1, 0),
    (-1, "Conv", 256, 3, 1, 1),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "MaxPool2d", 2, 2),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "Conv", 512, 1, 1, 0),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "MaxPool2d", 2, 2),
    (-1, "Conv", 512, 1, 1, 0),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "Conv", 512, 1, 1, 0),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "Conv", 1024, 3, 2, 1),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "YOLOv1Detect", 7, 2, 20, 1024),
]

yolov3_config = [
    # (from, module, *args)
    # Backbone
    (-1, "Conv", 32, 3, 1, 1),
    (-1, "Conv", 64, 3, 2, 1),
    (-1, "Bottleneck", 64),
    (-1, "Conv", 128, 3, 2, 1),
    (-1, "Bottleneck", 128),
    (-1, "Bottleneck", 128),
    (-1, "Conv", 256, 3, 2, 1),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256),
    (-1, "Bottleneck", 256), # 14
    (-1, "Conv", 512, 3, 2, 1),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512),
    (-1, "Bottleneck", 512), # 23
    (-1, "Conv", 1024, 3, 2, 1),
    (-1, "Bottleneck", 1024),
    (-1, "Bottleneck", 1024),
    (-1, "Bottleneck", 1024),
    (-1, "Bottleneck", 1024),
    # Head
    (-1, "Conv", 512, 1, 1, 0),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "Conv", 512, 1, 1, 0),
    (-1, "Conv", 1024, 3, 1, 1),
    (-1, "Conv", 512, 1, 1, 0),
    (-1, "Conv", 1024, 3, 1, 1), # 34
    (-1, "YOLOv3Detect", 3, 20),

    (34, "Conv", 256, 1, 1, 0),
    (-1, "Upsample", 2),
    ([-1, 23], "Concat"),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1),
    (-1, "Conv", 256, 1, 1, 0),
    (-1, "Conv", 512, 3, 1, 1), # 44
    (-1, "YOLOv3Detect", 3, 20),

    (44, "Conv", 128, 1, 1, 0),
    (-1, "Upsample", 2),
    ([-1, 14], "Concat"),
    (-1, "Conv", 128, 1, 1, 0),
    (-1, "Conv", 256, 3, 1, 1),
    (-1, "Conv", 128, 1, 1, 0),
    (-1, "Conv", 256, 3, 1, 1),
    (-1, "Conv", 128, 1, 1, 0),
    (-1, "Conv", 256, 3, 1, 1),
    (-1, "YOLOv3Detect", 3, 20),
]

yolov4_config = [
    # Backbone (CSPDarknet53)
    (-1, "ConvMish", 32, 3, 1, 1),
    (-1, "ConvMish", 64, 3, 2, 1),
    (-1, "CSPBlock", 64, 1),
    (-1, "ConvMish", 128, 3, 2, 1),
    (-1, "CSPBlock", 128, 2),
    (-1, "ConvMish", 256, 3, 2, 1),
    (-1, "CSPBlock", 256, 8),
    (-1, "ConvMish", 512, 3, 2, 1),
    (-1, "CSPBlock", 512, 8),
    (-1, "ConvMish", 1024, 3, 2, 1),
    (-1, "CSPBlock", 1024, 4),
    # Neck
    (-1, "SPP", 512),
    (-1, "ConvMish", 512, 1, 1),
    (-1, "ConvMish", 1024, 3, 1, 1),
    (-1, "ConvMish", 512, 1, 1),
    # PAN
    (-1, "ConvMish", 256, 1, 1),
    (-1, "Upsample", 2),
    ([-1, 8], "Concat"),
    (-1, "CSPBlock", 256, 2),
    (-1, "ConvMish", 128, 1, 1),
    (-1, "Upsample", 2),
    ([-1, 6], "Concat"),
    (-1, "CSPBlock", 128, 2),
    # Head
    (-1, "ConvMish", 256, 3, 1, 1),
    (-1, "YOLOv4Detect", 3, 20),
    (22, "ConvMish", 256, 3, 2, 1),
    ([-1, 19], "Concat"),
    (-1, "CSPBlock", 256, 2),
    (-1, "ConvMish", 512, 3, 1, 1),
    (-1, "YOLOv4Detect", 3, 20),
    (28, "ConvMish", 512, 3, 2, 1),
    ([-1, 14], "Concat"),
    (-1, "CSPBlock", 512, 2),
    (-1, "ConvMish", 1024, 3, 1, 1),
    (-1, "YOLOv4Detect", 3, 20),
]

class Conv(nn.Module):
    """
    CNN block for YOLOv1 and YOLOv3
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0, bias: bool = False):
        """
        Constructor for CNNBlock

        param: in_channels (int) - number of input channels
        param: out_channels (int) - number of output channels
        param: kernel_size (int) - size of kernel
        param: stride (int) - stride of kernel
        param: padding (int) - padding of kernel
        param: bias (bool) - whether to use bias or not
        """
        super(Conv, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNNBlock

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.conv(x)
        if not self.bias:
            x = self.batchnorm(x)
            x = self.leakyrelu(x)
        return x
    
class Bottleneck(nn.Module):
    """
    Bottleneck block for YOLOv3
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Constructor for Bottleneck

        param: in_channels (int) - number of input channels
        param: out_channels (int) - number of output channels
        param: kernel_size (int) - size of kernel
        param: stride (int) - stride of kernel
        param: padding (int) - padding of kernel
        """
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cv1 = Conv(in_channels, out_channels // 2, 1, 1, 0)
        self.cv2 = Conv(out_channels // 2, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Bottleneck

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        if self.in_channels == self.out_channels:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))
    
class Concat(nn.Module):
    """
    Concatenate block for YOLOv3
    """
    def __init__(self):
        """
        Constructor for Concat
        """
        super(Concat, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Concat

        param: x (torch.Tensor) - input tensor
        param: y (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        return torch.cat((x, y), dim=1)
    
class Mish(nn.Module):
    """
    Mish activation function

    Mish(x) = x * tanh(softplus(x))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Mish

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        return x * torch.tanh(nn.functional.softplus(x))

class ConvMish(nn.Module):
    """
    Convolutional block with Mish activation function
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0, bias: bool = False):
        """
        Constructor for ConvMish

        param: in_channels (int) - number of input channels
        param: out_channels (int) - number of output channels
        param: kernel_size (int) - size of kernel
        param: stride (int) - stride of kernel
        param: padding (int) - padding of kernel
        param: bias (bool) - whether to use bias or not
        """
        super(ConvMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ConvMish

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        return self.mish(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """
    CSP block for YOLOv4
    """
    def __init__(self, in_channels: int, out_channels: int, num_bottlenecks: int):
        """
        Constructor for CSPBlock

        param: in_channels (int) - number of input channels
        param: out_channels (int) - number of output channels
        param: num_bottlenecks (int) - number of bottlenecks
        """
        super(CSPBlock, self).__init__()
        self.conv1 = ConvMish(in_channels, out_channels // 2, 1, 1)
        self.conv2 = ConvMish(in_channels, out_channels // 2, 1, 1)
        self.conv3 = ConvMish(out_channels, out_channels, 1, 1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(out_channels // 2, out_channels // 2) for _ in range(num_bottlenecks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CSPBlock

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.bottlenecks(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class SPP(nn.Module):
    """
    Spatial Pyramid Pooling block for YOLOv4
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Constructor for SPP

        param: in_channels (int) - number of input channels
        param: out_channels (int) - number of output channels
        """
        super(SPP, self).__init__()
        self.conv1 = ConvMish(in_channels, in_channels // 2, 1, 1)
        self.conv2 = ConvMish(in_channels * 2, out_channels, 1, 1)
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=9, stride=1, padding=4),
            nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SPP

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.conv1(x)
        pooled = [mp(x) for mp in self.maxpools]
        x = torch.cat([x] + pooled, dim=1)
        return self.conv2(x)

class YOLOv4Detect(nn.Module):
    """
    YOLOv4 detection head
    """
    def __init__(self, in_channels: int, num_anchors: int = 3, num_classes: int = 80):
        """
        Constructor for YOLOv4Detect
        
        param: in_channels (int) - number of input channels
        param: num_anchors (int) - number of anchors
        param: num_classes (int) - number of classes
        """
        super(YOLOv4Detect, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for YOLOv4Detect

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.num_anchors, 5 + self.num_classes, x.shape[2], x.shape[3])
        return x.permute(0, 1, 3, 4, 2)
        
class YOLOv3Detect(nn.Module):
    """
    YOLOv3 detection head
    """
    def __init__(self, in_channels: int, num_anchors: int = 3, num_classes: int = 20):
        """
        Constructor for YOLOv3Detect

        param: in_channels (int) - number of input channels
        param: num_classes (int) - number of classes
        """
        super(YOLOv3Detect, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.block1 = Conv(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = Conv(2 * in_channels, num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for YOLOv3Detect

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.shape[0], self.num_anchors, 5 + self.num_classes, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 3, 4, 2)
        return x # (batch_size, grid_size, grid_size, num_boxes, 5 + num_classes)
    
class YOLOv1Detect(nn.Module):
    """
    YOLOv1 detection head
    """
    def __init__(self, in_channels: int, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20, linear_size: int = 1024):
        """
        Constructor for YOLOv1Detect

        param: in_channels (int) - number of input channels
        param: split_size (int) - split size of image
        param: num_boxes (int) - number of boxes per cell
        param: num_classes (int) - number of classes
        param: linear_size (int) - size of linear layer
        """
        super(YOLOv1Detect, self).__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * split_size * split_size, linear_size),
            nn.LeakyReLU(0.1),
            nn.Linear(linear_size, split_size * split_size * (num_classes + num_boxes * 5)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for YOLOv1Detect

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor - output tensor
        """
        x = self.seq(x)
        return x
    
class YOLO(nn.Module):
    def __init__(self, config: list[tuple] = yolov1_config, in_channels: int = 3):
        """
        Constructor for YOLO

        param: config (list[tuple]) - configuration for YOLO model
        param: in_channels (int) - number of input channels
        """
        super(YOLO, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        self.layers_to_save = set()
        self._create_layers()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """
        Forward pass for YOLO

        param: x (torch.Tensor) - input tensor
        return: torch.Tensor | list[torch.Tensor] - output tensor
        """
        outputs = []
        layer_outputs = {}
        
        for idx, (from_layer, module_type, *args) in enumerate(self.config):
            if isinstance(from_layer, list):
                x = [layer_outputs[i] if i != -1 else x for i in from_layer]
                x = self.layers[idx](*x)
            elif from_layer != -1:
                x = layer_outputs[from_layer]
                x = self.layers[idx](x)
            else:
                x = self.layers[idx](x)
            
            if idx in self.layers_to_save:
                layer_outputs[idx] = x
            
            if module_type in ["YOLOv1Detect", "YOLOv3Detect", "YOLOv4Detect"]:
                outputs.append(x)
        
        return outputs if len(outputs) > 1 else outputs[0]

    def _create_layers(self):
        """
        Create layers for YOLO model
        """
        channels = []

        for idx, (from_layer, module_type, *args) in enumerate(self.config):
            if idx == 0:
                in_channels = self.in_channels
            elif isinstance(from_layer, list):
                in_channels = sum(channels[i] for i in from_layer)
                for i in from_layer:
                    if i != -1:
                        self.layers_to_save.add(i)
            elif from_layer != -1:
                in_channels = channels[from_layer]
                self.layers_to_save.add(from_layer)
            else:
                in_channels = channels[-1]

            if module_type == "Conv":
                self.layers.append(Conv(in_channels, *args))
                channels.append(args[0])
            elif module_type == "ConvMish":
                self.layers.append(ConvMish(in_channels, *args))
                channels.append(args[0])
            elif module_type == "MaxPool2d":
                self.layers.append(nn.MaxPool2d(*args))
                channels.append(in_channels)
            elif module_type == "Bottleneck":
                self.layers.append(Bottleneck(in_channels, *args))
                channels.append(args[0])
            elif module_type == "CSPBlock":
                self.layers.append(CSPBlock(in_channels, *args))
                channels.append(args[0])
            elif module_type == "SPP":
                self.layers.append(SPP(in_channels, *args))
                channels.append(args[0])
            elif module_type == "YOLOv1Detect":
                self.layers.append(YOLOv1Detect(in_channels, *args))
                channels.append(-1)
            elif module_type == "YOLOv3Detect":
                self.layers.append(YOLOv3Detect(in_channels, *args))
                channels.append(-1)
            elif module_type == "YOLOv4Detect":
                self.layers.append(YOLOv4Detect(in_channels, *args))
                channels.append(-1)
            elif module_type == "Upsample":
                self.layers.append(nn.Upsample(scale_factor=args[0]))
                channels.append(in_channels)
            elif module_type == "Concat":
                self.layers.append(Concat())
                channels.append(in_channels)
            else:
                raise ValueError(f"Invalid module type: {module_type}")
            
            print(f"Layer {idx}: {module_type} with {in_channels} input channels and {channels[-1]} output channels")
        
        print(f"Layers to save: {self.layers_to_save}")

if __name__ == "__main__":
    model = YOLO(yolov4_config)
    x = torch.randn(1, 3, 416, 416)
    pred = model(x)
    print(pred[0].shape)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
