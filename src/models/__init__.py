"""CPU-feasible CNN architectures for CIFAR-10/100 (32x32 RGB input).

A  simplecnn  the historical SimpleCNN, module-for-module identical to
              evaluation/common.py, so cnn_baseline_FIXED.pth loads unchanged
B  lenet5     LeNet-5 topology adapted to 3-channel 32x32 input: two 5x5 valid
              convolutions (6, 16 channels) each followed by 2x2 max-pooling,
              then 400-120-84-C fully connected; ReLU activations
C  resnet8    CIFAR ResNet of He et al. (2016) with n = 1 (6n + 2 = 8 weight
              layers): 3x3 stem (16), three stages of one BasicBlock each
              (16, 32, 64 channels; strides 1, 2, 2), 1x1 projection shortcuts
              where the shape changes, BatchNorm, global average pooling, FC
C' smallvgg   the pre-registered fallback for C (used only if resnet8 exceeds the
              CPU budget in results/cpu_runtime_budget.md): three VGG stages of
              two 3x3 conv + BN + ReLU (32, 64, 128) with 2x2 max-pooling, then
              2048-256-C fully connected

All four take `num_classes` and are fully deterministic given the global torch
seed at construction time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Identical to evaluation/common.py SimpleCNN (same module names and order)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),     # 32 -> 28 -> 14
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2))    # 14 -> 10 -> 5
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet8(nn.Module):
    def __init__(self, num_classes=10, widths=(16, 32, 64)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = BasicBlock(widths[0], widths[0], 1)
        self.layer2 = BasicBlock(widths[0], widths[1], 2)
        self.layer3 = BasicBlock(widths[1], widths[2], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.pool(x), 1))


class SmallVGG(nn.Module):
    def __init__(self, num_classes=10, widths=(32, 64, 128)):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
                       nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(widths[-1] * 4 * 4, 256), nn.ReLU(),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


ARCHITECTURES = {"simplecnn": SimpleCNN, "lenet5": LeNet5, "resnet8": ResNet8, "smallvgg": SmallVGG}
PRIMARY = ["simplecnn", "lenet5", "resnet8"]
FALLBACK = {"resnet8": "smallvgg"}


def build_model(arch, num_classes=10, seed=None):
    """Construct an architecture; if `seed` is given, the initial weights are a function of it alone."""
    if seed is not None:
        torch.manual_seed(seed)
    return ARCHITECTURES[arch](num_classes=num_classes)


def load_checkpoint(path, arch, num_classes=10):
    """Load a plain state dict or a Phase-2 checkpoint dict ({"state_dict": ...}); returns an eval() model."""
    model = build_model(arch, num_classes)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(obj["state_dict"] if "state_dict" in obj else obj)
    model.eval()
    return model


def count_parameters(model):
    """Totals by kind. 'weights' counts the Conv2d/Linear weight tensors (the pruning denominator)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    bias = sum(m.bias.numel() for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None)
    bn = sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    assert conv + linear + bias + bn == total, "unaccounted parameters"
    return {"total": total, "trainable": trainable, "conv_weights": conv, "linear_weights": linear,
            "weights": conv + linear, "biases": bias, "batchnorm": bn, "buffers": buffers}
