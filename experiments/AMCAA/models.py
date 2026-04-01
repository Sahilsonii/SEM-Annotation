import torch
import torch.nn as nn
import torchvision.models as tv_models


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders   = nn.ModuleList()
        self.decoders   = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2)
        self.ups        = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x    = up(x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final(x)


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats   = base.fc.in_features
        base.fc    = nn.Linear(in_feats, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base    = tv_models.efficientnet_b0(weights=weights)
        base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_feats = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_feats, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


def build_model(name: str, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    registry = {
        "unet":          lambda: UNet(in_channels=1, out_channels=1),
        "resnet":        lambda: ResNetClassifier(num_classes, pretrained),
        "efficientnet":  lambda: EfficientNetClassifier(num_classes, pretrained),
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Choose from {list(registry)}")
    return registry[name]()
