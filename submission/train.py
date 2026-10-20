"""
EcoSort Challenge - Training script (lightweight CNN, 6-class classification)

Usage:
    python train.py --epochs 60 --batch_size 32 --lr 0.003
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 96


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 conv + pointwise 1x1 conv, each followed by BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class EcoSortNet(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        # Stem: 96x96 -> 48x48
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # (in_channels, out_channels, stride)
        cfg = [
            (32, 64, 1),     # 48x48
            (64, 96, 2),     # 24x24
            (96, 96, 1),
            (96, 160, 2),    # 12x12
            (160, 160, 1),
            (160, 256, 2),   # 6x6
            (256, 256, 1),
        ]
        self.blocks = nn.Sequential(
            *[DepthwiseSeparableBlock(i, o, s) for i, o, s in cfg]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 2):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        raise FileNotFoundError(f"Dataset not found in '{data_dir}'. Run download_data.py first.")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.15),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    if train_ds.classes != val_ds.classes:
        raise ValueError(f"Class mismatch: train {train_ds.classes} vs val {val_ds.classes}")
    print("Class order:", train_ds.classes)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def export_to_onnx(model: nn.Module, output_path: str):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    kwargs = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    try:
        torch.onnx.export(model, dummy, output_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, output_path, **kwargs)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported '{output_path}' ({size_mb:.2f} MB, limit 1.50 MB)")


def main():
    parser = argparse.ArgumentParser(description="EcoSort training")
    parser.add_argument("--data_dir", type=str, default="data/public")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    num_classes = len(train_loader.dataset.classes)
    model = EcoSortNet(num_classes=num_classes).to(device)
    print(f"Parameters: {count_parameters(model):,} / 350,000 max")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.2,
    )

    best_acc, best_state = 0.0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] "
              f"Train Loss {tr_loss:.3f} Acc {tr_acc:5.1f}% | "
              f"Val Loss {va_loss:.3f} Acc {va_acc:5.1f}% | Best {best_acc:5.1f}%")

    print(f"Training finished in {(time.time() - start) / 60:.1f} min. Best val acc: {best_acc:.2f}%")

    model.load_state_dict(best_state)
    export_to_onnx(model.cpu(), args.output)
    print(f"\nNext: python verify_submission.py --model {args.output} --val_dir {args.data_dir}/val")


if __name__ == "__main__":
    main()
