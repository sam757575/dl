"""
CNN for Image Detection and Recognition
Based on: "Convolutional Neural Network (CNN) for Image Detection and Recognition"
          (ResearchGate publication/332826568)

Architecture:
  - 4 convolutional blocks (Conv -> BN -> ReLU -> MaxPool)
  - Global Average Pooling
  - Fully-connected classifier head
  - Trained on CIFAR-10 (10 classes, 32x32 RGB images)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time


# ─── Model ────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> (optional) MaxPool"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1,
                 pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """
    4-block CNN for image classification.

    Input : (B, 3, H, W)  — tested on CIFAR-10 (32x32)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        # Feature extractor — spatial dims halved at each pooled block
        #   32 -> 16 -> 8 -> 4 -> (GAP -> 1)
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            ConvBlock(3,   32,  pool=True),
            # Block 2: 32 -> 64  (extra conv before pool for richer features)
            ConvBlock(32,  64,  pool=False),
            ConvBlock(64,  64,  pool=True),
            # Block 3: 64 -> 128
            ConvBlock(64,  128, pool=False),
            ConvBlock(128, 128, pool=True),
            # Block 4: 128 -> 256
            ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=True),
        )

        # Global Average Pooling collapses spatial dims -> (B, 256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ─── Data ─────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_dataloaders(data_dir: str = "./data",
                    batch_size: int = 128,
                    num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10."""

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


def train(num_epochs: int = 50,
          batch_size: int = 128,
          lr: float = 0.1,
          weight_decay: float = 5e-4,
          data_dir: str = "./data",
          save_path: str = "best_cnn.pth"):
    """Full training run with cosine LR decay and best-model checkpointing."""

    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device : {device}")

    train_loader, test_loader = get_dataloaders(data_dir, batch_size)

    model     = CNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Test Loss':>10}  {'Test Acc':>10}  {'LR':>8}  {'Time':>6}")
    print("-" * 75)

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(
            model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  "
              f"{test_loss:>10.4f}  {test_acc:>9.2f}%  {cur_lr:>8.5f}  "
              f"{elapsed:>5.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_acc": best_acc}, save_path)
            print(f"         ↑ New best: {best_acc:.2f}%  (saved → {save_path})")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    return model


# ─── Per-class accuracy ───────────────────────────────────────────────────────

@torch.no_grad()
def per_class_accuracy(model: nn.Module,
                       loader: DataLoader,
                       device: torch.device) -> dict[str, float]:
    model.eval()
    class_correct = [0] * len(CLASSES)
    class_total   = [0] * len(CLASSES)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = model(images).max(1)
        for label, pred in zip(labels, predicted):
            c = label.item()
            class_correct[c] += int(pred == label)
            class_total[c]   += 1

    return {cls: 100.0 * class_correct[i] / class_total[i]
            for i, cls in enumerate(CLASSES)}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CNN for Image Detection and Recognition (CIFAR-10)")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--weight-decay",type=float, default=5e-4)
    parser.add_argument("--data-dir",    type=str,   default="./data")
    parser.add_argument("--save-path",   type=str,   default="best_cnn.pth")
    args = parser.parse_args()

    trained_model = train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_dir=args.data_dir,
        save_path=args.save_path,
    )

    # ── Per-class breakdown ──────────────────────────────────────────────────
    device = next(trained_model.parameters()).device
    _, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    print("\nPer-class accuracy on test set:")
    print("-" * 30)
    for cls, acc in per_class_accuracy(trained_model, test_loader, device).items():
        print(f"  {cls:<12} {acc:>6.2f}%")
