#!/usr/bin/env python3
"""
Train the character classifier on the synthetic font dataset.

Run generate_font_dataset.py first to create the training data.

Usage:
  python tools/train_classifier.py
  python tools/train_classifier.py --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.char_classifier import CharClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data/char_dataset")
    p.add_argument("--checkpoint", default="data/char_classifier.pth")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val-split",  type=float, default=0.1)
    p.add_argument("--device",     default="")
    return p.parse_args()


def resolve_device(req: str) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    transform = T.Compose([
        T.Grayscale(1),
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
    ])

    dataset = ImageFolder(args.data_dir, transform=transform)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = CharClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += len(labels)
        train_acc = correct / total

        # --- validate ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += len(labels)
        val_acc = correct / total

        sched.step()
        print(f"Epoch {epoch+1:3d}  train={train_acc:.3f}  val={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model ({val_acc:.3f}) → {ckpt_path}")

    print(f"\nBest val acc: {best_val_acc:.3f}   Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
