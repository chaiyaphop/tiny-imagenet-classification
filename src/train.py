import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from model import ModifiedResNet18

from model import ModifiedResNet18
from dataset import get_dataloaders


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True) 

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())

    acc = 100. * correct / total
    return running_loss / len(loader), acc


def validate(model, loader, criterion, epoch):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    correct_5 = 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Top-1 Accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 Accuracy
            _, top5 = outputs.topk(5, 1, largest=True, sorted=True)
            correct_5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

    acc = 100. * correct / total
    acc_5 = 100. * correct_5 / total
    return running_loss / len(loader), acc, acc_5


def run_training(data_dir, batch_size, num_epochs):

    # Define Normalization Stats
    mean = [0.480, 0.448, 0.398]
    std = [0.276, 0.268, 0.281]

    # Load Data
    print(f"Loading data from: {data_dir}")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size, mean, std)

    # Setup Model
    model = ModifiedResNet18(num_classes=200).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    os.makedirs("../checkpoints", exist_ok=True)

    print(f"Training on: {torch.cuda.get_device_name(0)}")
    print("-" * 85)
    print(f"{'Epoch':<10} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Top-5 Acc':<10}")
    print("-" * 85)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, epoch)

        scheduler.step()

        print(f"{epoch:<10} | {train_loss:.4f}     | {train_acc:.2f}%      | {val_loss:.4f}     | {val_acc:.2f}%      | {val_top5:.2f}%")

        # Save Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': best_acc,
                },
                "../checkpoints/best_model.pth"
            )

    print(f"\nTraining Complete. Best Top-1 Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train Modified ResNet on Tiny ImageNet")
    parser.add_argument('--data_dir', type=str, default='../data/tiny-imagenet-200', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=3072, help='Batch size')

    args = parser.parse_args()

    run_training(args.data_dir, args.batch, args.epochs)
