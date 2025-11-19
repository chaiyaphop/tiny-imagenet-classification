import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import json

from model import ModifiedResNet18
from dataset import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(checkpoint_path, data_dir, batch_size):
    # Load the Data
    mean = [0.480, 0.448, 0.398]
    std = [0.276, 0.268, 0.281]

    print(f"Loading validation data from: {data_dir}")
    _, val_loader = get_dataloaders(data_dir, batch_size, mean, std)

    # Initialize Model Architecture
    print("Initializing model...")
    model = ModifiedResNet18(num_classes=200).to(device)

    # Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle cases where the model was saved with DataParallel (keys start with 'module.')
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Run Evaluation
    correct_1 = 0
    correct_5 = 0
    total = 0

    # Optional: Track per-class accuracy
    class_correct = list(0. for i in range(200))
    class_total = list(0. for i in range(200))

    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Top-1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_1 += predicted.eq(labels).sum().item()

            # Top-5
            _, top5 = outputs.topk(5, 1, largest=True, sorted=True)
            correct_5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

            # Per-class stats
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate Final Metrics
    acc_1 = 100. * correct_1 / total
    acc_5 = 100. * correct_5 / total

    results = {
        "Top-1 Accuracy": f"{acc_1:.2f}%",
        "Top-5 Accuracy": f"{acc_5:.2f}%",
        "Total Images Evaluated": total
    }

    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT")
    print("="*30)
    print(json.dumps(results, indent=4))
    print("="*30)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth', help='Path to .pth file')
    parser.add_argument('--data_dir', type=str, default='../data/tiny-imagenet-200', help='Path to dataset')
    parser.add_argument('--batch', type=int, default=128)

    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_dir, args.batch)