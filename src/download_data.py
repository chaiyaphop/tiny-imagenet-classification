import os
from datasets import load_dataset
from tqdm import tqdm


def save_split(dataset, split_name, output_dir):
    print(f"Processing {split_name} split...")

    if hasattr(dataset.features['label'], 'names'):
        idx_to_class = dataset.features['label'].names
    else:
        idx_to_class = {i: str(i) for i in set(dataset['label'])}

    for i, sample in enumerate(tqdm(dataset)):
        image = sample['image']
        label_idx = sample['label']
        class_name = idx_to_class[label_idx]

        class_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)

        filename = f"{class_name}_{i}.jpg"
        image.convert("RGB").save(os.path.join(class_dir, filename))


def main(output_dir):
    dataset = load_dataset("zh-plus/tiny-imagenet")
    os.makedirs(output_dir, exist_ok=True)

    if 'train' in dataset:
        save_split(dataset['train'], 'train', output_dir)

    if 'valid' in dataset:
        save_split(dataset['valid'], 'val', output_dir)

    print(f"\nDataset successfully saved to: {os.path.abspath(output_dir)}")
    print(f"Structure:\n  {output_dir}/train/class_id/xxx.jpg\n  {output_dir}/val/class_id/xxx.jpg")


if __name__ == "__main__":
    OUTPUT_DIR = "../data/tiny-imagenet-200"
    main(OUTPUT_DIR)
