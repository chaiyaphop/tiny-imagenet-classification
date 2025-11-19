# Tiny ImageNet Classification

## 📁 Project Structure

```text
.
├── src/
│   ├── dataset.py       # Custom Dataset class with Albumentations pipeline
│   ├── model.py         # Modified ResNet-18 architecture (optimized for 64x64)
│   ├── train.py         # Training loop with AMP, Logging, and Checkpointing
│   └── utils.py         # Helper functions for metrics and logging
├── notebooks/
│   └── eda.ipynb        # Exploratory Data Analysis & actionable insights
├── diagrams/
│   └── concept_flow.png # System Design Diagram for Task #2
├── checkpoints/         # Saved model weights
├── requirements.txt     # Reproducible python environment
└── README.md            # Documentation and approach
```

-----

## Image Classification (Tiny ImageNet)

### 1. Data Processing & EDA

**Findings:**

- **Resolution Constraint:** Since the images are only **64×64 pixels**, many standard image-classification models (like ResNet-50) aren’t ideal because they downsample too aggressively. Their early layers (e.g., 7×7 conv with stride 2 followed by max-pooling) shrink the feature maps so much that spatial details are almost completely lost.
- **Normalization:** Instead of using the default ImageNet statistics, I computed the dataset’s own normalization values.
    - **Mean:** `[0.4802, 0.4480, 0.3980]`
    - **Std:** `[0.2756, 0.2684, 0.2805]`
- **Class Balance:** Confirmed that the dataset is perfectly balanced, with 500 images in each class.

**Approach & Mitigation:**

- **Pipeline:** The entire preprocessing and augmentation pipeline was built with **Albumentations**, which offers both high speed and a lot of flexibility.
- **Regularization (Preventing Overfitting):** Because the model is being trained from scratch on a relatively small dataset, overfitting is the main concern.
    - **CoarseDropout (Cutout):** Randomly removes 16×16 patches from the image, encouraging the model to learn more robust, distributed features instead of fixating on specific regions.
    - **Augmentation:** Applied strong geometric transformations—shifts, scaling, rotations, and flips—to increase data variety and improve generalization.

### 2. Model Architecture: Modified ResNet-18

**Rationale:**
A standard ResNet-18 doesn’t work well for 64×64 images, so I adapted the architecture to better preserve spatial detail.

1. **Stem Modification:** Replaced the original **7×7 conv (stride 2)** with a smaller **3×3 conv (stride 1)**.
2. **Pooling Removal:** Removed the initial **MaxPool2d** layer entirely.
3. **Result:** These changes keep the feature maps at the full 64×64 resolution as they enter the residual blocks (instead of shrinking to 16×16 in the standard version), allowing the model to retain the fine-grained information needed to classify all 200 classes.

### 3. Training Strategy

- **Optimization:** Used the ***AdamW*** optimizer paired with a ***CosineAnnealingLR*** scheduler, which helps the model avoid getting stuck in local minima and promotes smoother, more stable convergence.
- **Loss Function:** Employed ***CrossEntropyLoss*** with **0.1 label smoothing** to reduce overconfidence, especially on noisy or ambiguous low-resolution inputs, leading to better generalization.
- **Hardware Acceleration:** Enabled **Automatic Mixed Precision (AMP)** on an NVIDIA A100, significantly boosting training speed while lowering memory usage.

### 4. Evaluation

- **Top-5 Accuracy:** With 200 classes that often overlap semantically (e.g., several dog breeds), Top-5 accuracy provides a more reliable sense of how well the model is capturing meaningful representations.

### 5. Results & Performance

The model was trained for 30 epochs on an NVIDIA A100 GPU. Updating the ResNet stem with a 3×3 convolution, along with using a `CosineAnnealingLR` scheduler, helped the model converge reliably.

#### Quantitative Metrics (Epoch 30)

| Metric             | Training (Final) | Validation (Best) |
| :----------------- | :--------------- | :---------------- |
| **Top-1 Accuracy** | **73.38%**       | **54.49%**        |
| **Top-5 Accuracy** | N/A              | **79.14%**        |
| **Loss**           | 1.9102           | 2.5675            |

> **Analysis:** The model reaches a **Top-5 Accuracy of about 79.1%**, which is the most important metric for this task. Considering the small input size (64×64) and the large number of classes (200), the model is able to include the correct label among its top five guesses nearly 80% of the time.

#### Training Dynamics

- **Generalization Gap:** As expected, there is a noticeable gap between training accuracy (73.38%) and validation accuracy (54.49%), suggesting the model has enough capacity to memorize parts of the training set.
- **Mitigation:** Using **Albumentations (CoarseDropout)** helped prevent this gap from growing even larger. Without these regularization techniques, training accuracy likely would have exceeded 90% while validation accuracy remained around 40% (***overfitting***).
- **Convergence:** Validation loss continued to improve even near the end of training (dropping from 1.9128 to 1.9102 around Epoch 30), showing that the learning rate scheduler was still guiding the optimization effectively. A few more epochs might have provided small additional gains.

#### Qualitative Conclusion

Despite the low image resolution, the model shows solid semantic understanding. Achieving a **Top-1 accuracy of 54.49%** surpasses what a standard ResNet typically achieves on this dataset (usually around 45–50% without adjustments). This supports the decision to modify the network to keep more spatial detail in the early layers.

-----

## OS & Dependencies

- **OS:** Linux (Ubuntu 20.04) / Google Colab Environment
- **Python:** 3.10
- **Key Libraries** (requirements.txt):
    - torch==2.9.1
    - torchvision==0.24.1
    - albumentations==2.0.8
    - opencv-python-headless==4.12.0.88
    - tqdm==4.67.1
    - matplotlib==3.10.7

## How to Run

1. **Create Conda Environment:**

    ```bash
    conda create -n venv python=3.10
    ```

2. **Activate Environment:**

    ```bash
    conda activate venv
    ```

3. **Install Requirements:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run Training:**

    ```bash
    python src/train.py --epochs 30 --batch_size 3072
    ```

5. **Checkpoints:**

    Model is saved to `checkpoints/`.
