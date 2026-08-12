# Cat vs Dog Image Classification with AlexNet

A PyTorch-based computer vision project for binary Cat vs Dog image classification using AlexNet Transfer Learning.

## Overview

The objective of this project is to develop a deep learning model capable of distinguishing between Cat and Dog images.

The project uses a pretrained AlexNet CNN and adapts its final classifier for two classes:

- 0 — Cat
- 1 — Dog

The pretrained feature extraction layers are frozen, while the classifier is trained specifically for the Cat vs Dog classification task.

## Project Workflow

Dataset
↓
Dataset Inspection
↓
Class Distribution Analysis
↓
Image Visualization
↓
Corrupted Image Check
↓
Severe Blur Check
↓
Data Augmentation
↓
Normalization
↓
Train / Validation / Test Split
↓
DataLoader
↓
Pretrained AlexNet
↓
Transfer Learning
↓
Model Training
↓
Validation
↓
Test Evaluation
↓
Confusion Matrix
↓
ROC / AUC
↓
Precision-Recall Analysis

## Dataset

The dataset follows the ImageFolder directory structure:

PetImages/
├── Cat/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── Dog/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...

### Dataset Statistics

| Category | Distribution |
|---|---:|
| Cat | 49.6% |
| Dog | 50.4% |
| Total Images | 24,542 |

The dataset is approximately balanced, reducing the risk of strong class imbalance.

## Data Quality Checks

Before model training, the dataset was inspected for common image-quality issues.

### Corrupted Image Detection

Each image was opened and verified using PIL:

img = Image.open(img_path)
img.verify()

Images that could not be verified were recorded as corrupted.

### Severe Blur Detection

OpenCV was used to detect severely blurred images using the variance of the Laplacian:

blur_value = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

A very low value indicates a severely blurred image.

The dataset produced:

Severely Blurred Images: 0

## Data Preprocessing

All images are resized to:

224 × 224 pixels

This matches the input size used for the AlexNet model.

### Training Transformations

The training pipeline applies:

- Random Rotation
- Random Horizontal Flip
- Random Translation
- Random Scaling
- Brightness Adjustment
- Tensor Conversion
- Normalization

Example:

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

### Validation and Test Transformations

Validation and test images only undergo:

1. Resizing
2. Tensor conversion
3. Normalization

Random augmentation is not applied to validation or test data.

## Dataset Split

The dataset is divided into three subsets:

| Dataset | Percentage | Images |
|---|---:|---:|
| Training | 70% | 17,179 |
| Validation | 15% | 3,681 |
| Testing | 15% | 3,682 |
| Total | 100% | 24,542 |

A fixed random seed is used to make the split reproducible:

torch.Generator().manual_seed(42)

## DataLoader

PyTorch DataLoader is used to efficiently load images in batches.

### Configuration

Batch Size: 128
Training Batches: 135
Validation Batches: 29
Test Batches: 29

Training data is shuffled, while validation and test data are not shuffled.

A sample training batch has the following shape:

Image Shape: [128, 3, 224, 224]
Label Shape: [128]

Where:

- 128 = batch size
- 3 = RGB channels
- 224 × 224 = image dimensions

# Model Architecture

## AlexNet Transfer Learning

The project uses a pretrained AlexNet model from Torchvision:

model = models.alexnet(
    weights=models.AlexNet_Weights.DEFAULT
)

The pretrained model provides previously learned visual features.

### Freezing the Feature Extractor

The convolutional feature extractor is frozen:

for param in model.features.parameters():
    param.requires_grad = False

This prevents the pretrained feature-extraction weights from being updated during training.

### Modifying the Final Classifier

Since the original AlexNet is designed for a larger number of classes, its final classifier layer is replaced with a two-output layer:

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    2
)

The outputs represent:

0 → Cat
1 → Dog

## Learning Approach

The project uses:

Supervised Deep Learning → CNN → Transfer Learning → AlexNet

The dataset contains labeled examples, allowing the model to learn the relationship between image features and their corresponding classes.

## Activation Function

The pretrained AlexNet architecture contains ReLU (Rectified Linear Unit) activation functions within its feature extraction layers.

ReLU introduces non-linearity and allows the network to learn complex visual patterns.

The ReLU layers are already part of the pretrained AlexNet architecture and therefore do not need to be manually defined in the project code.

# Training Configuration

| Parameter | Configuration |
|---|---|
| Model | AlexNet |
| Learning Type | Supervised Deep Learning |
| Technique | Transfer Learning |
| Task | Binary Classification |
| Classes | Cat, Dog |
| Input Size | 224 × 224 |
| Batch Size | 128 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | CrossEntropyLoss |
| Trainable Parameters | Classifier |

### Loss Function

criterion = nn.CrossEntropyLoss()

### Optimizer

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.0001
)

Only the classifier parameters are passed to the optimizer because the pretrained feature extractor is frozen.

# Training Process

For every training batch, the following process is performed:

Input Images
↓
AlexNet
↓
Class Scores
↓
Calculate Loss
↓
Backpropagation
↓
Update Classifier Weights

The model is trained for 10 epochs.

After each epoch, validation accuracy is calculated using the validation dataset.

# Model Evaluation

The trained model is evaluated on the held-out test set using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC
- Precision-Recall Curve

# Results

The final test-set results are:

| Metric | Score |
|---|---:|
| Accuracy | 96.33% |
| Precision | 96.88% |
| Recall | 96.02% |
| F1 Score | 96.45% |
| AUC | 99.49% |

The results demonstrate strong classification performance on the held-out test dataset.

# Confusion Matrix

The final confusion matrix is:

                    Predicted
                   Cat      Dog

Actual Cat        1714      59
Actual Dog          76     1833

### Interpretation

- 1,714 Cat images were correctly classified as Cat.
- 1,833 Dog images were correctly classified as Dog.
- 59 Cat images were incorrectly classified as Dog.
- 76 Dog images were incorrectly classified as Cat.

### Correct Predictions

1714 + 1833 = 3547

### Misclassifications

59 + 76 = 135

The confusion matrix is consistent with the reported test accuracy of approximately 96.33%.

# ROC Curve and AUC

The ROC curve evaluates how effectively the model separates the two classes across different classification thresholds.

### AUC Results

Dog AUC: 0.9949
Cat AUC: 0.9949

An AUC value close to 1.0 indicates strong class discrimination.

The results show that the model has excellent ability to distinguish between Cat and Dog images.

# Precision-Recall Curve

The Precision-Recall curve evaluates the relationship between precision and recall across different classification thresholds.

Separate curves are generated for Cat and Dog.

The curves remain close to the upper-right region, indicating strong precision and recall performance.

# Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- PIL
- Matplotlib
- Scikit-learn
- KaggleHub

# Installation

Install the required Python packages:

pip install torch torchvision
pip install opencv-python pillow matplotlib scikit-learn kagglehub

# How to Run

1. Download or connect the Cat vs Dog dataset.
2. Ensure the dataset follows the required PetImages/Cat and PetImages/Dog structure.
3. Run the notebook cells sequentially.
4. Perform dataset inspection and class distribution analysis.
5. Check for corrupted and severely blurred images.
6. Apply preprocessing and data augmentation.
7. Create the training, validation, and test datasets.
8. Create the PyTorch DataLoaders.
9. Load the pretrained AlexNet model.
10. Freeze the feature extractor.
11. Replace the final classifier with a two-class output layer.
12. Train the model for 10 epochs.
13. Evaluate the model on the test dataset.
14. Analyze the confusion matrix, ROC/AUC, and Precision-Recall curves.

# Recommended Repository Structure

cat-dog-alexnet/
│
├── README.md
│
├── notebooks/
│   └── cat_dog_alexnet.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
└── requirements.txt

The dataset itself does not need to be uploaded to GitHub.

# Key Findings

The project demonstrates that pretrained CNN features can be effectively adapted to a binary image classification task using transfer learning.

Key results:

Accuracy  → 96.33%
Precision → 96.88%
Recall    → 96.02%
F1 Score → 96.45%
AUC       → 99.49%

The model achieved strong performance while training only the final classifier on top of the frozen pretrained AlexNet feature extractor.

# Author

Mohammed Naeem

# Project Information

| Category | Details |
|---|---|
| Domain | Computer Vision |
| Task | Binary Image Classification |
| Framework | PyTorch |
| Architecture | AlexNet |
| Technique | Transfer Learning |
| Learning Type | Supervised Deep Learning |
| Classes | Cat and Dog |
