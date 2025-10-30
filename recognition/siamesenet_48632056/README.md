ISIC 2020 Melanoma Classification using Siamese Network
========================================================

Project Overview
------------------
This project implements a Siamese Neural Network to classify melanoma vs normal skin lesions from the ISIC 2020 Kaggle Challenge dataset.
The model learns to map lesion images into a shared embedding space, where visually similar lesions (same diagnosis) lie close together. The goal is to achieve approximately 0.8 accuracy on unseen test data. This a challenging, real-world binary classification task in medical imaging.

The Algorithm
------------------------
A Siamese network consists of two identical convolutional branches that share weights. Each branch encodes an image into a 512-dimensional embedding vector using a pretrained EfficientNet-B0 backbone.
The model is trained using batch-hard triplet loss, which encourages embeddings of the same class (benign/malignant) to be close together while pushing apart embeddings of different classes.
After training, we compute class prototypes (average embeddings per class). During inference, each new image is embedded and classified based on its cosine similarity to the class prototypes.

The final architecture can be visualised as:
```python
Image A ──┐
           │ Siamese Encoder (EfficientNet + MLP → 512D normalized embedding)
Image B ──┘
Triplet Loss → learns to minimize distance for same-class pairs
```

Data Preprocessing
------------------
* Input images were resized to 384×384.

* Missing metadata values were handled via omission or categorical fallback.

* Patient-wise GroupShuffleSplit (80/20) ensured that images from the same patient never appeared in both train and validation sets (preventing data leakage).

* Training was balanced using a WeightedRandomSampler to handle class imbalance (melanoma ≈ 2%).

* Validation used deterministic transforms (no augmentation).

* No resizing or normalization differences between train and test.

Dependencies
-------------
| Library        | Version |
| -------------- | ------- |
| Python         | 3.12.6  |
| PyTorch        | 2.6.0   |
| torchvision    | 0.21.0  |
| timm           | 1.0.9   |
| albumentations | 1.4.18  |
| scikit-learn   | 1.5.2   |
| pandas         | 2.2.3   |
| matplotlib     | 3.9.2   |
| opencv-python  | 4.10.0  |

How To Use
-------------
### 1. Split and verify data
```python
python train.py --split --data_root data/train --images_dir data/train/train-img
```

### 2. Smoke Test
```python
python train.py --model_smoke --data_root data/train --images_dir data/train/train-img
```

### 3. Train
```python
python train.py --train_run --data_root data/train --images_dir data/train/train-img --epochs 10 --batch_size 16 --lr 0.0003 --backbone "tf_efficientnet_b0.ns_jft_in1k"
```

### 4. Predict
```python
python predict.py --data_root data/train --images_dir data/train/train-img --csv_path data/train/val_split.csv --checkpoint checkpoints/best.pt
```