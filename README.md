# Road Issues Classification with a Small CNN (PyTorch)

Classifying street-level photos into 6 categories of urban road problems
(potholes, damaged roads, broken signs, illegal parking, littering, vandalism)
with a compact CNN trained from scratch on CPU.

## Motivation

My graduate research was in deep reinforcement learning, so I had little hands-on
experience with computer vision. This project is my from-scratch walkthrough of a
CNN pipeline: data preparation, training loop, evaluation, and error analysis,
with a deliberately small model so it trains in ~30 minutes on an office laptop
(Intel i5-12400, no GPU).

The goal is not the highest accuracy, but to show the full reasoning loop:
train, measure, find out *why* it fails, fix it, measure again.

## Dataset

[Road Issues Detection Dataset](https://www.kaggle.com/datasets/programmerrdai/road-issues-detection-dataset)
(Kaggle, originally from Roboflow). Images are resized to 128x128.
The "Mixed Issues" multi-label category is excluded, leaving 6 classes,
split 80 / 10 / 10 (stratified) into train / val / test.

| Class            | Train | Val | Test |
|------------------|------:|----:|-----:|
| pothole          | 2678  | 335 |  335 |
| vandalism        | 1702  | 213 |  213 |
| broken_road_sign | 1435  | 179 |  179 |
| littering        | 1135  | 142 |  142 |
| damaged_road     |  542  |  67 |   68 |
| illegal_parking  |   83  |  11 |   10 |
| **Total**        | 7575  | 947 |  947 |

The dataset is heavily imbalanced: `pothole` alone is ~35% of the data while
`illegal_parking` is ~1%. This turns out to be the main story of the project (see below).

`prepare_dataset.py` handles a Windows-specific problem: the Kaggle archive has
paths longer than 260 characters, which makes `zipfile` fail. It extracts with the
`\\?\` long-path prefix and re-maps classes to short folder names.

## Model

A 3-block CNN:
[Conv3x3(16) -> BatchNorm -> ReLU -> MaxPool2 -> Dropout] x 3
-> Flatten -> FC(4096->16) -> ReLU -> FC(16->16) -> ReLU -> FC(16->6)

Optimizer AdamW (lr 1e-3), batch size 32, 50 epochs, cross-entropy loss.
The checkpoint with the best validation score is saved to `outputs/experiment_name/best_model.pth`
and used for the test set.

## Results

### v1 - baseline

| Metric              | Test    |
|---------------------|--------:|
| Accuracy            |  84.48% |
| Balanced accuracy   | ~67.75% |
| Best epoch (of 50)  |      16 |

| Class            | Test acc (recall) | Main confusion                  |
|------------------|------------------:|---------------------------------|
| pothole          |             99.7% | -                               |
| littering        |            94.37% | -                               |
| vandalism        |            79.34% | ->     broken_road_sign(11),    |
|                  |                   |    littering(12), pothole(20)   |
| broken_road_sign |            81.01% | ->     vandalism (27)           |
| illegal_parking  |             30.0% |        (only 10 samples)        |
| damaged_road     |             22.1% | ->     pothole (52 of 68)       |

![Training curves](outputs/baseline/training_curves.png)
![Result](outputs/baseline/evaluation_result.png)

### Interpretation:

* **Overall accuracy is misleading.** 84.48% is mostly `pothole` and `littering`.
  Balanced accuracy (mean per-class recall) is a fairer number: ~67.75%.
* **`damaged_road` collapses into `pothole`** (78% of the time). Both are
  "defects on the road surface", and `pothole` has 5x more samples, so the model
  defaults to the majority class when unsure. The 128x128 input resolution may also make
  fine-grained surface details harder to distinguish.
* **`broken_road_sign` -> `vandalism` and `vandalism` -> `littering`** suggest the
  model is partly learning "street scene with something wrong" rather than the object.
* **Validation accuracy oscillated wildly during training**. With this
  imbalance, a small shift in the decision boundary sends whole minority classes
  into `pothole`, which shows up as a dramatic oscillation of accuracy of 30-40 percentage points.


### v2 - class-weighted loss

The baseline showed a large gap between overall accuracy and balanced accuracy, with `damaged_road` and `illegal_parking` performing particularly poorly.

To address the class imbalance, I enabled class-weighted cross-entropy loss. The class weights are computed from the training-set class frequencies using inverse-frequency weighting.

The rest of the training configuration was kept unchanged, and model selection still used validation accuracy.

| Metric              | Test    |
|---------------------|--------:|
| Accuracy            |  79.41% |
| Balanced accuracy   | ~72.03% |
| Best epoch (of 50)  |      48 |

| Class            | Test acc (recall) | Main confusion                  |
|------------------|------------------:|---------------------------------|
| pothole          |             92.2% | ->     damaged_road(25)         |
| littering        |             94.4% | -                               |
| vandalism        |             66.2% | ->     broken_road_sign(13),    |
|                  |                   | damaged_road(11), littering(37) |
| broken_road_sign |             67.6% | -> littering(29), vandalism(24) |
| illegal_parking  |             50.0% |        (only 10 samples)        |
| damaged_road     |             61.8% | ->     pothole (25 of 68)       |

![Training curves](outputs/class-weighted_training/training_curves.png)
![Result](outputs/class-weighted_training/evaluation_result.png)

Comparison of v1 and v2

| Metric              | v1 baseline | v2 weighted |
|---------------------|------------:|------------:|
| Accuracy            |      84.48% |      79.41% |
| Balanced accuracy   |     ~67.75% |     ~72.03% |
| Macro F1            |       0.715 |       0.704 |
| damaged_road recall |       22.1% |       61.8% |


Class weighting reduced overall accuracy by 5.07 percentage points, but improved balanced accuracy by 4.28 percentage points.

The most noticeable change was `damaged_road` recall, which increased from 22.1% to 61.8%. In the baseline, 52 of 68 `damaged_road` test samples were classified as `pothole`; with class weighting, this decreased to 25 of 68.

At the same time, performance on some majority classes decreased. For example, `pothole` recall decreased from 99.7% to 92.2% and `vandalism` recall decreased from 79.3% to 66.2%.

This suggests that class weighting changed the decision boundary rather than simply improving every class: the model became less biased toward the majority classes, trading some overall accuracy for more balanced performance.

### Interpretation:

* **Overall accuracy decreased:** 84.48% → 79.41% (-5.07 percentage points).
* **`damaged_road` improved substantially:** recall increased from 22.1% → 61.8%, while misclassification as `pothole` decreased from 52/68 → 25/68 samples.
* **Balanced performance improved:** balanced accuracy increased from 67.75% → 72.03% (+4.28 percentage points).
* **The trade-off is not uniformly positive:** `pothole`, `vandalism`, and `broken_road_sign` recall all decreased.
* **`illegal_parking` improved from 30% → 50%**, but the test set contains only 10 samples, so this result should be interpreted cautiously.


## Project structure

config.py           hyperparameters, model shape, output paths, experiment switches
prepare_dataset.py  download from Kaggle and build dataset/raw/<class>/*.jpg
get_data.py         Dataset / DataLoader, stratified split, class weights
cnn_model.py        model, train / validate loops, metrics (confusion matrix, F1, ...)
plot.py             training curves, confusion matrix, per-class accuracy
main.py             end-to-end entry point
outputs/            all experiment result, save best_model.pth (git-ignored), figures,
                    csv file and config.txt

## How to run

bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Kaggle credentials: place kaggle.json in %USERPROFILE%\.kaggle\ (see kagglehub docs)
python main.py

Runs on CPU; ~30 min for 50 epochs on an i5-12400.

## Things I would do next

* Data augmentation (flip, random crop, color jitter) - the model currently sees
  each image exactly once per epoch with no variation.
* Higher input resolution for the `damaged_road` vs `pothole` distinction.
* Learning-rate schedule (cosine / ReduceLROnPlateau) to reduce validation oscillation.
