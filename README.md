# Road Issues Classification with a Small CNN (PyTorch)

Classifying street-level photos into 6 categories of urban road problems
(potholes, damaged roads, broken signs, illegal parking, littering, vandalism)
with a compact CNN trained from scratch on CPU.

![Training curves](outputs/baseline/training_curves.png)

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

A 3-block CNN, ~70K parameters:
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
| damaged_road     |            22.06% | ->     pothole (52 of 68)       |

![Result](outputs/baseline/evaluation_result.png)

### What the confusion matrix says

* **Overall accuracy is misleading.** 84.48% is mostly `pothole` and `littering`.
  Balanced accuracy (mean per-class recall) is a fairer number: ~67.75%.
* **`damaged_road` collapses into `pothole`** (78% of the time). Both are
  "defects on the road surface", and `pothole` has 5x more samples, so the model
  defaults to the majority class when unsure. 128x128 resolution probably also
  removes the crack detail that separates them.
* **`broken_road_sign` -> `vandalism` and `vandalism` -> `littering`** suggest the
  model is partly learning "street scene with something wrong" rather than the object.
* **Validation accuracy oscillated wildly during training**. With this
  imbalance, a small shift in the decision boundary sends whole minority classes
  into `pothole`, which shows up as a dramatic oscillation of accuracy of 30-40 percentage points.

## Project structure

config.py            hyperparameters, model shape, output paths, experiment switches
prepare_dataset.py   download from Kaggle and build dataset/raw/<class>/*.jpg
get_data.py          Dataset / DataLoader, stratified split, class weights
cnn_model.py         model, train / validate loops, metrics (confusion matrix, F1, ...)
plot.py              training curves, confusion matrix, per-class accuracy
main.py              end-to-end entry point
outputs/             all experiment result, save best_model.pth (git-ignored), figures
                     and csv file

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

