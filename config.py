from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

@dataclass
class Dataset_info():
    dataset_path:str = str(PROJECT_DIR / "dataset")
    dataset_name:str = "programmerrdai/road-issues-detection-dataset"

    raw_path:str = str(PROJECT_DIR / "dataset" / "raw")
    cache_path:str = str(PROJECT_DIR / "dataset" / "cache")

@dataclass
class Output_paths():
    experiment_name = "baseline"   # Creat experiment result folder
    result_dir: str = str(PROJECT_DIR / "outputs" / experiment_name)
    best_model_path: str = str(PROJECT_DIR / "outputs" / "best_model.pth")

@dataclass
class Hyperparameters():
    learning_rate:float = 0.001
    num_epochs:int = 50
    batch_size:int = 32
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

    use_class_weights: bool = False
    model_selection_metric: str = "accuracy"  # "accuracy" or "balanced_accuracy"

@dataclass
class CNN_Parameters():
    input_channels:int =3
    conv1_out_channels:int = 16
    conv2_out_channels:int = 16
    conv3_out_channels:int = 16
    conv_kernel_size:int = 3
    num_classes:int = 6

    dropout1_rate:float = 0.5
    dropout2_rate:float = 0.25
    dropout3_rate:float = 0.1

    bn1_in_channels:int = 16
    bn2_in_channels:int = 16
    bn3_in_channels:int = 16

    fc1_in:int = 16 * 16 * 16
    fc2_in:int = 16
    fc3_in:int = 16

    pool_kernel_size:int = 2
