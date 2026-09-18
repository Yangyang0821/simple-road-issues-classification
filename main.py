from cnn_model import set_model, model_training, validate_epoch, load_best_model
from config import CNN_Parameters, Config, Hyperparameters
from get_data import (download_dataset, load_dataset, create_dataloaders,
                      count_labels, compute_class_weights)
from prepare_dataset import TRAIN_CLASSES
from plot import plot_training_curves, plot_evaluation_result, save_report_csv, save_config

if __name__ == "__main__":
    class_names = list(TRAIN_CLASSES)

    dataset_root = download_dataset()
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_root)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    # Class distribution of the training set
    train_counts = count_labels(train_dataset, CNN_Parameters.num_classes)
    print("Train class distribution:", {name: int(n) for name, n in zip(class_names, train_counts)})

    class_weights = None
    if Hyperparameters.use_class_weights:
        class_weights = compute_class_weights(train_dataset, CNN_Parameters.num_classes)

    # Initialize model, loss function, optimizer and device
    cnn_model, loss_fn, adam_optimizer, h_device = set_model(class_weights)

    # Training (saves best model)
    history = model_training(h_device, loss_fn, cnn_model, adam_optimizer, train_loader, val_loader)

    # Evaluate best model on validation & test set
    cnn_model = load_best_model(cnn_model, h_device)
    _, _, val_cm = validate_epoch(h_device, loss_fn, cnn_model, val_loader)
    test_accuracy, test_loss, test_cm = validate_epoch(h_device, loss_fn, cnn_model, test_loader)
    print(f"Test Loss: {test_loss:.4f}   Test Accuracy: {test_accuracy:.2f}%")

    plot_training_curves(history)
    plot_evaluation_result(test_cm, val_cm, class_names)
    save_report_csv(test_cm, class_names, filename="result_test.csv")
    config = Config()
    save_config(config, filename="config.json")
