from matplotlib import pyplot as plt
import requests
from pathlib import Path


def plot_loss(epoch_count, train_loss, test_loss):
    plt.figure(figsize=(10, 7))

    plt.plot(epoch_count, train_loss, label='Training Loss')
    plt.plot(epoch_count, test_loss, label='Test Loss')
    plt.title('Training and Test Loss Plot')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def my_plot_decision_boundary(model, x_train, y_train, x_test, y_test):
    if Path("helper_functions.py").is_file():
        print("helper_functions.py already exists, skipping download")
    else:
        print("Downloading helper_functions.py")
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        with open("helper_functions.py", "wb") as file:
            file.write(request.content)
    from helper_functions import plot_predictions, plot_decision_boundary
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.show()