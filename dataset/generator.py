import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt


def generate(device):
    num_samples = 10000
    x, y = make_circles(n_samples=num_samples, noise=.03, random_state=42)
    print(f"First 5 X features:\n{x[:5]}")
    print(f"\nFirst 5 y labels:\n{y[:5]}")
    circles = pd.DataFrame({
        "X1": x[:, 0],
        "X2": x[:, 1],
        "Label": y
    })
    print(circles.head(10))
    print(circles.Label.value_counts())
    plt.scatter(circles.X1, circles.X2, c=circles.Label)
    plt.show()
    print(f"X shape: {x.shape}, Y shape: {y.shape}")
    x = torch.from_numpy(x).type(torch.float).to(device)
    y = torch.from_numpy(y).type(torch.float).to(device)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(x_train.device)
    return x_train, x_test, y_train, y_test
