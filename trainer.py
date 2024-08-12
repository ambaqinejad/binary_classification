import torch
from utils.metrics import accuracy_fn
from utils.plotter import plot_loss, my_plot_decision_boundary
from pathlib import Path


def train(model, x_train, y_train, x_test, y_test):
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=.1)
    torch.random.manual_seed(42)
    epochs = 10000
    epoch_counts = []
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        y_logits = model(x_train).squeeze(dim=1)
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss_value = loss(torch.sigmoid(y_logits), y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze(dim=1)
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss(torch.sigmoid(test_logits), y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        if epoch % 100 == 0:
            epoch_counts.append(epoch)
            train_losses.append(loss_value.item())
            test_losses.append(test_loss.item())
            print(f'Epoch: {epoch} | Loss: {loss_value} | Accuracy: {acc}')
    plot_loss(epoch_counts, train_losses, test_losses)
    my_plot_decision_boundary(model, x_train, y_train, x_test, y_test)
    save_model(model)

def save_model(model):
    path = Path("./model_dir")
    path.mkdir(parents=True, exist_ok=True)
    model_name = "binary_classifier_v0.pth"
    model_path = path / model_name
    torch.save(model.state_dict(), model_path)
