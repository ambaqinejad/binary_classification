from dataset import generator
from model.circle_model_v0 import CircleModelV0
import torch
import pandas as pd
import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, x_test, y_train, y_test = generator.generate(device)
model = CircleModelV0().to(device)
"""
Also can create this kind of model using:

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=5),
    torch.nn.Linear(in_features=5, out_features=1)
).to(device)
"""

print(model)
print(model.state_dict())

untrained_preds = model(x_test)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")

# df = pd.DataFrame({
#   "preds": untrained_preds[:].squeeze(1).cpu().detach().numpy(),
#   "true_answer": y_test[:].cpu().detach().numpy()
# })

# print(df.head())
trainer.train(model, x_train, y_train, x_test, y_test)