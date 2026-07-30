import os
import numpy as np
import torch
# =========================
# Random seed
# =========================

import sys

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 42
print("Random seed:", seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from eegnet import EEGNet



# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:")
print(device)



# =========================
# Dataset
# =========================

class EEGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )


        # 原始标签:
        # 7,8,9,10

        # 转换:
        # 0,1,2,3

        self.y = torch.tensor(
            y - 7,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.y)


    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )



# =========================
# Load data
# =========================

X = np.load(
    "data/processed/A01T_DL_X.npy"
)


y = np.load(
    "data/processed/A01T_DL_y.npy"
)


print("----------------")
print("Data:")
print(X.shape)



# =========================
# Train/Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



train_dataset = EEGDataset(
    X_train,
    y_train
)


test_dataset = EEGDataset(
    X_test,
    y_test
)



train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)



# =========================
# Model
# =========================

model = EEGNet(
    num_classes=4
)


model = model.to(device)


print("----------------")
print(model)



# =========================
# Loss
# =========================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



# =========================
# Training
# =========================

epochs = 100


for epoch in range(epochs):

    model.train()

    loss_total = 0


    for X_batch, y_batch in train_loader:


        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        optimizer.zero_grad()


        output = model(
            X_batch
        )


        loss = criterion(
            output,
            y_batch
        )


        loss.backward()


        optimizer.step()


        loss_total += loss.item()



    print(
        f"Epoch {epoch+1}/{epochs}, Loss={loss_total:.4f}"
    )



# =========================
# Evaluation
# =========================

model.eval()


preds = []


with torch.no_grad():

    for i in range(len(test_dataset)):


        X_sample, y_sample = test_dataset[i]


        X_sample = X_sample.unsqueeze(0)


        X_sample = X_sample.to(device)



        output = model(
            X_sample
        )


        pred = torch.argmax(
            output,
            dim=1
        )


        preds.append(
            pred.cpu().item()
        )



acc = accuracy_score(
    test_dataset.y.numpy(),
    preds
)



print("----------------")
print("EEGNet Accuracy:")
print(acc)



# =========================
# Save model
# =========================

os.makedirs(
    "results/models",
    exist_ok=True
)


torch.save(
    model.state_dict(),
    "results/models/eegnet_A01T.pth"
)



# =========================
# Save prediction
# =========================

os.makedirs(
    "results/logs",
    exist_ok=True
)


np.save(
    "results/logs/eegnet_preds.npy",
    np.array(preds)
)


np.save(
    "results/logs/eegnet_true.npy",
    test_dataset.y.numpy()
)


print("----------------")
print("EEGNet model saved")