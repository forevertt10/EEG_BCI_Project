import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# =========================
# GPU
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)



# =========================
# Dataset
# =========================

class EEGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        # 标签转0-3
        self.y = torch.tensor(
            y-7,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.y)


    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



# =========================
# CNN Model
# =========================

class EEG_CNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(3,15),
                padding=(1,7)
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2,4)
            ),


            nn.Conv2d(
                16,
                32,
                kernel_size=(3,15),
                padding=(1,7)
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),


            nn.AdaptiveAvgPool2d(
                (1,1)
            )

        )


        self.classifier = nn.Linear(
            32,
            4
        )


    def forward(self,x):

        x=self.features(x)

        x=x.view(
            x.size(0),
            -1
        )

        x=self.classifier(x)

        return x



# =========================
# Load data
# =========================

X=np.load(
    "data/processed/A01T_DL_X.npy"
)

y=np.load(
    "data/processed/A01T_DL_y.npy"
)



X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


train_loader=DataLoader(
    EEGDataset(X_train,y_train),
    batch_size=16,
    shuffle=True
)



test_dataset=EEGDataset(
    X_test,
    y_test
)



# =========================
# Model
# =========================

model=EEG_CNN().to(device)


criterion=nn.CrossEntropyLoss()


optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



# =========================
# Training
# =========================

epochs=50


for epoch in range(epochs):

    model.train()

    total_loss=0


    for X_batch,y_batch in train_loader:

        X_batch=X_batch.to(device)

        y_batch=y_batch.to(device)


        optimizer.zero_grad()


        output=model(X_batch)


        loss=criterion(
            output,
            y_batch
        )


        loss.backward()


        optimizer.step()


        total_loss+=loss.item()


    print(
        f"Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}"
    )



# =========================
# Evaluation
# =========================

model.eval()


preds=[]


with torch.no_grad():

    for i in range(len(test_dataset)):

        X_sample,y_sample=test_dataset[i]


        X_sample=X_sample.unsqueeze(0).to(device)


        output=model(X_sample)


        pred=torch.argmax(
            output,
            dim=1
        )


        preds.append(
            pred.cpu().item()
        )
# 保存预测结果

np.save(
    "results/logs/cnn_preds.npy",
    np.array(preds)
)


np.save(
    "results/logs/cnn_true.npy",
    test_dataset.y.numpy()
)


accuracy=accuracy_score(
    test_dataset.y.numpy(),
    preds
)


print("----------------")
print("CNN Accuracy:")
print(accuracy)

# =========================
# Save model
# =========================

import os


os.makedirs(
    "results/models",
    exist_ok=True
)


torch.save(
    model.state_dict(),
    "results/models/cnn_A01T.pth"
)


print("----------------")
print("Model saved")