from eeg_model import EEGAnglePredictionModel
from dataset_utils import EEGDataset, preprocess_data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def train():
    model = EEGAnglePredictionModel(input_channels=64, sample_length=128)
    all_data, all_angles = [], []

    for subject_id in range(1, 3):  # just 2 subjects for now
        data, angles = preprocess_data(str(subject_id), data_folder="DATA")
        all_data.extend(data)
        all_angles.extend(angles)

    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]
    all_angles = [all_angles[i] for i in indices]

    train_size = int(len(all_data) * 0.7)
    valid_size = int(len(all_data) * 0.15)

    train_dataset = EEGDataset(all_data[:train_size], all_angles[:train_size])
    valid_dataset = EEGDataset(all_data[train_size:train_size+valid_size], all_angles[train_size:train_size+valid_size])
    test_dataset = EEGDataset(all_data[train_size+valid_size:], all_angles[train_size+valid_size:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(3):  # shorter for quick testing
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved model to model.pth")

if __name__ == '__main__':
    train()
