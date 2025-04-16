import torch
from eeg_model import EEGAnglePredictionModel
from dataset_utils import EEGDataset, preprocess_data
from torch.utils.data import DataLoader
import numpy as np

def compute_average_channel_importance(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_attn_weights = []

    class Hook:
        def __init__(self):
            self.out = []

        def __call__(self, module, input, output):
            b, c, _ = input[0].shape
            avg_out = module.mlp(module.avg_pool(input[0]).view(b, c))
            max_out = module.mlp(module.max_pool(input[0]).view(b, c))
            weights = torch.sigmoid(avg_out + max_out).cpu()
            self.out.append(weights)

    hook = Hook()
    model.ca1.register_forward_hook(hook)

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            model(x)

    all_weights = torch.cat(hook.out, dim=0)
    avg_weights = torch.mean(all_weights, dim=0).numpy()
    return avg_weights

if __name__ == "__main__":
    model = EEGAnglePredictionModel()
    model.load_state_dict(torch.load("model.pth"))
    
    data, angles = preprocess_data("1", data_folder="DATA")
    dataset = EEGDataset(data, angles)
    loader = DataLoader(dataset, batch_size=16)
    
    weights = compute_average_channel_importance(model, loader)
    top_channels = np.argsort(weights)[-5:][::-1]
    print("Top Channels by Attention:")
    for i in top_channels:
        print(f"Channel {i}: {weights[i]:.4f}")
