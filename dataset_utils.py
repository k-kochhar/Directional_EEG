import numpy as np
import torch
import os
import mne
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data, angles):
        self.data = [torch.tensor(d.astype(np.float32)).T for d in data]
        self.angles = [torch.tensor([a], dtype=torch.float32) for a in angles]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.angles[idx]

def preprocess_data(sub_id, freq_bands=[[1, 50]], data_folder="DATA", fs_new=128):
    angle_mapping = {
        range(0, 4): 90,
        range(4, 8): 60,
        range(8, 12): 45,
        range(12, 16): 30,
        range(16, 20): 5
    }

    trial_angles = {}
    for trial_range, angle in angle_mapping.items():
        for trial_idx in trial_range:
            trial_angles[trial_idx] = angle

    def add_direction(angle, direction):
        return -angle if direction == 0 else angle

    directions = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    tmp_name = f'{int(sub_id):03d}'
    data_list, angle_list = [], []

    for j in range(20):
        fifname = f'S{tmp_name}_E1_Trial{j + 1}_raw.fif'
        path = f'{data_folder}/S{tmp_name}/E1/'
        if not os.path.exists(path + fifname):
            continue

        raw = mne.io.read_raw_fif(path + fifname, preload=True)
        raw.set_eeg_reference(ref_channels='average', verbose=False)

        combined_data = []
        for band in freq_bands:
            raw_band = raw.copy()
            raw_band.filter(band[0] if band[0] > 1 else None, band[1], fir_design='firwin', verbose=False)
            raw_band.resample(fs_new, npad="auto", verbose=False)
            combined_data.append(raw_band.get_data().T)

        combined_data = np.concatenate(combined_data, axis=-1) if len(combined_data) > 1 else combined_data[0]
        angle = add_direction(trial_angles[j], directions[j])

        for start in range(0, combined_data.shape[0] - 128 + 1, 64):
            segment = combined_data[start:start + 128, :]
            data_list.append(segment)
            angle_list.append(angle)

    return data_list, angle_list
