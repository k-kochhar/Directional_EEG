import numpy as np
import torch
import os
import mne
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data, angles):
        """
        Custom PyTorch Dataset for EEG data.
        
        Args:
            data: List of EEG data arrays with shape [time, channels] or tensors with shape [channels, time]
            angles: List of angle labels or tensors
        """
        self.data = []
        for d in data:
            if isinstance(d, torch.Tensor):
                self.data.append(d)
            else:
                self.data.append(torch.tensor(d.astype(np.float32)).transpose(0, 1))
        
        self.angles = []
        for a in angles:
            if isinstance(a, torch.Tensor):
                self.angles.append(a)
            else:
                self.angles.append(torch.tensor([a], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.angles[idx]

def preprocess_data(sub_id, freq_bands=[[1, 50]], data_folder="DATA", fs_new=128, time_len=128, window_lap=64):
    """
    Preprocesses EEG data for a given subject with angle labels.
    
    Args:
        sub_id: Subject ID
        freq_bands: List of frequency bands to filter
        data_folder: Folder containing the data
        fs_new: New sampling rate
        time_len: Fixed length of time window in samples (e.g., 128 samples = 1 second at 128 Hz)
        window_lap: Overlap between consecutive windows in samples
        
    Returns:
        data_list: List of preprocessed EEG data
        angle_list: List of corresponding angles in degrees
    """
    # Define angles for each trial group (in degrees)
    angle_mapping = {
        range(0, 4): 90,    # 90 degree trials
        range(4, 8): 60,    # 60 degree trials
        range(8, 12): 45,   # 45 degree trials
        range(12, 16): 30,  # 30 degree trials
        range(16, 20): 5    # 5 degree trials
    }
    
    # Map each trial index to its angle
    trial_angles = {}
    for trial_range, angle in angle_mapping.items():
        for trial_idx in trial_range:
            trial_angles[trial_idx] = angle
    
    # Function to determine left or right direction and add sign to angle
    def add_direction(angle, direction):
        if direction == 0:  # Left
            return -angle
        else:  # Right
            return angle
    
    # Direction labels for each trial (0 for left, 1 for right)
    directions = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    
    print(f"Processing Subject S{sub_id}")
    data_list = []
    angle_list = []
    
    # Format the subject ID for file naming
    if len(sub_id) == 1:
        tmp_name = '00' + sub_id
    else:
        tmp_name = '0' + sub_id
    
    # Process each trial
    for j in range(20):  # Process all 20 trials
        fifname = 'S' + tmp_name + '_E1_Trial' + str(j + 1) + '_raw.fif'
        path = data_folder + '/S' + tmp_name + '/E1/'
        
        # Skip if file doesn't exist
        if not os.path.exists(path + fifname):
            continue
            
        # Read raw data
        raw = mne.io.read_raw_fif(path + fifname, preload=True)
        raw = raw.set_eeg_reference(ref_channels='average', verbose=False)
        
        combined_data = []
        # Process each frequency band
        for freq_band in freq_bands:
            l_freq, h_freq = freq_band
            
            # Apply filtering
            if l_freq <= 1:
                raw.filter(None, h_freq, fir_design='firwin', verbose=False)
            else:
                raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
                
            # Resample
            raw_resampled = raw.copy().resample(fs_new, npad="auto", verbose=False)
            
            # Extract and transpose data
            data = raw_resampled.get_data()
            data = data.transpose()  # [time, channels]
            
            combined_data.append(data)
        
        # Concatenate data across frequency bands if multiple bands are used
        if len(combined_data) > 1:
            combined_data = np.concatenate(combined_data, axis=-1)
        else:
            combined_data = combined_data[0]
            
        # Get angle with sign (negative for left, positive for right)
        angle = add_direction(trial_angles[j], directions[j])
        
        # Apply sliding window to handle variable length trials
        # Calculate step size (stride) based on window_lap
        step = time_len - window_lap
        
        # Check if the data is long enough for at least one window
        if combined_data.shape[0] >= time_len:
            # Extract segments using sliding window
            for start in range(0, combined_data.shape[0] - time_len + 1, step):
                segment = combined_data[start:start + time_len, :]
                data_list.append(segment)
                angle_list.append(angle)
    
    return data_list, angle_list
