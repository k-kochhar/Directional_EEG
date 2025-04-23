import os
import statistics
import mne
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

# Testing flag (set to True for quick testing, False for full training)
TESTING = True

# Path settings
DATA_DIR = "DATA"
result_folder_name = 'results'

# Preprocessing settings
label_type = 'all_direction'
time_len = 1
sampling_rate = 128
sample_len, channels_num = int(sampling_rate * time_len), 64
overlap_rate = 0.5
window_sliding = int(sample_len * overlap_rate)
freq_bands = [[1, 50]]

# Hyperparameters
lr = 5e-4
epochs = 5 if TESTING else 100  # Use fewer epochs when testing
batch_size = 64
num_class = 2  # Binary classification (left/right)

# Create results directory
os.makedirs(result_folder_name, exist_ok=True)


def normalize_data(data):
    """
    Normalize the data to have a mean of 0 and standard deviation of 1.
    
    Args:
        data: The data to be normalized.
    
    Returns:
        The normalized data.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def trails_split(x, time_len, window_lap, sampling_rate=128):
    """
    Divide the signal into segments.
    
    Args:
        x: list[np.ndarray], len(x) is trails_num, each element shape as [time, channel]
        time_len: Duration of each segment in seconds
        window_lap: Step size for sliding window
        sampling_rate: Sampling rate of the data
    
    Returns:
        split_x: List of np.ndarray, each element shape as [sample, time, channel]
        split_index: Record of start and end positions
    """
    sample_len = int(sampling_rate * time_len)

    split_index = []
    split_x = []
    
    # Divide for each trial
    for i_tra in range(len(x)):
        trail_len = x[i_tra].shape[0]
        left = np.arange(start=0, stop=trail_len, step=window_lap)
        
        # Remove segments that would go beyond the trial length
        while left[-1] + sample_len - 1 > trail_len - 1:
            left = np.delete(left, -1, axis=0)
        
        right = left + sample_len

        split_index.append(np.stack([left, right], axis=-1))

        # Extract segments
        temp = [x[i_tra][left[i]: right[i]] for i in range(left.shape[0])]
        split_x.append(np.stack(temp, axis=0))

    return split_x, split_index


def list_data_split(eeg, voice, label, time_len, window_lap, sampling_rate=128, num_classes=2):
    """
    Divide the data, windowing the data at fixed intervals.
    
    Args:
        eeg: EEG data
        voice: Voice data (can be None)
        label: Labels
        time_len: Sample duration in seconds
        window_lap: Step size for sliding window
        sampling_rate: Sampling rate
        num_classes: Number of classes for one-hot encoding
    
    Returns:
        eeg: Divided EEG data
        label: Divided labels
        split_index: Record of window positions
    """
    # Process voice data if provided
    if voice is not None:
        for i in range(len(voice)):
            voice[i] = np.stack(voice[i], axis=-1)
            if voice[i].shape[0] != eeg[i].shape[0]:
                raise ValueError("The length of voice and eeg must be the same")
        voice, split_index = trails_split(voice, time_len, window_lap, sampling_rate=sampling_rate)
    
    # Split EEG data
    eeg, split_index = trails_split(eeg, time_len, window_lap, sampling_rate=sampling_rate)

    # Prepare labels with one-hot encoding
    total_label = []
    for i_tra in range(len(eeg)):
        samples_num = eeg[i_tra].shape[0]
        sub_label = label[i_tra] * np.ones(samples_num)
        # Convert to one-hot encoding
        sub_label = np.eye(num_classes)[sub_label.astype(int)]
        total_label.append(sub_label)

    label = total_label

    if voice is not None:
        return eeg, voice, label, split_index
    else:
        return eeg, label, split_index


def five_fold(data, label, split_index=None, shuffle=False, keep_trial=True):
    """
    Create 5-fold cross-validation indices.
    
    Args:
        data: Data to split
        label: Labels to split
        split_index: Split indices (optional)
        shuffle: Whether to shuffle the data
        keep_trial: Whether to keep trials together
    
    Returns:
        train_fold: List of training indices for each fold
        test_fold: List of testing indices for each fold
    """
    kf = KFold(n_splits=5, shuffle=shuffle)
    
    # Create empty lists for each fold
    train_fold = [[] for _ in range(5)]
    test_fold = [[] for _ in range(5)]
    
    # Generate trial indices
    trial_indices = list(range(len(data)))
    
    # Split trials into folds
    for fold_idx, (train_trials, test_trials) in enumerate(kf.split(trial_indices)):
        # Process training trials
        for trial_idx in train_trials:
            # Add the indices for segments within this trial
            if trial_idx < len(split_index) and split_index is not None:
                train_fold[fold_idx].append(split_index[trial_idx])
            else:
                # Create an array of indices for the training data
                if trial_idx < len(data):
                    segment_indices = np.arange(len(data[trial_idx]))
                    train_fold[fold_idx].append(segment_indices)
        
        # Process testing trials
        for trial_idx in test_trials:
            # Add the indices for segments within this trial
            if trial_idx < len(split_index) and split_index is not None:
                test_fold[fold_idx].append(split_index[trial_idx])
            else:
                # Create an array of indices for the testing data
                if trial_idx < len(data):
                    segment_indices = np.arange(len(data[trial_idx]))
                    test_fold[fold_idx].append(segment_indices)
    
    return train_fold, test_fold


def create_model(sample_len=128, channels_num=64, lr=5e-4):
    """
    Create a CNN model for EEG classification.
    
    Args:
        sample_len: Length of each segment (samples)
        channels_num: Number of EEG channels
        lr: Learning rate
    
    Returns:
        Compiled Keras model
    """
    # Using Sequential API
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', 
                              input_shape=(sample_len, channels_num, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def preprocess_data(sub_id, freq_bands=[[1, 50]], data_folder="DATA", fs_new=128, label_type='all_direction'):
    """
    Preprocess EEG data for a given subject.
    
    Args:
        sub_id: Subject identifier (e.g., 'S001')
        freq_bands: List of frequency bands for filtering
        data_folder: Path to the data folder
        fs_new: New sampling rate (after resampling)
        label_type: Type of labels to use
    
    Returns:
        data_list: List of preprocessed EEG data
        label_list: List of corresponding labels
    """
    # Define labels based on direction (0-left, 1-right)
    labels = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    
    # Select trials based on the label type
    if label_type == 'all_direction':
        selected_trails = range(20)
    elif label_type == '90':
        selected_trails = [0, 1, 2, 3]
    elif label_type == '60':
        selected_trails = [4, 5, 6, 7]
    elif label_type == '45':
        selected_trails = [8, 9, 10, 11]
    elif label_type == '30':
        selected_trails = [12, 13, 14, 15]
    elif label_type == '5':
        selected_trails = [16, 17, 18, 19]
    
    # Print the subject ID being processed
    print(f"Processing Subject {sub_id}")
    data_list = []
    label_list = []
    
    # Loop over the selected trials to read and preprocess the data
    for j in selected_trails:
        # Construct file name and path
        fifname = f'{sub_id}_E1_Trial{j + 1}_raw.fif'
        path = os.path.join(data_folder, sub_id, 'E1')
        file_path = os.path.join(path, fifname)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
        
        try:
            # Read the raw EEG data file
            raw = mne.io.read_raw_fif(file_path, preload=True)
            
            # Perform average rereferencing on the EEG data
            raw = raw.set_eeg_reference(ref_channels='average', verbose=False)
            
            combined_data = []
            # Loop over each frequency band to filter and resample the data
            for freq_band in freq_bands:
                l_freq, h_freq = freq_band
                
                # Apply a low-pass filter if the lower frequency is below 1Hz
                if l_freq <= 1:
                    raw.filter(None, h_freq, fir_design='firwin', verbose=False)
                else:
                    # Otherwise, apply a band-pass filter
                    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
                
                # Resample the data to the new sampling rate
                raw = raw.resample(fs_new, npad="auto", verbose=False)
                
                # Extract the data and transpose it for consistency
                data = raw.get_data()
                data = data.transpose()
                
                # Get only the first 64 channels if there are more
                if data.shape[1] > channels_num:
                    data = data[:, :channels_num]
                
                # Append the filtered and resampled data to the combined list
                combined_data.append(data)
            
            # Concatenate the data along the channel dimension
            if len(combined_data) > 1:
                combined_data = np.concatenate(combined_data, axis=-1)
            else:
                combined_data = combined_data[0]
            
            # Assign the label based on the trial index
            label = labels[j]
            
            # Append the combined data and label to the respective lists
            data_list.append(combined_data)
            label_list.append(label)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    return data_list, label_list


def calculate_mean_and_std_accuracy(file_path):
    """
    Calculate mean accuracy and standard deviation from results file.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        subject_results: Dictionary of (mean, std) tuples for each subject
        subject_means: List of mean accuracies
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Store all accuracies and each subject's average accuracy and standard deviation
    subject_means = []
    subject_results = {}
    
    for line in lines:
        parts = line.strip().split(':')
        subject_id = parts[0].strip()
        accuracies = [float(x) for x in parts[1].split(',')]
        mean_accuracy = sum(accuracies) / len(accuracies)
        subject_means.append(mean_accuracy)
        std_deviation = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
        subject_results[subject_id] = (mean_accuracy, std_deviation)
    
    return subject_results, subject_means


def main():
    """
    Main function to preprocess, split, and train a model with cross-validation on EEG data.
    
    The function processes data for available subjects, normalizes it, and performs 5-fold cross-validation.
    It then trains a model on the training folds and evaluates it on the test folds, collecting results.
    """
    # Get all subjects from the DATA directory
    subjects = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('S')]
    if TESTING:
        subjects = [s for s in subjects if s == "S001"]  # Only test with S001 when testing
    subjects.sort()
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    # Initialize dictionaries to store data, labels, and fold indices for all subjects
    all_subjects_data = {}
    all_subjects_label = {}
    all_subjects_fold_index = {}
    all_subjects_segments = {}  # Store the segmented data before concatenation
    
    # Process each subject
    for subject_id in subjects:
        # Load and preprocess data for the subject
        data, label = preprocess_data(
            sub_id=subject_id,
            freq_bands=freq_bands,
            data_folder=DATA_DIR,
            fs_new=sampling_rate,
            label_type=label_type
        )
        
        # Skip if no data was loaded
        if len(data) == 0:
            print(f"No data found for subject {subject_id}, skipping...")
            continue
        
        # Partition the data into training and validation sets
        data_segments, label_segments, split_index = list_data_split(
            data, None, label, time_len, window_sliding,
            sampling_rate=sampling_rate, num_classes=num_class
        )
        
        # Normalize the data to have zero mean and unit variance
        data_segments = [normalize_data(d) for d in data_segments]
        
        # Store the segmented data for later use
        all_subjects_segments[subject_id] = {
            'data': data_segments,
            'labels': label_segments
        }
        
        # Split the data into 5 folds, ensuring the trials are not shuffled
        train_fold, test_fold = five_fold(
            data_segments, label_segments, split_index, shuffle=False, keep_trial=True
        )
        
        # Store the fold indices for the subject
        all_subjects_fold_index[subject_id] = [train_fold, test_fold]
    
    if not all_subjects_segments:
        print("No subjects were successfully processed. Exiting.")
        return
    
    # List to store the results of all folds
    all_folds_results = []
    
    # Perform 5-fold cross-validation
    for i_fold in range(5):
        print(f"\nTraining fold {i_fold + 1}/5")
        
        # Prepare training and testing data for the current fold
        x_train_all = []
        y_train_all = []
        
        # Process each subject
        for subject_id, [train_fold, test_fold] in all_subjects_fold_index.items():
            # Get the subject's segmented data
            subject_segments = all_subjects_segments[subject_id]
            data_segments = subject_segments['data']
            label_segments = subject_segments['labels']
            
            # Extract training data for this fold
            for trial_idx, indices in enumerate(train_fold[i_fold]):
                # Extract segments from this trial based on indices
                if isinstance(indices, np.ndarray) and indices.ndim == 2:
                    # Handle start-end indices
                    for start, end in indices:
                        trial_data = data_segments[trial_idx][start:end]
                        trial_labels = label_segments[trial_idx][start:end]
                        x_train_all.append(trial_data)
                        y_train_all.append(trial_labels)
                else:
                    # Indices are directly usable
                    x_train_all.append(data_segments[trial_idx])
                    y_train_all.append(label_segments[trial_idx])
        
        # Concatenate all training data
        x_train = np.vstack(x_train_all) if x_train_all else np.array([])
        y_train = np.vstack(y_train_all) if y_train_all else np.array([])
        
        if len(x_train) == 0:
            print(f"No training data for fold {i_fold + 1}, skipping...")
            continue
        
        # Reshape data for CNN (samples, time, channels, 1)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        
        # Create and train the model with the training data
        model = create_model(sample_len=sample_len, channels_num=channels_num, lr=lr)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=8, mode='auto')
        
        model.fit(
            x_train, y_train,
            epochs=epochs,  # Use the epochs set by the TESTING flag
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Dictionary to store results for each subject in the current fold
        sub_results = {}
        fold_accuracies = []  # To collect accuracies for calculating the mean accuracy
        
        # Evaluate the model on the test data for the current fold
        for subject_id, [train_fold, test_fold] in all_subjects_fold_index.items():
            # Get the subject's segmented data
            subject_segments = all_subjects_segments[subject_id]
            data_segments = subject_segments['data']
            label_segments = subject_segments['labels']
            
            subject_accuracies = []
            
            # Extract testing data for this fold
            for trial_idx, indices in enumerate(test_fold[i_fold]):
                # Extract test segments from this trial
                if isinstance(indices, np.ndarray) and indices.ndim == 2:
                    # Handle start-end indices
                    for start, end in indices:
                        trial_data = data_segments[trial_idx][start:end]
                        trial_labels = label_segments[trial_idx][start:end]
                        
                        # Skip empty segments
                        if trial_data.shape[0] == 0:
                            continue
                        
                        # Reshape for CNN
                        x_test = trial_data.reshape(trial_data.shape[0], trial_data.shape[1], trial_data.shape[2], 1)
                        
                        # Evaluate
                        loss, acc = model.evaluate(x_test, trial_labels, batch_size=batch_size, verbose=0)
                        subject_accuracies.append(acc)
                else:
                    # Use all segments from this trial
                    trial_data = data_segments[trial_idx]
                    trial_labels = label_segments[trial_idx]
                    
                    # Skip empty trials
                    if trial_data.shape[0] == 0:
                        continue
                    
                    # Reshape for CNN
                    x_test = trial_data.reshape(trial_data.shape[0], trial_data.shape[1], trial_data.shape[2], 1)
                    
                    # Evaluate
                    loss, acc = model.evaluate(x_test, trial_labels, batch_size=batch_size, verbose=0)
                    subject_accuracies.append(acc)
            
            # Store the accuracies for the current subject
            if subject_accuracies:
                sub_results[subject_id] = subject_accuracies
                fold_accuracies.extend(subject_accuracies)
            else:
                print(f"No test data for subject {subject_id} in fold {i_fold + 1}")
                sub_results[subject_id] = [0.0]  # Default value if no test data
        
        # Store the results of the current fold
        all_folds_results.append(sub_results)
        
        # Calculate and print the mean accuracy for the current fold
        if fold_accuracies:
            mean_accuracy_fold = np.mean(fold_accuracies)
            print(f"Mean accuracy for fold {i_fold + 1}: {mean_accuracy_fold:.4f}")
        else:
            print(f"No accuracy data for fold {i_fold + 1}")
    
    return all_folds_results


if __name__ == '__main__':
    """
    Main entry point of the script.
    Processes the data, trains models, and saves results.
    """
    # Call the main function to get the results of all folds
    all_folds_results = main()
    
    if all_folds_results is None:
        print("No results to save. Exiting.")
        exit()
    
    # Dictionary to store the mean accuracy scores for each subject, per fold
    subject_fold_scores = {}
    
    # Initialize the dictionary to ensure each subject has an entry for each fold
    for fold_index, fold_results in enumerate(all_folds_results):
        for sub_id in fold_results.keys():
            # Create a list of zeros for each subject's fold, assuming equal number of folds
            if sub_id not in subject_fold_scores:
                subject_fold_scores[sub_id] = [0] * len(all_folds_results)
    
    # Calculate the mean accuracy for each fold and store it in the dictionary
    for fold_index, fold_results in enumerate(all_folds_results):
        for sub_id, sub_results in fold_results.items():
            # Sum the accuracies of all trials in the current fold for the current subject
            fold_accuracy_sum = sum(sub_results)
            # Calculate the mean accuracy for the current fold
            mean_fold_accuracy = fold_accuracy_sum / len(sub_results)
            # Store the mean accuracy in the subject's fold scores
            subject_fold_scores[sub_id][fold_index] = mean_fold_accuracy
    
    # Construct the file name for the results
    model_name = "CNN"  # Using CNN model
    result_file_name = f"results_{model_name}_SR{sampling_rate}_OR{int(overlap_rate*100)}.txt"
    # Define the full path for the result file
    result_file_path = os.path.join(result_folder_name, result_file_name)
    
    # Check if the result folder exists, and create it if it does not
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)
    
    # Open a file at the specified path and write the results
    with open(result_file_path, 'w') as file:
        for sub_id, scores in subject_fold_scores.items():
            # Write the subject's ID and their mean accuracy scores for each fold to the file
            file.write(f"Subject {sub_id}: {', '.join([f'{score:.4f}' for score in scores])}\n")
    
    # Print the location where the results have been saved
    print(f"Results saved to {result_file_path}")
    
    # Calculate each subject's mean accuracy and standard deviation, and collect all accuracies
    subject_results, subject_means = calculate_mean_and_std_accuracy(result_file_path)
    total_mean_accuracy = sum(result[0] for result in subject_results.values()) / len(subject_results)
    
    # Calculate the variance of the mean accuracies
    variance = sum((x - total_mean_accuracy) ** 2 for x in subject_means) / len(subject_means)
    
    # Calculate the standard deviation of the mean accuracies
    std_deviation_across_subjects = variance ** 0.5
    
    # Prepare data to be saved to file
    lines_to_save = [
        f"{subject_id} average accuracy: {result[0]:.4f}  standard deviation across 5 folds: {result[1]:.4f}"
        for subject_id, result in subject_results.items()
    ]
    lines_to_save.append(f"Overall average accuracy across all subjects: {total_mean_accuracy:.4f}")
    lines_to_save.append(f"Standard deviation across all subjects: {std_deviation_across_subjects:.4f}")
    
    # Construct the output file path
    output_dir = os.path.dirname(result_file_path)
    output_file_path = os.path.join(output_dir, 'averages_' + os.path.basename(result_file_path)[8:])
    
    # Save the results to a file
    with open(output_file_path, 'w') as output_file:
        for line in lines_to_save:
            output_file.write(line + '\n')
    
    print(f"Statistics saved to {output_file_path}") 