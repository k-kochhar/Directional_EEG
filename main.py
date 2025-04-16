import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

from eeg_model import EEGAnglePredictionModel
from dataset_utils import EEGDataset, preprocess_data


def train_model(model, train_loader, valid_loader, epochs=100, lr=0.0001):
    """
    Train the EEG angle prediction model.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Mean Squared Error loss for regression
    criterion = nn.MSELoss()
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_mae': [],
        'valid_mae': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - targets)).item()
        
        # Average metrics over batches
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_mae = 0.0
        
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Track metrics
                valid_loss += loss.item()
                valid_mae += torch.mean(torch.abs(outputs - targets)).item()
        
        # Average metrics over batches
        valid_loss /= len(valid_loader)
        valid_mae /= len(valid_loader)
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_mae'].append(train_mae)
        history['valid_mae'].append(valid_mae)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Valid Loss: {valid_loss:.4f} | '
              f'Train MAE: {train_mae:.2f}° | '
              f'Valid MAE: {valid_mae:.2f}°')
    
    return model, history


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data and compute metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Lists to store predictions and true values
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and true values
            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_true = np.array(all_true).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)
    
    # Also calculate discrete accuracy (sign prediction)
    sign_accuracy = np.mean((np.sign(all_preds) == np.sign(all_true)))
    
    return {
        'mae': mae,
        'r2': r2,
        'sign_accuracy': sign_accuracy,
        'predictions': all_preds,
        'true_values': all_true
    }


def visualize_results(history, evaluation_results):
    """
    Create visualizations of training history and model performance.
    
    Args:
        history: Dictionary with training history
        evaluation_results: Dictionary with evaluation metrics
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot training and validation loss
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['valid_loss'], label='Validation Loss')
    ax1.set_title('Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot training and validation MAE
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history['train_mae'], label='Train MAE')
    ax2.plot(history['valid_mae'], label='Validation MAE')
    ax2.set_title('Mean Absolute Error During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (degrees)')
    ax2.legend()
    
    # Plot predicted vs true angles
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(evaluation_results['true_values'], evaluation_results['predictions'], alpha=0.5)
    # Add diagonal line (perfect predictions)
    min_val = min(np.min(evaluation_results['true_values']), np.min(evaluation_results['predictions']))
    max_val = max(np.max(evaluation_results['true_values']), np.max(evaluation_results['predictions']))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax3.set_title('Predicted vs True Angles')
    ax3.set_xlabel('True Angle (degrees)')
    ax3.set_ylabel('Predicted Angle (degrees)')
    ax3.text(0.05, 0.95, f'MAE: {evaluation_results["mae"]:.2f}°\nR²: {evaluation_results["r2"]:.2f}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top')
    
    # Plot error distribution
    ax4 = fig.add_subplot(2, 2, 4)
    errors = evaluation_results['predictions'] - evaluation_results['true_values']
    sns.histplot(errors, kde=True, ax=ax4)
    ax4.set_title('Error Distribution')
    ax4.set_xlabel('Prediction Error (degrees)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=0, color='r', linestyle='--')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('eeg_angle_prediction_results.png')
    plt.close()
    
    # Create a heatmap to visualize angle classification accuracy
    # Group predictions by true angle
    df = pd.DataFrame({
        'true_angle': evaluation_results['true_values'].round().astype(int),
        'pred_angle': evaluation_results['predictions'].round().astype(int)
    })
    
    # Create confusion matrix for angle classification
    unique_angles = sorted(list(set(df['true_angle'].unique()) | set(df['pred_angle'].unique())))
    confusion_matrix = pd.DataFrame(0, index=unique_angles, columns=unique_angles)
    
    for true_angle, pred_angle in zip(df['true_angle'], df['pred_angle']):
        confusion_matrix.loc[true_angle, pred_angle] += 1
    
    # Normalize by row
    confusion_matrix_norm = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix_norm, annot=False, cmap='viridis', 
                xticklabels=5, yticklabels=5)
    plt.title('Angle Classification Confusion Matrix')
    plt.xlabel('Predicted Angle')
    plt.ylabel('True Angle')
    plt.savefig('eeg_angle_confusion_matrix.png')
    plt.close()
    
    # Create summary plot for each direction (left vs right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left direction (negative angles)
    left_mask = evaluation_results['true_values'] < 0
    if np.any(left_mask):
        left_true = np.abs(evaluation_results['true_values'][left_mask])
        left_pred = np.abs(evaluation_results['predictions'][left_mask])
        axes[0].scatter(left_true, left_pred, alpha=0.5, c='blue')
        axes[0].plot([0, max(left_true)], [0, max(left_true)], 'r--')
        axes[0].set_title('Left Direction (Negative Angles)')
        axes[0].set_xlabel('True Angle Magnitude (degrees)')
        axes[0].set_ylabel('Predicted Angle Magnitude (degrees)')
        left_mae = mean_absolute_error(left_true, left_pred)
        axes[0].text(0.05, 0.95, f'MAE: {left_mae:.2f}°', 
                    transform=axes[0].transAxes, fontsize=10, verticalalignment='top')
    
    # Right direction (positive angles)
    right_mask = evaluation_results['true_values'] > 0
    if np.any(right_mask):
        right_true = evaluation_results['true_values'][right_mask]
        right_pred = evaluation_results['predictions'][right_mask]
        axes[1].scatter(right_true, right_pred, alpha=0.5, c='green')
        axes[1].plot([0, max(right_true)], [0, max(right_true)], 'r--')
        axes[1].set_title('Right Direction (Positive Angles)')
        axes[1].set_xlabel('True Angle (degrees)')
        axes[1].set_ylabel('Predicted Angle (degrees)')
        right_mae = mean_absolute_error(right_true, right_pred)
        axes[1].text(0.05, 0.95, f'MAE: {right_mae:.2f}°', 
                    transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('eeg_angle_direction_performance.png')
    plt.close()
    
    # Print summary metrics
    print("\n===== Model Evaluation =====")
    print(f"Mean Absolute Error: {evaluation_results['mae']:.2f}°")
    print(f"R² Score: {evaluation_results['r2']:.4f}")
    print(f"Direction Accuracy: {evaluation_results['sign_accuracy']*100:.2f}%")
    print("============================\n")


def plot_attention_maps(model, sample_data, sample_label, output_file='attention_maps.png'):
    """
    Visualize the attention maps for a sample input.
    
    Args:
        model: Trained model
        sample_data: Sample EEG data
        sample_label: True angle for the sample
        output_file: Output file path
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Ensure sample_data is a tensor with batch dimension
    if not isinstance(sample_data, torch.Tensor):
        sample_data = torch.tensor(sample_data, dtype=torch.float32)
    if sample_data.dim() == 2:
        sample_data = sample_data.unsqueeze(0)  # Add batch dimension
    sample_data = sample_data.to(device)
    
    # Create hooks to extract attention maps
    channel_attention_maps = []
    spatial_attention_maps = []
    
    class ChannelHook:
        def __init__(self):
            self.activations = None
        def __call__(self, module, input, output):
            # Store channel attention weights
            b, c, _ = input[0].size()
            squeeze = module.avg_pool(input[0]).view(b, c)
            avg_out = module.mlp(squeeze)
            max_out = module.mlp(module.max_pool(input[0]).view(b, c))
            self.activations = torch.sigmoid(avg_out + max_out).view(b, c, 1).detach().cpu().numpy()
    
    class SpatialHook:
        def __init__(self):
            self.activations = None
        def __call__(self, module, input, output):
            # Store spatial attention weights
            avg_out = torch.mean(input[0], dim=1, keepdim=True)
            max_out, _ = torch.max(input[0], dim=1, keepdim=True)
            attn = torch.cat([avg_out, max_out], dim=1)
            self.activations = module.sigmoid(module.conv(attn)).detach().cpu().numpy()
    
    # Register hooks
    channel_hook1 = ChannelHook()
    channel_hook2 = ChannelHook()
    spatial_hook1 = SpatialHook()
    spatial_hook2 = SpatialHook()
    
    # Attach hooks
    hook1 = model.ca1.register_forward_hook(channel_hook1)
    hook2 = model.ca2.register_forward_hook(channel_hook2)
    hook3 = model.sa1.register_forward_hook(spatial_hook1)
    hook4 = model.sa2.register_forward_hook(spatial_hook2)
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_data)
    
    # Remove hooks
    hook1.remove()
    hook2.remove()
    hook3.remove()
    hook4.remove()
    
    # Create visualization
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])
    
    # Plot channel attention maps
    ax1 = fig.add_subplot(gs[0, 0])
    channel_attn1 = channel_hook1.activations[0, :, 0]
    ax1.bar(range(len(channel_attn1)), channel_attn1)
    ax1.set_title('First Layer Channel Attention')
    ax1.set_xlabel('Channel Index')
    ax1.set_ylabel('Attention Weight')
    
    ax2 = fig.add_subplot(gs[0, 1])
    channel_attn2 = channel_hook2.activations[0, :, 0]
    ax2.bar(range(len(channel_attn2)), channel_attn2)
    ax2.set_title('Second Layer Channel Attention')
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Attention Weight')
    
    # Plot spatial attention maps
    ax3 = fig.add_subplot(gs[1, 0])
    spatial_attn1 = spatial_hook1.activations[0, 0, :]
    ax3.plot(spatial_attn1)
    ax3.set_title('First Layer Spatial Attention')
    ax3.set_xlabel('Time Point')
    ax3.set_ylabel('Attention Weight')
    
    ax4 = fig.add_subplot(gs[1, 1])
    spatial_attn2 = spatial_hook2.activations[0, 0, :]
    ax4.plot(spatial_attn2)
    ax4.set_title('Second Layer Spatial Attention')
    ax4.set_xlabel('Time Point')
    ax4.set_ylabel('Attention Weight')
    
    # Plot input EEG data with top channels highlighted
    ax5 = fig.add_subplot(gs[2, :])
    input_data = sample_data[0].cpu().numpy()
    
    # Get top 5 channels based on attention
    top_channels = np.argsort(channel_attn1)[-5:]
    
    # Plot all channels in gray
    for i in range(input_data.shape[0]):
        if i in top_channels:
            continue
        ax5.plot(input_data[i], color='gray', alpha=0.2)
    
    # Plot top channels with higher opacity and different colors
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for idx, channel in enumerate(top_channels):
        ax5.plot(input_data[channel], color=colors[idx % len(colors)], label=f'Channel {channel}')
    
    ax5.set_title(f'Input EEG Data (True Angle: {sample_label:.1f}°, Predicted: {output.item():.1f}°)')
    ax5.set_xlabel('Time Point')
    ax5.set_ylabel('Amplitude')
    ax5.legend()
    
    # Show spatial attention as a shaded area
    max_val = np.max(np.abs(input_data))
    for i in range(len(spatial_attn1)):
        if spatial_attn1[i] > 0.7:  # Highlight high attention areas
            ax5.axvspan(i, i+1, alpha=0.3, color='yellow')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def generate_subject_level_results(model, all_subjects_data, all_subjects_angles):
    """
    Generate and visualize results for each subject separately.
    
    Args:
        model: Trained model
        all_subjects_data: Dictionary with subject data
        all_subjects_angles: Dictionary with subject angles
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Dictionary to store results for each subject
    subject_results = {}
    
    # Create figure for subject comparison
    plt.figure(figsize=(12, 8))
    
    # Process each subject
    for subject_id, subject_data in all_subjects_data.items():
        # Skip subjects with no data
        if len(subject_data) == 0:
            print(f"Skipping subject {subject_id} - no data available")
            continue
            
        subject_angles = all_subjects_angles[subject_id]
        
        # Convert to PyTorch tensors if not already
        data_tensor = []
        for d in subject_data:
            if isinstance(d, torch.Tensor):
                data_tensor.append(d)
            else:
                data_tensor.append(torch.tensor(d.astype(np.float32)).transpose(0, 1))
        
        angle_tensor = []
        for a in subject_angles:
            if isinstance(a, torch.Tensor):
                angle_tensor.append(a)
            else:
                angle_tensor.append(torch.tensor([a], dtype=torch.float32))
        
        # Create dataset and dataloader
        dataset = EEGDataset(data_tensor, angle_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Evaluate model on this subject
        evaluation = evaluate_model(model, dataloader)
        subject_results[subject_id] = evaluation
        
        # Plot data point for this subject
        plt.scatter(subject_id, evaluation['mae'], label=f'Subject {subject_id}')
    
    # Only save the plot if we have results
    if subject_results:
        # Finalize and save subject comparison plot
        plt.title('Model Performance Across Subjects')
        plt.xlabel('Subject ID')
        plt.ylabel('Mean Absolute Error (degrees)')
        plt.grid(True)
        plt.axhline(y=np.mean([res['mae'] for res in subject_results.values()]), color='r', linestyle='--', 
                    label=f'Average MAE: {np.mean([res["mae"] for res in subject_results.values()]):.2f}°')
        plt.legend()
        plt.tight_layout()
        plt.savefig('subject_comparison.png')
        plt.close()
    else:
        print("No subject-level results to plot")
    
    return subject_results


def main():
    # Model and training parameters
    input_channels = 64
    sample_length = 128  # 1 second at 128 Hz
    batch_size = 32
    epochs = 5  # Reduced from 100 for testing
    lr = 0.0001
    data_folder = "DATA"  # Change if needed
    
    # Create model
    model = EEGAnglePredictionModel(
        input_channels=input_channels, 
        sample_length=sample_length
    )
    
    # Lists to store data and angles for all subjects
    all_data = []
    all_angles = []
    
    # Dictionary to store data per subject for later analysis
    all_subjects_data = {}
    all_subjects_angles = {}
    
    # Process data for all subjects
    for subject_id in range(1, 21):  # Subjects 1-20
        sub_id = str(subject_id)
        
        try:
            # Load and preprocess data
            data, angles = preprocess_data(
                sub_id=sub_id,
                freq_bands=[[1, 50]],
                data_folder=data_folder,
                fs_new=128
            )
            
            # Store per subject
            all_subjects_data[sub_id] = data
            all_subjects_angles[sub_id] = angles
            
            # Add to combined lists
            all_data.extend(data)
            all_angles.extend(angles)
            
            print(f"Subject {sub_id}: {len(data)} trials loaded")
            
        except Exception as e:
            print(f"Error processing subject {sub_id}: {e}")
    
    # Split data into training, validation, and test sets (70% train, 15% validation, 15% test)
    train_ratio = 0.7
    valid_ratio = 0.15
    train_size = int(len(all_data) * train_ratio)
    valid_size = int(len(all_data) * valid_ratio)
    
    # Shuffle data while keeping same indices for data and angles
    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]
    all_angles = [all_angles[i] for i in indices]
    
    # Create PyTorch datasets
    train_dataset = EEGDataset(all_data[:train_size], all_angles[:train_size])
    valid_dataset = EEGDataset(all_data[train_size:train_size+valid_size], all_angles[train_size:train_size+valid_size])
    test_dataset = EEGDataset(all_data[train_size+valid_size:], all_angles[train_size+valid_size:])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
        lr=lr
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(model, test_loader)
    
    # Visualize results
    visualize_results(history, evaluation_results)
    
    # Get a sample for attention map visualization
    sample_data, sample_label = test_dataset[0]
    
    # Plot attention maps for a sample
    plot_attention_maps(model, sample_data, sample_label.item())
    
    # Generate subject-level results
    subject_results = generate_subject_level_results(model, all_subjects_data, all_subjects_angles)
    
    # Save model
    torch.save(model.state_dict(), 'eeg_angle_prediction_model.pth')
    print("Model saved to 'eeg_angle_prediction_model.pth'")
    
    return model, history, evaluation_results, subject_results


if __name__ == "__main__":
    main() 