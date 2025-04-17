import torch
from eeg_model import EEGAnglePredictionModel
from dataset_utils import EEGDataset, preprocess_data
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from tqdm import tqdm


def compute_channel_importance(model, dataloader):
    """
    Compute the importance of each original EEG channel by tracing attention
    weights from both layers back to the input channels.
    
    Args:
        model: Trained EEG model
        dataloader: DataLoader with EEG data
        
    Returns:
        Dictionary with channel importance for each layer and combined
    """
    # Make sure IMAGES directory exists
    os.makedirs("IMAGES", exist_ok=True)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("Computing channel importance...")
    
    # Get convolutional weights
    conv1_weights = model.conv1.weight.data.cpu().numpy()  # [32, 64, kernel_size]
    conv2_weights = model.conv2.weight.data.cpu().numpy()  # [64, 32, kernel_size]
    
    # Hooks for attention layers
    first_layer_attn = []
    second_layer_attn = []
    
    # Hook functions
    def first_layer_hook(module, input, output):
        b, c, _ = input[0].shape
        avg_out = module.mlp(module.avg_pool(input[0]).view(b, c))
        max_out = module.mlp(module.max_pool(input[0]).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).detach().cpu().numpy()
        first_layer_attn.append(attn)
    
    def second_layer_hook(module, input, output):
        b, c, _ = input[0].shape
        avg_out = module.mlp(module.avg_pool(input[0]).view(b, c))
        max_out = module.mlp(module.max_pool(input[0]).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).detach().cpu().numpy()
        second_layer_attn.append(attn)
    
    # Register hooks
    hook1 = model.ca1.register_forward_hook(first_layer_hook)
    hook2 = model.ca2.register_forward_hook(second_layer_hook)
    
    # Process batches
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Processing batches"):
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Remove hooks
    hook1.remove()
    hook2.remove()
    
    # Average attention weights across all samples
    avg_attn_layer1 = np.mean(np.concatenate(first_layer_attn, axis=0), axis=0)  # [32]
    avg_attn_layer2 = np.mean(np.concatenate(second_layer_attn, axis=0), axis=0)  # [64]
    
    # Trace first layer attention back to input channels
    first_layer_importance = np.zeros(64)
    for i in range(64):  # For each input channel
        for j in range(32):  # For each feature map in first layer
            # Sum over kernel dimension
            weight_contribution = np.sum(np.abs(conv1_weights[j, i, :]))
            first_layer_importance[i] += weight_contribution * avg_attn_layer1[j]
    
    # Normalize first layer importance
    first_layer_importance = first_layer_importance / np.max(first_layer_importance)
    
    # Trace second layer attention through both convolutional layers
    second_layer_importance = np.zeros(64)
    
    # First, map second layer attention to first layer feature maps
    first_layer_feature_importance = np.zeros(32)
    for i in range(32):  # For each feature map in first layer
        for j in range(64):  # For each feature map in second layer
            # Sum over kernel dimension
            weight_contribution = np.sum(np.abs(conv2_weights[j, i, :]))
            first_layer_feature_importance[i] += weight_contribution * avg_attn_layer2[j]
    
    # Then, map first layer feature importance to input channels
    for i in range(64):  # For each input channel
        for j in range(32):  # For each feature map in first layer
            # Sum over kernel dimension
            weight_contribution = np.sum(np.abs(conv1_weights[j, i, :]))
            second_layer_importance[i] += weight_contribution * first_layer_feature_importance[j]
    
    # Normalize second layer importance
    second_layer_importance = second_layer_importance / np.max(second_layer_importance)
    
    # Combine both layers (equal weighting)
    combined_importance = (first_layer_importance + second_layer_importance) / 2
    
    # Create ranking (descending order of importance)
    combined_ranking = np.argsort(combined_importance)[::-1]
    
    # Create visualizations
    create_visualizations(first_layer_importance, second_layer_importance, combined_importance, combined_ranking)
    
    return {
        'first_layer': first_layer_importance,
        'second_layer': second_layer_importance,
        'combined': combined_importance,
        'ranking': combined_ranking
    }


def create_visualizations(first_layer_importance, second_layer_importance, combined_importance, ranking):
    """
    Create clear visualizations of channel importance.
    
    Args:
        first_layer_importance: Importance scores from first layer
        second_layer_importance: Importance scores from second layer
        combined_importance: Combined importance scores
        ranking: Ranking of channels by importance
    """
    # 1. Bar chart of combined importance with top 10 channels highlighted
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(64), combined_importance, alpha=0.7)
    
    # Highlight top 10 channels
    top_channels = ranking[:10]
    for idx in top_channels:
        bars[idx].set_color('red')
        bars[idx].set_alpha(1.0)
    
    # Add labels for top 5 channels
    for idx in ranking[:5]:
        plt.text(idx, combined_importance[idx] + 0.03, f"Ch {idx}", 
                 ha='center', fontweight='bold')
    
    plt.xlabel('EEG Channel Index', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title('Combined Channel Importance (Top 10 in Red)', fontsize=14)
    plt.tight_layout()
    plt.savefig('IMAGES/combined_channel_importance.png')
    
    # 2. Comparison of importance across layers
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Channel': range(64),
        'First Layer': first_layer_importance,
        'Second Layer': second_layer_importance,
        'Combined': combined_importance
    })
    
    # Melt for seaborn
    df_melted = pd.melt(df, id_vars=['Channel'], 
                        value_vars=['First Layer', 'Second Layer', 'Combined'],
                        var_name='Layer', value_name='Importance')
    
    # Plot
    sns.lineplot(data=df_melted, x='Channel', y='Importance', hue='Layer', marker='o')
    
    # Highlight top 5 channels with text
    for idx in ranking[:5]:
        plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
        plt.text(idx, 1.05, f"Ch {idx}", ha='center', fontweight='bold')
    
    plt.xlabel('EEG Channel Index', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title('Channel Importance Across Different Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig('IMAGES/layer_comparison.png')
    
    # 3. Circular visualization of channel importance
    plt.figure(figsize=(10, 10))
    
    # Create a circle of points
    angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plot points with size and color based on importance
    sizes = 100 + 900 * combined_importance
    scatter = plt.scatter(x, y, s=sizes, c=combined_importance, cmap='hot')
    
    # Add channel numbers for top channels
    for i in ranking[:10]:
        plt.text(x[i]*1.1, y[i]*1.1, str(i), ha='center', va='center', 
                fontweight='bold', fontsize=12)
    
    plt.axis('equal')
    plt.axis('off')
    plt.title('EEG Channel Importance (Top 10 Labeled)', fontsize=14)
    plt.colorbar(scatter, label='Importance Score')
    plt.savefig('IMAGES/channel_importance_circular.png')
    
    # Print top channels
    print("\nTop 10 most important EEG channels:")
    for i, ch in enumerate(ranking[:10]):
        print(f"{i+1}. Channel {ch}: {combined_importance[ch]:.4f}")


def main(model_path='eeg_angle_prediction_model.pth', data_folder='DATA'):
    """
    Main function to run the channel importance analysis.
    
    Args:
        model_path: Path to the trained model
        data_folder: Path to the data folder
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGAnglePredictionModel()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model for demonstration")
    
    model = model.to(device)
    
    # Load data for analysis
    try:
        # Try to load some data for analysis
        # Adjust this based on your actual data loading code
        data_list, angle_list = preprocess_data(
            sub_id='1',  # Use first subject
            freq_bands=[[1, 50]],
            data_folder=data_folder,
            fs_new=128
        )
        
        # Create dataset and dataloader
        dataset = EEGDataset(data_list, angle_list)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Run the analysis
        results = compute_channel_importance(model, dataloader)
        
        print("\nAnalysis complete. Results saved in IMAGES directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
