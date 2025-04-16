import torch
from eeg_model import EEGAnglePredictionModel
from dataset_utils import EEGDataset, preprocess_data
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def compute_channel_importance(model, dataloader, layer='all'):
    """
    Compute the importance of each channel based on attention weights.
    
    Args:
        model: Trained EEG model
        dataloader: DataLoader with EEG data
        layer: Which attention layer to analyze ('first', 'second', or 'all')
        
    Returns:
        Dictionary with channel importance weights for each layer
    """
    # Make sure IMAGES directory exists
    os.makedirs("IMAGES", exist_ok=True)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Hooks for first and second attention layers
    first_layer_weights = []
    second_layer_weights = []

    class FirstLayerHook:
        def __init__(self):
            self.weights = []

        def __call__(self, module, input, output):
            b, c, _ = input[0].shape
            avg_out = module.mlp(module.avg_pool(input[0]).view(b, c))
            max_out = module.mlp(module.max_pool(input[0]).view(b, c))
            attn_weights = torch.sigmoid(avg_out + max_out).detach().cpu().numpy()
            self.weights.append(attn_weights)

    class SecondLayerHook:
        def __init__(self):
            self.weights = []

        def __call__(self, module, input, output):
            b, c, _ = input[0].shape
            avg_out = module.mlp(module.avg_pool(input[0]).view(b, c))
            max_out = module.mlp(module.max_pool(input[0]).view(b, c))
            attn_weights = torch.sigmoid(avg_out + max_out).detach().cpu().numpy()
            self.weights.append(attn_weights)

    # Register hooks
    first_hook = FirstLayerHook()
    second_hook = SecondLayerHook()
    
    if layer in ['first', 'all']:
        hook1 = model.ca1.register_forward_hook(first_hook)
    if layer in ['second', 'all']:
        hook2 = model.ca2.register_forward_hook(second_hook)

    # Process data through the model
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            model(x)
    
    # Remove hooks
    if layer in ['first', 'all']:
        hook1.remove()
    if layer in ['second', 'all']:
        hook2.remove()
    
    # Process weights
    results = {}
    
    if layer in ['first', 'all']:
        if first_hook.weights:
            first_weights = np.concatenate(first_hook.weights, axis=0)  # [N, C]
            first_avg_weights = np.mean(first_weights, axis=0)  # [C]
            results['first_layer'] = first_avg_weights
    
    if layer in ['second', 'all']:
        if second_hook.weights:
            second_weights = np.concatenate(second_hook.weights, axis=0)  # [N, C]
            second_avg_weights = np.mean(second_weights, axis=0)  # [C]
            results['second_layer'] = second_avg_weights
    
    return results


def visualize_channel_importance(weights, top_n=10, output_file='channel_attention_importance.png', 
                                title='Channel Importance Based on Attention', channel_names=None, 
                                figsize=(18, 8)):
    """
    Visualize the importance of each channel with highlighting for top channels.
    
    Args:
        weights: NumPy array of channel importance weights
        top_n: Number of top channels to highlight
        output_file: Path to save the output figure
        title: Title for the plot
        channel_names: Optional list of channel names (if None, uses channel indices)
        figsize: Figure size (width, height)
    """
    # Make sure IMAGES directory exists
    os.makedirs("IMAGES", exist_ok=True)
    
    # Update the output file path to save in the IMAGES folder
    output_file = os.path.join("IMAGES", output_file)
    
    # Get channel indices and top channels
    channel_indices = np.arange(len(weights))
    top_channels = np.argsort(weights)[-top_n:][::-1]  # Sort in descending order
    
    # Create a DataFrame for better visualization
    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(len(weights))]
    
    df = pd.DataFrame({
        'Channel': channel_names,
        'Importance': weights,
        'Rank': np.zeros(len(weights))
    })
    
    # Mark top channels
    for i, channel_idx in enumerate(top_channels):
        df.loc[channel_idx, 'Rank'] = i + 1
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bar plot
    ax = plt.subplot(111)
    bars = plt.bar(channel_indices, weights, color='lightgray')
    
    # Highlight top channels
    colors = plt.cm.rainbow(np.linspace(0, 1, top_n))
    for i, channel_idx in enumerate(top_channels):
        bars[channel_idx].set_color(colors[i])
    
    # Add labels and title
    plt.xlabel('Channel Index', fontsize=12)
    plt.ylabel('Average Attention Weight', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Adjust x-ticks
    if len(weights) > 32:
        plt.xticks(np.arange(0, len(weights), 4))
    else:
        plt.xticks(channel_indices)
    
    # Add a table with top channel information
    top_channels_info = df.iloc[top_channels].reset_index()
    table_data = []
    for i, row in top_channels_info.iterrows():
        table_data.append([f"{int(row['index'])}", f"{row['Channel']}", f"{row['Importance']:.4f}"])
    
    # Add table with top channels
    table = plt.table(cellText=table_data,
                     colLabels=['Index', 'Channel', 'Importance'],
                     colWidths=[0.1, 0.3, 0.2],
                     loc='right', bbox=[1.02, 0.1, 0.3, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved channel importance visualization to {output_file}")
    
    # Also return the DataFrame for further analysis
    return df


def visualize_all_layers_importance(importance_dict, top_n=10, 
                                   output_file='all_layers_channel_importance.png',
                                   channel_names=None):
    """
    Visualize channel importance across all layers.
    
    Args:
        importance_dict: Dictionary with channel importance for each layer
        top_n: Number of top channels to highlight
        output_file: Path to save the output figure
        channel_names: Optional list of channel names
    """
    # Make sure IMAGES directory exists
    os.makedirs("IMAGES", exist_ok=True)
    
    # Update the output file path to save in the IMAGES folder
    output_file = os.path.join("IMAGES", output_file)
    
    if not importance_dict:
        print("No importance data available")
        return
        
    n_layers = len(importance_dict)
    fig, axes = plt.subplots(n_layers, 1, figsize=(18, 6*n_layers))
    
    if n_layers == 1:
        axes = [axes]
    
    for i, (layer_name, weights) in enumerate(importance_dict.items()):
        # Get channel indices and top channels
        channel_indices = np.arange(len(weights))
        top_channels = np.argsort(weights)[-top_n:][::-1]
        
        # Create bar plot
        bars = axes[i].bar(channel_indices, weights, color='lightgray')
        
        # Highlight top channels
        colors = plt.cm.rainbow(np.linspace(0, 1, top_n))
        for j, channel_idx in enumerate(top_channels):
            bars[channel_idx].set_color(colors[j])
        
        # Add titles and labels
        axes[i].set_title(f'{layer_name.replace("_", " ").title()} Channel Importance', fontsize=14)
        axes[i].set_xlabel('Channel Index', fontsize=12)
        axes[i].set_ylabel('Attention Weight', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Adjust x-ticks
        if len(weights) > 32:
            axes[i].set_xticks(np.arange(0, len(weights), 4))
        else:
            axes[i].set_xticks(channel_indices)
        
        # Add text labels for top channels
        for j, channel_idx in enumerate(top_channels):
            axes[i].annotate(f"{channel_idx}",
                            xy=(channel_idx, weights[channel_idx]),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-layer channel importance visualization to {output_file}")


def create_channel_heatmap(importance_dict, output_file='channel_heatmap.png', channel_names=None):
    """
    Create a heatmap showing importance of channels across different layers.
    
    Args:
        importance_dict: Dictionary with channel importance for each layer
        output_file: Path to save the output figure
        channel_names: Optional list of channel names
    """
    # Make sure IMAGES directory exists
    os.makedirs("IMAGES", exist_ok=True)
    
    # Update the output file path to save in the IMAGES folder
    output_file = os.path.join("IMAGES", output_file)
    
    if len(importance_dict) < 2:
        print("Need at least two layers for heatmap comparison")
        return
    
    # Ensure all arrays have the same length
    max_length = max(len(weights) for weights in importance_dict.values())
    aligned_dict = {}
    
    for layer_name, weights in importance_dict.items():
        if len(weights) < max_length:
            # If this layer has fewer channels, pad with zeros
            print(f"Warning: {layer_name} has {len(weights)} channels, padding to {max_length}")
            padded_weights = np.zeros(max_length)
            padded_weights[:len(weights)] = weights
            aligned_dict[layer_name] = padded_weights
        else:
            aligned_dict[layer_name] = weights
    
    # Create DataFrame with importance scores for each layer
    data = pd.DataFrame(aligned_dict)
    
    # If channel names provided, use them as index
    if channel_names is not None:
        if len(channel_names) >= max_length:
            data.index = channel_names[:max_length]
        else:
            # Pad channel names if needed
            extended_names = channel_names + [f'Channel {i}' for i in range(len(channel_names), max_length)]
            data.index = extended_names
    
    # Create heatmap
    plt.figure(figsize=(10, 14))
    ax = sns.heatmap(data, cmap='viridis', annot=False, cbar_kws={'label': 'Importance'})
    
    plt.title('Channel Importance Across Layers', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved channel importance heatmap to {output_file}")


def analyze_top_channels_per_condition(model, all_data, all_angles, top_n=10):
    """
    Analyze which channels are most important for different angle conditions.
    
    Args:
        model: Trained model
        all_data: List of all data segments
        all_angles: List of all angle labels
        top_n: Number of top channels to report
        
    Returns:
        Dictionary with top channels for each condition
    """
    try:
        # Group data by angle
        angle_groups = {}
        for data, angle in zip(all_data, all_angles):
            # Extract scalar angle value
            if isinstance(angle, torch.Tensor):
                angle_val = angle.item() if angle.numel() == 1 else angle[0].item()
            else:
                angle_val = angle
            
            # Round to nearest whole angle for grouping
            angle_val = round(angle_val)
            
            if angle_val not in angle_groups:
                angle_groups[angle_val] = []
            angle_groups[angle_val].append(data)
        
        # Process each angle group
        results = {}
        for angle, data_list in angle_groups.items():
            # Skip if too few samples
            if len(data_list) < 5:
                print(f"Skipping angle {angle}° - only {len(data_list)} samples")
                continue
                
            try:
                # Create dataset and dataloader
                angle_dataset = EEGDataset(data_list, [angle] * len(data_list))
                angle_loader = DataLoader(angle_dataset, batch_size=min(16, len(data_list)))
                
                # Compute channel importance
                importance = compute_channel_importance(model, angle_loader)
                
                # Store top channels for this angle
                if 'first_layer' in importance:
                    top_channels = np.argsort(importance['first_layer'])[-top_n:][::-1]
                    results[angle] = {
                        'top_indices': top_channels.tolist(),
                        'weights': importance['first_layer'][top_channels].tolist()
                    }
            except Exception as e:
                print(f"Error analyzing angle {angle}°: {e}")
        
        return results
    except Exception as e:
        print(f"Error in angle condition analysis: {e}")
        return {}


def generate_comprehensive_report(model_path='eeg_angle_prediction_model.pth', data_folder='DATA', top_n=10):
    """
    Generate a comprehensive report on channel importance.
    
    Args:
        model_path: Path to the saved model
        data_folder: Path to the data folder
        top_n: Number of top channels to highlight
    """
    try:
        # Make sure IMAGES directory exists
        os.makedirs("IMAGES", exist_ok=True)
        
        # Load model
        model = EEGAnglePredictionModel(input_channels=64, sample_length=128)
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
        
        # Load some data for analysis (using subject 1 and 2)
        all_data = []
        all_angles = []
        
        for sub_id in ['1', '2']:
            try:
                data, angles = preprocess_data(sub_id, data_folder=data_folder)
                all_data.extend(data)
                all_angles.extend(angles)
                print(f"Loaded {len(data)} samples from subject {sub_id}")
            except Exception as e:
                print(f"Error loading data for subject {sub_id}: {e}")
        
        if not all_data:
            print("No data loaded. Cannot perform analysis.")
            return {}, {}
            
        # Create dataset and dataloader
        dataset = EEGDataset(all_data, all_angles)
        dataloader = DataLoader(dataset, batch_size=32)
        
        # Compute channel importance for all layers
        print(f"\nComputing channel importance across all {len(all_data)} samples...")
        importance = compute_channel_importance(model, dataloader, layer='all')
        
        # Visualize first layer importance
        if 'first_layer' in importance:
            weights = importance['first_layer']
            print(f"\n=== First Layer Channel Importance (All {len(weights)} Channels) ===")
            
            # Sort channels by importance
            sorted_indices = np.argsort(weights)[::-1]
            top_channels = sorted_indices[:top_n]
            
            # Print detailed information for all channels
            print(f"Channel importance values (first layer):")
            for i, idx in enumerate(sorted_indices):
                rank = i + 1
                if rank <= top_n:
                    print(f"  {rank}. Channel {idx}: {weights[idx]:.4f} ★")
                else:
                    print(f"  {rank}. Channel {idx}: {weights[idx]:.4f}")
            
            # Create visualization
            df = visualize_channel_importance(
                weights, 
                top_n=top_n,
                output_file='first_layer_importance.png',
                title='First Layer Channel Importance'
            )
        
        # Visualize second layer importance
        if 'second_layer' in importance:
            weights = importance['second_layer']
            print(f"\n=== Second Layer Channel Importance (All {len(weights)} Channels) ===")
            
            # Sort channels by importance
            sorted_indices = np.argsort(weights)[::-1]
            top_channels = sorted_indices[:top_n]
            
            # Print detailed information for all channels
            print(f"Channel importance values (second layer):")
            for i, idx in enumerate(sorted_indices):
                rank = i + 1
                if rank <= top_n:
                    print(f"  {rank}. Channel {idx}: {weights[idx]:.4f} ★")
                else:
                    print(f"  {rank}. Channel {idx}: {weights[idx]:.4f}")
            
            # Create visualization
            df = visualize_channel_importance(
                weights, 
                top_n=top_n,
                output_file='second_layer_importance.png',
                title='Second Layer Channel Importance'
            )
        
        # Create multi-layer visualization if both layers are available
        if len(importance) > 1:
            try:
                visualize_all_layers_importance(importance, top_n=top_n)
                create_channel_heatmap(importance)
            except Exception as e:
                print(f"Error creating multi-layer visualizations: {e}")
        
        # Analyze top channels for different angle conditions
        print("\n=== Channel Importance by Angle Condition ===")
        angle_results = analyze_top_channels_per_condition(model, all_data, all_angles, top_n=5)
        
        # Print results for angle conditions
        if angle_results:
            print("\nTop channels for different angle conditions:")
            for angle, result in angle_results.items():
                print(f"\nAngle {angle}° - Top 5 channels:")
                for i, (idx, weight) in enumerate(zip(result['top_indices'], result['weights'])):
                    print(f"  {i+1}. Channel {idx}: {weight:.4f}")
        else:
            print("No angle-specific results available.")
        
        return importance, angle_results
    
    except Exception as e:
        print(f"Error in comprehensive report generation: {e}")
        return {}, {}


if __name__ == "__main__":
    print("=" * 50)
    print("EEG Channel Attention Analysis")
    print("=" * 50)
    
    # Make sure IMAGES directory exists
    os.makedirs("IMAGES", exist_ok=True)
    
    # Generate comprehensive report
    importance, angle_results = generate_comprehensive_report(model_path="eeg_angle_prediction_model.pth", top_n=10)
    
    print("\nAnalysis complete. Check the generated visualization files in the IMAGES folder.")
    print("=" * 50)
