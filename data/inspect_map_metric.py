
import os 
import torch

from training.noc_dataset import NocDataset

import matplotlib.pyplot as plt

def plot_execution_time_distributions(execution_time_dict, plot_path, plot_name):
    """
    Plots separate distribution graphs for execution times for each directory in subplots.
    """
    num_dirs = len(execution_time_dict)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_dirs + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    for i, (dir_index, execution_times) in enumerate(execution_time_dict.items()):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        # Flatten execution times for the current directory
        flattened_times = [time.item() for time in execution_times]

        # Plot histogram for the current directory
        ax.hist(flattened_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f"Dir {dir_index}")
        ax.set_xlabel('Execution Time')
        ax.set_ylabel('Frequency')

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis('off')

    # Adjust layout and save the plot
    plt.tight_layout()
    save_path = os.path.join(plot_path, plot_name)
    plt.savefig(save_path)
    print(f"Saved execution time distribution plots to {save_path}")

if __name__ == "__main__"  : 

    
    map_test_directory = "data/training_data/simulator/map_test"
    num_dirs = len(os.listdir(map_test_directory))

    execution_time_dict = {}

    for i in range(num_dirs): 
        dir_path    = os.path.join(map_test_directory, f"{i}")
        dataset     = NocDataset(dir_path)

        execution_time_list     = []
        execution_time_dict[i]  = {}

        for data in dataset: 
            max_index       = torch.argmax(data["task"].y[:, 1])
            execution_time  = data["task"].y[max_index, 1]

            execution_time_list.append(execution_time)

        execution_time_dict[i] = execution_time_list

    plot_save_path = "data/training_data/simulator"
    plot_execution_time_distributions(execution_time_dict, plot_save_path, "map_test_distribution.png")




