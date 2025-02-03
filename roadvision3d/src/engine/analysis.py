import os
import json
import matplotlib.pyplot as plt

def load_training_data(json_file):
    """Load the training data from the JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_lr_and_loss_data(training_data):
    """Extract learning rate, overall loss, and individual losses over epochs."""
    epoch_lr_map = {}
    epoch_overall_loss_map = {}
    epoch_individual_loss_map = {}
    loss_names = set()

    for entry in training_data:
        epoch = entry['epoch']
        lr = entry.get('lr', None)
        loss_data = entry['data']

        # Keep track of all possible loss names
        loss_names.update(loss_data.keys())

        # Calculate overall loss as the sum of all individual losses
        overall_loss = sum(loss_data.values())

        # Store learning rate per epoch
        if lr is not None:
            if epoch not in epoch_lr_map:
                epoch_lr_map[epoch] = []
            epoch_lr_map[epoch].append(lr)
        
        # Store overall loss per epoch
        if epoch not in epoch_overall_loss_map:
            epoch_overall_loss_map[epoch] = []
        epoch_overall_loss_map[epoch].append(overall_loss)

        # Store individual losses per epoch
        if epoch not in epoch_individual_loss_map:
            epoch_individual_loss_map[epoch] = {}
            for loss_name in loss_data.keys():
                epoch_individual_loss_map[epoch][loss_name] = []

        for loss_name, loss_value in loss_data.items():
            epoch_individual_loss_map[epoch][loss_name].append(loss_value)
    
    # Average learning rate and losses per epoch
    lr_per_epoch = {epoch: sum(lrs) / len(lrs) for epoch, lrs in epoch_lr_map.items()}
    overall_loss_per_epoch = {epoch: sum(losses) / len(losses) for epoch, losses in epoch_overall_loss_map.items()}

    # Average individual losses per epoch
    individual_losses_per_epoch = {}
    for epoch in epoch_individual_loss_map:
        individual_losses_per_epoch[epoch] = {}
        for loss_name in loss_names:
            losses = epoch_individual_loss_map[epoch].get(loss_name, [])
            if losses:
                avg_loss = sum(losses) / len(losses)
            else:
                avg_loss = 0  # You can choose to set this to None if preferred
            individual_losses_per_epoch[epoch][loss_name] = avg_loss

    return lr_per_epoch, overall_loss_per_epoch, individual_losses_per_epoch, loss_names

def plot_lr_evolution(lr_per_epoch, output_path):
    """Plot the learning rate evolution over epochs and save as a PNG file."""
    epochs = sorted(lr_per_epoch.keys())
    lr_values = [lr_per_epoch[epoch] for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_values, linestyle='-', color='b', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Learning Rate Evolution Over Epochs (Log Scale)')
    plt.grid(True, which="both", ls="--")  # Grid lines for both major and minor ticks
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_loss_evolution(loss_per_epoch, output_path):
    """Plot the overall loss evolution over epochs and save as a PNG file."""
    epochs = sorted(loss_per_epoch.keys())
    loss_values = [loss_per_epoch[epoch] for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, linestyle='-', color='r', label='Overall Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Overall Loss')
    plt.title('Overall Loss Evolution Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_individual_losses(individual_losses_per_epoch, loss_names, output_dir):
    """Plot each individual loss over epochs and save each as a PNG file."""
    epochs = sorted(individual_losses_per_epoch.keys())

    for loss_name in loss_names:
        loss_values = [individual_losses_per_epoch[epoch][loss_name] for epoch in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_values, linestyle='-', label=loss_name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title(f'{loss_name} Evolution Over Epochs')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # Construct the output file path
        output_path = os.path.join(output_dir, f'{loss_name}_evolution.png')
        plt.savefig(output_path)
        plt.close()

def generate_and_save_plots(json_file):
    """Generate and save plots for learning rate and losses from the JSON log file."""
    # Load and process training data
    training_data = load_training_data(json_file)
    lr_per_epoch, overall_loss_per_epoch, individual_losses_per_epoch, loss_names = extract_lr_and_loss_data(training_data)
    
    # Get the directory of the JSON file
    base_dir = os.path.dirname(json_file)

    # Define the new directory for plots
    output_dir = os.path.join(base_dir, 'plots')

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot and save Learning Rate Evolution
    lr_plot_path = os.path.join(output_dir, 'learning_rate_evolution.png')
    plot_lr_evolution(lr_per_epoch, lr_plot_path)
    
    # Plot and save Overall Loss Evolution
    loss_plot_path = os.path.join(output_dir, 'overall_loss_evolution.png')
    plot_loss_evolution(overall_loss_per_epoch, loss_plot_path)
    
    # Plot and save Individual Losses
    plot_individual_losses(individual_losses_per_epoch, loss_names, output_dir)