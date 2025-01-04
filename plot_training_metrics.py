import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

try:
    # Read the CSV file
    df = pd.read_csv('lightning_logs/version_0/metrics.csv')

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Training and Validation Accuracy
    ax1 = plt.subplot(2, 2, 1)
    train_acc = df[['epoch', 'train_acc_epoch']].dropna()
    val_acc = df[['epoch', 'val_acc']].dropna()
    ax1.plot(train_acc['epoch'], train_acc['train_acc_epoch'], label='Train Accuracy', marker='o')
    ax1.plot(val_acc['epoch'], val_acc['val_acc'], label='Validation Accuracy', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Validation Accuracy over Time')
    ax1.legend()
    ax1.grid(True)

    # 2. Training and Validation Loss
    ax2 = plt.subplot(2, 2, 2)
    train_loss = df[['epoch', 'train_loss_epoch']].dropna()
    val_loss = df[['epoch', 'val_loss']].dropna()
    ax2.plot(train_loss['epoch'], train_loss['train_loss_epoch'], label='Train Loss', marker='o')
    ax2.plot(val_loss['epoch'], val_loss['val_loss'], label='Validation Loss', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss over Time')
    ax2.legend()
    ax2.grid(True)

    # 3. Learning Rate Schedule
    ax3 = plt.subplot(2, 2, 3)
    lr_data = df[['step', 'lr-SGD']].dropna()
    ax3.plot(lr_data['step'], lr_data['lr-SGD'], label='Learning Rate', color='green')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule (One Cycle Policy)')
    ax3.grid(True)
    # Set x-axis to show all steps using the actual data
    ax3.set_xlim(0, lr_data['step'].max())  # Dynamically get the max step

    # 4. Training Time per Epoch
    ax4 = plt.subplot(2, 2, 4)
    train_time = df['train_epoch_time'].dropna()
    val_time = df['val_epoch_time'].dropna()
    epochs_train = range(len(train_time))
    epochs_val = range(len(val_time))
    ax4.plot(epochs_train, train_time, label='Training Time', marker='s')
    ax4.plot(epochs_val, val_time, label='Validation Time', marker='s')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Training and Validation Time per Epoch')
    ax4.legend()
    ax4.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate plot for step-wise training metrics
    plt.figure(figsize=(15, 6))
    step_acc = df[['step', 'train_acc_step']].dropna()
    plt.plot(step_acc['step'], step_acc['train_acc_step'], label='Training Accuracy', alpha=0.6)
    plt.title('Training Accuracy per Step')
    plt.xlabel('Step')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    # Set x-axis to show all steps
    plt.xlim(0, step_acc['step'].max())  # Dynamically get the max step
    plt.savefig('training_steps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Final Training Accuracy: {train_acc['train_acc_epoch'].iloc[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_acc['val_acc'].iloc[-1]:.2f}%")
    print(f"Best Validation Accuracy: {val_acc['val_acc'].max():.2f}%")
    print(f"Average Training Time per Epoch: {train_time.mean():.2f} seconds")
    print(f"Total Training Time: {train_time.sum()/3600:.2f} hours")

except FileNotFoundError:
    print("Error: Could not find the metrics.csv file in lightning_logs/version_0/")
except Exception as e:
    print(f"An error occurred: {str(e)}") 
