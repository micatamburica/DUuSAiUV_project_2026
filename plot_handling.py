import os
import matplotlib.pyplot as plt

def histogram_steering_angle(df_sequences):
    """Histogram of steering angle distribution."""
    
    if not os.path.exists('histograms'):
        os.makedirs('histograms')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df_sequences['steering_angle'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Steering Angle', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('histograms/steering_angle.png', dpi=300, bbox_inches='tight')
    plt.show()


def graph_training_history(history):
    """Graphing of the training history to visualize the training and validation loss and MAE over epochs."""

    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    plt.figure(figsize=(14, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    plt.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)

    plt.title('Model Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], 'b-', linewidth=2, label='Training MAE')
    plt.plot(history.history['val_mae'], 'r-', linewidth=2, label='Validation MAE')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)

    plt.title('Mean Absolute Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()   
    
    plt.savefig('graphs/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def graph_original_data(df):
    """"Graphing of the original data for analysis purposes."""

    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    # Multiple line plot with different y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(df['steering_angle'], label='Steering Angle', linewidth=1)
    ax1.plot(df['throttle'], 'darkgreen', label='Throttle', linewidth=1)
    ax1.plot(df['reverse'], 'darkred', label='Reverse', linewidth=1)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Throttle / Reverse / Steering Angle')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(df['speed'], 'purple', label='Speed', linewidth=1)
    ax2.set_ylabel('Speed')
    ax2.legend(loc='upper right')

    plt.title('Multiple Metrics Over Frames')
    plt.tight_layout()
    plt.show()

    fig.savefig('graphs/all.png', dpi=300, bbox_inches='tight')

    # Steering Angle
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['steering_angle'], linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Steering Angle')
    ax.grid(True)

    ax.set_title('Steering Angle')
    plt.tight_layout()
    plt.show()

    fig.savefig('graphs/steering_angle.png', dpi=300, bbox_inches='tight')

    # Throttle
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['throttle'], color='darkgreen', linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Throttle')
    ax.grid(True)

    ax.set_title('Throttle')
    plt.tight_layout()
    plt.show()

    fig.savefig('graphs/throttle.png', dpi=300, bbox_inches='tight')

    # Reverse
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['reverse'], color='darkred', linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reverse')
    ax.grid(True)

    ax.set_title('Reverse')
    plt.tight_layout()
    plt.show()

    fig.savefig('graphs/reverse.png', dpi=300, bbox_inches='tight')

    # Speed
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['speed'], color='purple', linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Speed')
    ax.grid(True)

    ax.set_title('Speed')
    plt.tight_layout()
    plt.show()

    fig.savefig('graphs/speed.png', dpi=300, bbox_inches='tight')
